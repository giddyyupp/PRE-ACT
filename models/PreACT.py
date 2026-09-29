import torch
import torch.nn as nn

from transformers import AutoModel, AutoProcessor
from transformers import VideoMAEModel
from transformers import XCLIPModel

from models.common import (
    _freeze_module,
    _resolve_module,
    _unfreeze_module,
)


class VJEPA2Anticipation(nn.Module):
    def __init__(
        self,
        backbone_name: str = "facebook/vjepa2-vitl-fpc16-256",
        freeze_backbone: bool = False,
        unfreeze_last_n_blocks: int = 0,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        pool: str = "mean",   # "mean" or "cls"
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.pool = pool

        hidden_size = self.backbone.config.hidden_size

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if unfreeze_last_n_blocks > 0:
            self.unfreeze_last_blocks(unfreeze_last_n_blocks)

        # self.common_head = nn.Sequential(
        #     nn.LayerNorm(hidden_size),
        #     nn.Linear(hidden_size, hidden_dim),
        #     nn.GELU(),
        # )

        # self.risk_head = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, 1),
        # )

        # self.prog_head = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, 1),
        # )

        self.risk_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.prog_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.pref_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def unfreeze_last_blocks(self, n: int, unfreeze_final_norm: bool = True) -> None:
        """
        Unfreeze the last n transformer blocks of a V-JEPA 2 backbone loaded with
        AutoModel.from_pretrained(...).

        This tries a few common internal layouts and also unfreezes the final norm
        if present.
        """
        # Freeze everything first so partial unfreezing is deterministic
        for p in self.backbone.parameters():
            p.requires_grad = False

        blocks = None
        final_norm = None

        # Common possible layouts
        candidate_block_paths = [
            ("encoder", "layer"),
            ("encoder", "layers"),
            ("vision_model", "encoder", "layer"),
            ("vision_model", "encoder", "layers"),
            ("model", "encoder", "layer"),
            ("model", "encoder", "layers"),
            ("backbone", "encoder", "layer"),
            ("backbone", "encoder", "layers"),
            ("blocks",),
        ]

        candidate_norm_paths = [
            ("layernorm",),
            ("norm",),
            ("encoder", "layernorm"),
            ("encoder", "norm"),
            ("vision_model", "post_layernorm"),
            ("vision_model", "layernorm"),
            ("vision_model", "norm"),
            ("model", "layernorm"),
            ("model", "norm"),
        ]

        # Resolve block list
        for path in candidate_block_paths:
            obj = self.backbone
            ok = True
            for name in path:
                if hasattr(obj, name):
                    obj = getattr(obj, name)
                else:
                    ok = False
                    break
            if ok:
                blocks = obj
                break

        if blocks is None:
            raise ValueError(
                "Could not find V-JEPA 2 transformer blocks. "
                "Please print model and inspect self.backbone structure."
            )

        # Resolve final norm if available
        for path in candidate_norm_paths:
            obj = self.backbone
            ok = True
            for name in path:
                if hasattr(obj, name):
                    obj = getattr(obj, name)
                else:
                    ok = False
                    break
            if ok:
                final_norm = obj
                break

        # Convert to an indexable sequence if possible
        if not hasattr(blocks, "__len__"):
            raise ValueError(
                f"Found blocks object at type {type(blocks)}, but it is not indexable."
            )

        n = min(n, len(blocks))

        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True

        if unfreeze_final_norm and final_norm is not None:
            for p in final_norm.parameters():
                p.requires_grad = True

    def forward(self, frames: torch.Tensor):
        """
        frames: [B, T, C, H, W]
        """
        # frames: [B, T, C, H, W]
        outputs = self.backbone(pixel_values_videos=frames)
        feats = outputs.last_hidden_state        # notebook example returns [B, N, D]
        # e.g. torch.Size([1, 9216, 1408]) for vit-g example
        if self.pool == "cls":
            video_feat = feats[:, 0]
        else:
            video_feat = feats.mean(dim=1)

        # interim_features = self.common_head(video_feat)
        risk_logit = self.risk_head(video_feat).squeeze(-1)
        progress_logit = self.prog_head(video_feat).squeeze(-1)
        preference_logit = self.pref_head(video_feat).squeeze(-1)
        return {"risk_logit": risk_logit, "progress_logit": progress_logit, "preference_logit": preference_logit}


class XCLIPAnticipation(nn.Module):
    def __init__(
        self,
        backbone_name: str = "microsoft/xclip-base-patch32",
        freeze_backbone: bool = False,
        unfreeze_last_n_vision_blocks: int = 0,
        unfreeze_mit: bool = False,
        unfreeze_prompts_generator: bool = False,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = XCLIPModel.from_pretrained(backbone_name)
        feat_dim = self.backbone.config.projection_dim

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if unfreeze_last_n_vision_blocks > 0:
            self.unfreeze_last_vision_blocks(unfreeze_last_n_vision_blocks)

        if unfreeze_mit:
            self.unfreeze_mit()

        if unfreeze_prompts_generator:
            self.unfreeze_prompts_generator()

        self.risk_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.prog_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.pref_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        
    
    def unfreeze_last_vision_blocks(self, n: int) -> None:
        if not hasattr(self.backbone, "vision_model"):
            raise ValueError("Could not find vision_model in XCLIPModel.")

        vision_model = self.backbone.vision_model

        if not hasattr(vision_model, "encoder") or not hasattr(vision_model.encoder, "layers"):
            raise ValueError("Could not find vision encoder layers in XCLIP vision_model.")

        blocks = vision_model.encoder.layers
        n = min(n, len(blocks))

        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True

        if hasattr(vision_model, "post_layernorm"):
            for p in vision_model.post_layernorm.parameters():
                p.requires_grad = True

    def unfreeze_mit(self) -> None:
        if hasattr(self.backbone, "mit"):
            for p in self.backbone.mit.parameters():
                p.requires_grad = True
        else:
            raise ValueError("Could not find MIT module (`backbone.mit`) in XCLIPModel.")

    def unfreeze_prompts_generator(self) -> None:
        if hasattr(self.backbone, "prompts_generator"):
            for p in self.backbone.prompts_generator.parameters():
                p.requires_grad = True
        else:
            raise ValueError("Could not find prompts_generator in XCLIPModel.")

    def forward(self, frames: torch.Tensor):
        """
        frames: [B, T, C, H, W]
        For xclip-base-patch32, pad your 5-frame clips to 8 in the collate fn.
        """
        video_out = self.backbone.get_video_features(pixel_values=frames)

        if hasattr(video_out, "pooler_output") and video_out.pooler_output is not None:
            video_embeds = video_out.pooler_output
        elif hasattr(video_out, "last_hidden_state") and video_out.last_hidden_state is not None:
            video_embeds = video_out.last_hidden_state[:, 0]
        elif torch.is_tensor(video_out):
            video_embeds = video_out
        else:
            raise TypeError(f"Unexpected output type from get_video_features: {type(video_out)}")

        risk_logit = self.risk_head(video_embeds).squeeze(-1)
        progress_logit = self.prog_head(video_embeds).squeeze(-1)
        preference_logit = self.pref_head(video_embeds).squeeze(-1)
        return {"risk_logit": risk_logit, "progress_logit": progress_logit, "preference_logit": preference_logit}


class TemporalTransformerHead(nn.Module):
    def __init__(self, in_dim, model_dim=256, num_heads=4, num_layers=2, dropout=0.1, max_len=32):
        super().__init__()
        self.proj = nn.Linear(in_dim, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, model_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        x: [B, T, D]
        """
        B, T, _ = x.shape
        x = self.proj(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, T+1, model_dim]
        x = x + self.pos_embed[:, : T + 1]

        x = self.encoder(x)
        cls_out = x[:, 0]  # [B, model_dim]
        risk_logit = self.head(cls_out).squeeze(-1)
        return risk_logit


class DINOv2TransformerAnticipation(nn.Module):
    def __init__(
            self,
            backbone_name="facebook/dinov2-large", 
            freeze_backbone=False,
            unfreeze_last_n_blocks=0,
            transformer_dim=256,
            transformer_heads=4,
            transformer_layers=2,
            transformer_dropout=0.1,
            max_len=32,
            ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        feat_dim = self.backbone.config.hidden_size

        self.temporal_head = TemporalTransformerHead(
            in_dim=feat_dim,
            model_dim=transformer_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            dropout=transformer_dropout,
            max_len=max_len,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if unfreeze_last_n_blocks > 0:
            self.unfreeze_last_blocks(unfreeze_last_n_blocks)

    def unfreeze_last_blocks(self, n: int):
        """
        Unfreeze the last n DINOv2 encoder blocks and final norm.
        """
        for p in self.backbone.parameters():
            p.requires_grad = False

        # common HF DINOv2 layouts
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            blocks = self.backbone.encoder.layer
            final_norm = getattr(self.backbone, "layernorm", None)
        elif hasattr(self.backbone, "dinov2") and hasattr(self.backbone.dinov2, "encoder") and hasattr(self.backbone.dinov2.encoder, "layer"):
            blocks = self.backbone.dinov2.encoder.layer
            final_norm = getattr(self.backbone.dinov2, "layernorm", None)
        else:
            raise ValueError("Could not find DINOv2 encoder blocks for selective unfreezing.")

        n = min(n, len(blocks))

        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True

        if final_norm is not None:
            for p in final_norm.parameters():
                p.requires_grad = True

    def encode_frames(self, frames):
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)
        out = self.backbone(pixel_values=x)
        feats = out.last_hidden_state[:, 0]
        feats = feats.view(B, T, -1)
        return feats

    def forward(self, frames):
        feats = self.encode_frames(frames)
        risk_logit = self.temporal_head(feats)
        return {"risk_logit": risk_logit}


class DINOv2GRUAnticipation(nn.Module):
    def __init__(self, 
                 backbone_name="facebook/dinov2-large", 
                 hidden_dim=768, 
                 gru_dim=512, 
                 freeze_backbone=False, 
                 unfreeze_last_n_blocks=0):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.feat_dim = self.backbone.config.hidden_size

        self.temporal = nn.GRU(
            input_size=self.feat_dim,
            hidden_size=gru_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.risk_head = nn.Sequential(
            nn.LayerNorm(gru_dim),
            nn.Linear(gru_dim, gru_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(gru_dim, 1),
        )

        self.prog_head = nn.Sequential(
            nn.LayerNorm(gru_dim),
            nn.Linear(gru_dim, gru_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(gru_dim, 1),
        )

        self.pref_head = nn.Sequential(
            nn.LayerNorm(gru_dim),
            nn.Linear(gru_dim, gru_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(gru_dim, 1),
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if unfreeze_last_n_blocks > 0:
            self.unfreeze_last_blocks(unfreeze_last_n_blocks)

    def unfreeze_last_blocks(self, n: int):
        """
        Unfreeze the last n encoder blocks and final norm of DINOv2.
        """
        # keep everything frozen first
        for p in self.backbone.parameters():
            p.requires_grad = False

        # HF DINOv2 layout
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            blocks = self.backbone.encoder.layer
        elif hasattr(self.backbone, "dinov2") and hasattr(self.backbone.dinov2.encoder, "layer"):
            blocks = self.backbone.dinov2.encoder.layer
        else:
            raise ValueError("Could not find DINOv2 encoder blocks for selective unfreezing.")

        n = min(n, len(blocks))

        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True

        # also unfreeze final norm if present
        if hasattr(self.backbone, "layernorm"):
            for p in self.backbone.layernorm.parameters():
                p.requires_grad = True

        if hasattr(self.backbone, "dinov2") and hasattr(self.backbone.dinov2, "layernorm"):
            for p in self.backbone.dinov2.layernorm.parameters():
                p.requires_grad = True

    def encode_frames(self, frames):
        # frames: [B, T, C, H, W]
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)

        out = self.backbone(pixel_values=x)
        feats = out.last_hidden_state[:, 0]  # CLS token
        feats = feats.view(B, T, -1)
        return feats

    def forward(self, frames):
        feats = self.encode_frames(frames)         # [B, T, D]
        _, h = self.temporal(feats)                # h: [1, B, H]
        h = h[-1]                                  # [B, H]
        # interim_features = self.common_head(h)
        risk_logit = self.risk_head(h).squeeze(-1)
        progress_logit = self.prog_head(h).squeeze(-1)
        preference_logit = self.pref_head(h).squeeze(-1)
        return {"risk_logit": risk_logit, "progress_logit": progress_logit, "preference_logit": preference_logit}


class VideoMAEAnticipation_v4(nn.Module):
    def __init__(
        self,
        backbone_name="MCG-NJU/videomae-large",
        num_classes=1,
        freeze_backbone=False,
        unfreeze_last_n_blocks=0,
        unfreeze_final_norm=True,
    ):
        super().__init__()
        self.backbone = VideoMAEModel.from_pretrained(backbone_name)
        hidden_dim = self.backbone.config.hidden_size

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if unfreeze_last_n_blocks > 0:
            self.unfreeze_last_blocks(
                n=unfreeze_last_n_blocks,
                unfreeze_final_norm=unfreeze_final_norm,
            )

        self.risk_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

        self.prog_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

        self.pref_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def unfreeze_last_blocks(self, n: int, unfreeze_final_norm: bool = True):
        """
        Unfreeze the last n encoder blocks of VideoMAE.
        Assumes the backbone is already frozen if you want partial unfreezing.
        """
        if not hasattr(self.backbone, "encoder") or not hasattr(self.backbone.encoder, "layer"):
            raise ValueError("Could not find VideoMAE encoder blocks at backbone.encoder.layer")

        blocks = self.backbone.encoder.layer
        n = min(n, len(blocks))

        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True

        if unfreeze_final_norm and hasattr(self.backbone, "layernorm") and self.backbone.layernorm is not None: # final guard is for kinetics finetuned models
            for p in self.backbone.layernorm.parameters():
                p.requires_grad = True

    def forward(self, frames):
        # frames: [B, T, C, H, W]
        out = self.backbone(pixel_values=frames)
        feats = out.last_hidden_state[:, 0]
        risk_logit = self.risk_head(feats).squeeze(-1)
        progress_logit = self.prog_head(feats).squeeze(-1)
        preference_logit = self.pref_head(feats).squeeze(-1)
        return {"risk_logit": risk_logit, "progress_logit": progress_logit, "preference_logit": preference_logit}


class RiskPropHead(nn.Module):
    def __init__(
        self,
        in_channels=2048,
        num_classes=1,
        dropout=0.4,
        init_std=0.01,
    ):
        super().__init__()

        self.avg_pool2d = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc_cls = nn.Linear(in_channels, num_classes)

        # RiskProp / MMAction initialization
        nn.init.normal_(self.fc_cls.weight, mean=0.0, std=init_std)
        nn.init.constant_(self.fc_cls.bias, 0.0)

    def forward(self, x):
        """
        x: [B, C, T, H, W]
        e.g. [B, 2048, 5, 7, 7]
        """

        # RiskProp:
        # x = self.avg_pool2d(x).squeeze(-1).squeeze(-1)
        #
        # AdaptiveAvgPool2d works on the final H,W dimensions and
        # preserves B,C,T.
        x = self.avg_pool2d(x).squeeze(-1).squeeze(-1)
        # [B, C, T]

        # with_decoder=False:
        x = x.mean(dim=-1)
        # [B, C]

        x = self.dropout(x)
        x = self.fc_cls(x)
        # [B, 1]

        return x.squeeze(-1)


class X3DAnticipation(nn.Module):
    """
    X3D anticipation model using the pretrained PyTorchVideo X3D-M backbone.

    Expected input shape: [B, T, C, H, W].
    X3D itself receives [B, C, T, H, W].
    For the standard pretrained X3D-M checkpoint, use 16 x 224 x 224 clips.
    """

    def __init__(
        self,
        model_name: str = "x3d_m",
        freeze_backbone: bool = False,
        unfreeze_last_n_stages: int = 0,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
        num_classes: int = 1,
        pretrained: bool = True,
        hub_repo: str = "facebookresearch/pytorchvideo",
    ):
        super().__init__()

        supported_models = {"x3d_xs", "x3d_s", "x3d_m", "x3d_l"}
        if model_name not in supported_models:
            raise ValueError(
                f"Unsupported X3D model '{model_name}'. "
                f"Choose one of {sorted(supported_models)}."
            )

        self.model_name = model_name
        self.backbone = torch.hub.load(
            hub_repo,
            model_name,
            pretrained=pretrained,
        )

        # PyTorchVideo X3D uses a 2048-D representation immediately before
        # the Kinetics classifier. Keeping the complete X3D head preserves
        # its projected pooling; only the final classifier is removed.
        x3d_head = self.backbone.blocks[-1]
        if not hasattr(x3d_head, "proj") or not isinstance(x3d_head.proj, nn.Linear):
            raise ValueError(
                "Unexpected PyTorchVideo X3D head structure: expected "
                "backbone.blocks[-1].proj to be nn.Linear."
            )

        feat_dim = x3d_head.proj.in_features
        x3d_head.proj = nn.Identity()
        x3d_head.activation = None

        self.risk_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.prog_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.pref_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        if freeze_backbone:
            _freeze_module(self.backbone)

        if unfreeze_last_n_stages > 0:
            self.unfreeze_last_stages(unfreeze_last_n_stages)

    def unfreeze_last_stages(
        self,
        n: int,
        unfreeze_x3d_head: bool = True,
    ) -> None:
        """
        Freeze the X3D backbone, then unfreeze its last ``n`` residual stages.

        PyTorchVideo X3D block layout:
          blocks[0]: stem
          blocks[1:5]: four residual stages
          blocks[5]: projected-pooling head
        """
        _freeze_module(self.backbone)

        stages = list(self.backbone.blocks[1:-1])
        n = min(max(n, 0), len(stages))

        if n > 0:
            for stage in stages[-n:]:
                _unfreeze_module(stage)

        if unfreeze_x3d_head:
            _unfreeze_module(self.backbone.blocks[-1])

    def forward(self, frames: torch.Tensor):
        """
        Args:
            frames: Tensor of shape [B, T, C, H, W].
        """
        if frames.ndim != 5:
            raise ValueError(
                f"Expected frames with shape [B, T, C, H, W], got {tuple(frames.shape)}"
            )

        x = frames.permute(0, 2, 1, 3, 4).contiguous()
        feat = self.backbone(x)

        # With the classifier replaced by Identity, X3D returns [B, 2048].
        if feat.ndim > 2:
            feat = feat.flatten(1)

        risk_logit = self.risk_head(feat)
        if risk_logit.shape[-1] == 1:
            risk_logit = risk_logit.squeeze(-1)

        progress_logit = self.prog_head(feat).squeeze(-1)
        preference_logit = self.pref_head(feat).squeeze(-1)

        return {
            "risk_logit": risk_logit,
            "progress_logit": progress_logit,
            "preference_logit": preference_logit,
            "feat": feat,
        }


class CosmosEmbedAnticipation(nn.Module):
    def __init__(
        self,
        backbone_name: str = "nvidia/Cosmos-Embed1-224p",
        freeze_backbone: bool = False,
        unfreeze_last_n_stages: int = 0,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        token: str | None = None,
    ):
        super().__init__()

        revision = "413fcdd76cadb9267702cfff66c24475b58595fb"

        self.backbone_name = backbone_name
        self.processor = AutoProcessor.from_pretrained(
            backbone_name,
            trust_remote_code=True,
            token=token,
            revision=revision,
        )
        self.backbone = AutoModel.from_pretrained(
            backbone_name,
            trust_remote_code=True,
            token=token,
            revision=revision,
        )

        print(type(self.backbone))
        print(self.backbone)

        print("\nTop-level children:")
        for name, module in self.backbone.named_children():
            print(name, type(module))

        print("\nParameter name samples:")
        for i, (name, _) in enumerate(self.backbone.named_parameters()):
            print(name)
            if i >= 80:
                break

        if "224p" in backbone_name:
            feat_dim = 256
        else:
            feat_dim = 768

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if unfreeze_last_n_stages > 0:
            self.unfreeze_last_vision_blocks(unfreeze_last_n_stages)

        # if unfreeze_last_n_qformer_blocks > 0:
        #     self.unfreeze_last_qformer_blocks(unfreeze_last_n_qformer_blocks)

        self.cls_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 1),
        )

        self.prog_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 1),
        )

    def unfreeze_last_vision_blocks(self, n: int):
        for p in self.backbone.parameters():
            p.requires_grad = False

        if not hasattr(self.backbone, "visual_encoder"):
            raise ValueError("Cosmos backbone has no visual_encoder")
        if not hasattr(self.backbone.visual_encoder, "blocks"):
            raise ValueError("Cosmos visual_encoder has no blocks")

        blocks = self.backbone.visual_encoder.blocks
        n = min(max(n, 0), len(blocks))

        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True

        for name, p in self.backbone.named_parameters():
            if name.startswith("visual_encoder.norm") or name.startswith("visual_encoder.fc_norm"):
                p.requires_grad = True

        print(f"Unfroze last {n} vision blocks from visual_encoder.blocks")
        
    def unfreeze_last_qformer_blocks(self, n: int, unfreeze_final_norm: bool = True):
        """
        Unfreeze the last n QFormer blocks.
        """
        qformer_candidates = [
            ("qformer", "encoder", "layer"),
            ("qformer", "encoder", "layers"),
            ("qformer", "bert", "encoder", "layer"),
            ("qformer", "bert", "encoder", "layers"),
            ("model", "qformer", "encoder", "layer"),
            ("model", "qformer", "bert", "encoder", "layer"),
        ]

        norm_candidates = [
            ("qformer", "layernorm"),
            ("qformer", "norm"),
            ("qformer", "bert", "encoder", "layernorm"),
            ("model", "qformer", "norm"),
        ]

        blocks, used_path = _resolve_module(self.backbone, qformer_candidates)
        if blocks is None or not hasattr(blocks, "__len__"):
            raise ValueError("Could not find Cosmos QFormer blocks. Print model.backbone and inspect names.")

        n = min(max(n, 0), len(blocks))
        for block in blocks[-n:]:
            _unfreeze_module(block)

        if unfreeze_final_norm:
            final_norm, _ = _resolve_module(self.backbone, norm_candidates)
            if final_norm is not None:
                _unfreeze_module(final_norm)

        print(f"Unfroze last {n} QFormer blocks from path: {'.'.join(used_path)}")

    def forward(self, frames: torch.Tensor):
        """
        frames: [B, T, C, H, W] in [0,1]
        Cosmos processor expects BTCHW.
        """
        if frames.dtype != torch.float32:
            frames = frames.float()

        frames = torch.clamp(frames, 0.0, 1.0)

        # Keep BTCHW
        videos = frames.detach().cpu().numpy()

        video_inputs = self.processor(videos=videos, return_tensors="pt")
        video_inputs = {k: v.to(frames.device) for k, v in video_inputs.items()}

        video_out = self.backbone.get_video_embeddings(**video_inputs)

        if hasattr(video_out, "visual_proj"):
            feat = video_out.visual_proj
        elif isinstance(video_out, torch.Tensor):
            feat = video_out
        else:
            raise ValueError("Unexpected Cosmos video embedding output format.")

        cls_logit = self.cls_head(feat).squeeze(-1)
        prog_logit = self.prog_head(feat).squeeze(-1)

        return {
            "risk_logit": cls_logit,
            "progress_logit": prog_logit,
            "feat": feat,
        }


class VaViMAnticipation(nn.Module):
    """
    VaViM backbone for PRE-ACT.

    Expected input:
        frames: [B, T, C, H, W]

    Uses:
        - VaViM checkpoint:
          width_1024_pretrained_139k_total_155k.pt

        - LlamaGen tokenizer:
          VQ_ds16_16384_llamagen_encoder.jit

    VaViM expects:
        RGB 288 x 512
        -> LlamaGen VQ tokenizer (downsample x16)
        -> 18 x 32 = 576 visual tokens per frame.
    """

    def __init__(
        self,
        backbone_path: str,
        tokenizer_path: str,
        freeze_backbone: bool = False,
        unfreeze_last_n_blocks: int = 0,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
        num_classes: int = 1,
        stop_layer_idx: int = 12,
        input_imagenet_normalized: bool = True,
    ):
        super().__init__()

        from vam.video_pretraining import load_pretrained_gpt

        # ---------------------------------------------------------
        # Load pretrained VaViM
        # ---------------------------------------------------------
        self.backbone = load_pretrained_gpt(
            backbone_path,
            device="cpu",
        )

        self.feat_dim = self.backbone.embedding_dim
        self.stop_layer_idx = stop_layer_idx
        self.input_imagenet_normalized = input_imagenet_normalized

        if stop_layer_idx >= len(self.backbone.transformer.h):
            raise ValueError(
                f"stop_layer_idx={stop_layer_idx}, but VaViM only has "
                f"{len(self.backbone.transformer.h)} blocks."
            )

        # ---------------------------------------------------------
        # LlamaGen VQ tokenizer
        # ---------------------------------------------------------
        device = torch.device("cuda", torch.cuda.current_device())

        self.tokenizer = torch.jit.load(
            tokenizer_path,
            map_location=device,
        )

        self.tokenizer.eval()
        self.tokenizer.float()
        # self.tokenizer.requires_grad_(False)
        for p in self.tokenizer.parameters():
            p.requires_grad = False

        # Current PRE-ACT dataloader uses ImageNet normalization.
        self.register_buffer(
            "imagenet_mean",
            torch.tensor(
                [0.485, 0.456, 0.406]
            ).view(1, 1, 3, 1, 1),
            persistent=False,
        )

        self.register_buffer(
            "imagenet_std",
            torch.tensor(
                [0.229, 0.224, 0.225]
            ).view(1, 1, 3, 1, 1),
            persistent=False,
        )

        # ---------------------------------------------------------
        # Backbone freezing / partial fine-tuning
        # ---------------------------------------------------------

        if freeze_backbone:
            self.backbone.requires_grad_(False)

        elif unfreeze_last_n_blocks > 0:

            # Freeze complete VaViM first.
            self.backbone.requires_grad_(False)

            # Only blocks up to stop_layer_idx are actually used.
            used_blocks = self.backbone.transformer.h[
                : self.stop_layer_idx + 1
            ]

            n = min(
                unfreeze_last_n_blocks,
                len(used_blocks),
            )

            for block in used_blocks[-n:]:
                block.requires_grad_(True)

            # Token/spatial/temporal embeddings remain frozen.

        else:
            # Fine-tune all blocks used to obtain the representation.
            self.backbone.requires_grad_(False)

            self.backbone.transformer.wie.requires_grad_(True)
            self.backbone.transformer.wse.requires_grad_(True)
            self.backbone.transformer.wte.requires_grad_(True)

            for block in self.backbone.transformer.h[
                : self.stop_layer_idx + 1
            ]:
                block.requires_grad_(True)

        # Tokenizer is ALWAYS frozen.
        # self.tokenizer.requires_grad_(False)

        # ---------------------------------------------------------
        # PRE-ACT heads
        # ---------------------------------------------------------

        self.prog_head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    # -------------------------------------------------------------
    # RGB -> LlamaGen visual tokens
    # -------------------------------------------------------------

    @torch.no_grad()
    def tokenize(self, frames):
        B, T, C, H, W = frames.shape

        # Explicit FP32 for tokenizer preprocessing
        x = frames.float()

        if self.input_imagenet_normalized:
            x = (
                x * self.imagenet_std.float()
                + self.imagenet_mean.float()
            )

        x = x.clamp(0.0, 1.0)

        x = x.reshape(B * T, C, H, W)

        x = torch.nn.functional.interpolate(
            x,
            size=(288, 512),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

        # [0, 1] -> [-1, 1]
        x = 2.0 * x - 1.0

        # IMPORTANT: VQ tokenizer must run outside AMP
        with torch.autocast(device_type=x.device.type, enabled=False):
            tokens = self.tokenizer(x.float())

        if tokens.ndim != 3:
            raise RuntimeError(
                f"Unexpected tokenizer output shape: {tuple(tokens.shape)}"
            )

        h, w = tokens.shape[-2:]

        if h * w != self.backbone.nb_tokens_per_timestep:
            raise RuntimeError(
                f"Tokenizer produced {h}x{w}={h*w} tokens, "
                f"but VaViM expects "
                f"{self.backbone.nb_tokens_per_timestep} tokens/frame."
            )

        tokens = tokens.reshape(B, T, h, w)

        return tokens.long()

    # -------------------------------------------------------------
    # Differentiable equivalent of get_intermediate_layers()
    # -------------------------------------------------------------

    def extract_vavim_features(self, visual_tokens):
        """
        visual_tokens:
            [B, T, H, W]

        returns:
            [B, T, H, W, D]

        This follows VaViM's get_intermediate_layers(), but without
        its @torch.no_grad() decorator so that selected transformer
        blocks can be fine-tuned.
        """

        B, T, H, W = visual_tokens.shape

        # VaViM has a maximum temporal context.
        if T > self.backbone.nb_timesteps:
            visual_tokens = visual_tokens[
                :, -self.backbone.nb_timesteps :
            ]

            B, T, H, W = visual_tokens.shape

        # [B,T,H,W] -> [B,T*H*W]
        token_sequence = visual_tokens.reshape(
            B,
            T * H * W,
        )

        device = token_sequence.device

        # ---------------------------------------------------------
        # Spatial positions
        # ---------------------------------------------------------
        spatial_positions = torch.arange(
            H * W,
            device=device,
        )

        spatial_positions = (
            spatial_positions
            .view(1, 1, H * W)
            .expand(B, T, H * W)
            .reshape(B, T * H * W)
        )

        # ---------------------------------------------------------
        # Temporal positions
        # ---------------------------------------------------------
        temporal_positions = torch.arange(
            T,
            device=device,
        )

        temporal_positions = (
            temporal_positions
            .view(1, T, 1)
            .expand(B, T, H * W)
            .reshape(B, T * H * W)
        )

        # ---------------------------------------------------------
        # Token + spatial + temporal embeddings
        # ---------------------------------------------------------
        x = self.backbone._get_emb(
            token_sequence,
            spatial_positions,
            temporal_positions,
        )

        seq_len = token_sequence.shape[1]

        # Causal attention exactly as in VaViM.
        attn_mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=device,
                dtype=torch.bool,
            )
        )

        # ---------------------------------------------------------
        # Transformer
        # ---------------------------------------------------------
        for layer_idx, block in enumerate(
            self.backbone.transformer.h
        ):

            x = block(
                x,
                attn_mask,
            )

            # Follow official get_intermediate_layers behavior:
            # stop immediately at requested layer.
            if layer_idx == self.stop_layer_idx:
                break

        # [B, T*H*W, D]
        # -> [B,T,H,W,D]
        x = x.reshape(
            B,
            T,
            H,
            W,
            self.feat_dim,
        )

        return x

    # -------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------

    def forward(self, frames):
        """
        frames:
            [B, T, C, H, W]
        """

        if frames.ndim != 5:
            raise ValueError(
                "Expected [B,T,C,H,W], got "
                f"{tuple(frames.shape)}"
            )

        # ---------------------------------------------------------
        # RGB -> visual tokens
        # ---------------------------------------------------------
        visual_tokens = self.tokenize(frames)

        # ---------------------------------------------------------
        # VaViM
        # ---------------------------------------------------------
        features = self.extract_vavim_features(
            visual_tokens
        )

        # features:
        # [B,T,18,32,D]

        # VaViM is causal. Therefore the representation of the last
        # observed frame contains information from previous frames.
        #
        # Official HummingBird evaluation also extracts x[:, -1].
        feat = features[:, -1].mean(dim=(1, 2))

        # feat: [B,D]

        # ---------------------------------------------------------
        # PRE-ACT heads
        # ---------------------------------------------------------

        progress_logit = self.prog_head(
            feat
        ).squeeze(-1)

        return {
            "risk_logit": None,
            "progress_logit": progress_logit,
            "preference_logit": None,
            "feat": feat,
        }