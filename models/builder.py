from argparse import Namespace

from models.PreACT import (
    CosmosEmbedAnticipation,
    DINOv2GRUAnticipation,
    DINOv2TransformerAnticipation,
    VJEPA2Anticipation,
    VideoMAEAnticipation_v4,
    X3DAnticipation,
    XCLIPAnticipation,
    VaViMAnticipation,
)


def build_model(args):
    if isinstance(args, dict):
        args = Namespace(**args)
    # print(args.root)
    if args.backbone_type == "dinov2":
        if args.transformer_head:
            model = DINOv2TransformerAnticipation(
                backbone_name=args.backbone_name,
                freeze_backbone=args.freeze_backbone,
                unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
                transformer_dim=args.transformer_dim,
                transformer_heads=args.transformer_heads,
                transformer_layers=args.transformer_layers,
                transformer_dropout=args.transformer_dropout,
                max_len=max(32, args.snippet_len + 4),
            )
        else:   
            model = DINOv2GRUAnticipation(
                backbone_name=args.backbone_name,
                gru_dim=args.temporal_hidden_dim,
                freeze_backbone=args.freeze_backbone,
                unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
        )
    elif args.backbone_type == "videomae":
        model = VideoMAEAnticipation_v4(
            backbone_name=args.backbone_name,
            num_classes=args.num_classes,
            freeze_backbone=args.freeze_backbone,
            unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
            unfreeze_final_norm=True,  # TODO: maybe add to args if needed.
        )
        
    elif args.backbone_type == "xclip":
        model = XCLIPAnticipation(
            backbone_name=args.backbone_name,
            freeze_backbone=args.freeze_backbone,
            unfreeze_last_n_vision_blocks=args.unfreeze_last_n_blocks,
            unfreeze_mit=True,
            hidden_dim=2048,
            dropout=0.1,
        )

    elif args.backbone_type == "vjepa2":
        model = VJEPA2Anticipation(
            backbone_name=args.backbone_name,
            freeze_backbone=args.freeze_backbone,
            unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
        )

    elif args.backbone_type == "cosmos":
        model = CosmosEmbedAnticipation(
            backbone_name=args.backbone_name,
            freeze_backbone=args.freeze_backbone,
            unfreeze_last_n_stages=args.unfreeze_last_n_blocks,
            hidden_dim=args.temporal_hidden_dim,
            token=getattr(args, "hf_token", None),
        )
    elif args.backbone_type == "x3d":
        model = X3DAnticipation(
            model_name="x3d_m",
            pretrained=True,
            freeze_backbone=args.freeze_backbone,
            hidden_dim=2048,
            dropout=0.1,
            num_classes=args.num_classes,
        )
    elif args.backbone_type == "vavim":
        model = VaViMAnticipation(
            backbone_path=args.backbone_name,
            tokenizer_path="./VQ_ds16_16384_llamagen_encoder.jit",
            freeze_backbone=args.freeze_backbone,
            unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
        )
    else:
        raise ValueError(args.backbone_type)
    return model
