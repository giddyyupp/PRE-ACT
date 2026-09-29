import os
import math
import random
import argparse
import json

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision import transforms
from tqdm import tqdm
import mlflow
import numpy as np
from contextlib import nullcontext
from functools import partial

from dataloaders.dataloader import MMAUConfig, MMAUAnticipationDataset
from dataloaders.dataloader_nexar import NexarConfig, NexarAnticipationDataset
from dataloaders.dataloader_dad import DADConfig, DADAnticipationDataset

from test_encoder_anticipation_ddp import run_binary_clip_eval

from losses import compute_losses, compute_losses_ordinal_ttc
from models import build_model, load_weights
from engine.collate import anticipation_eval_collate_fn_pad, anticipation_collate_fn_pad
from engine.optim import build_optimizer
from evaluation.run_mmau import evaluate_predictions as evaluate_predictions_mmau
from evaluation.run_nexar import evaluate_nexar_official as evaluate_predictions_nexar
from evaluation.run_dad import evaluate_predictions_dad as evaluate_predictions_dad
from dataloaders.transforms import ClipTransform
from evaluation.eval_utils import print_validation_metrics

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed():
    """Initialize DDP from torchrun environment variables.

    Launch with:
        torchrun --standalone --nproc_per_node=N train_autoencoder_anticipation.py ...
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed GPU training requested, but CUDA is unavailable.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

    return distributed, rank, local_rank, world_size


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def reduce_epoch_running(running, device, world_size):
    """Average accumulated loss sums across ranks."""
    if world_size == 1:
        return running

    keys = list(running.keys())
    values = torch.tensor([running[k] for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= world_size
    return {k: values[i].item() for i, k in enumerate(keys)}


def gather_eval_outputs(local_outputs, distributed, rank, world_size):
    """Gather variable-length prediction lists from all ranks onto rank 0."""
    if not distributed:
        return local_outputs

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_outputs)

    if rank != 0:
        return None

    merged = []
    for rank_outputs in gathered:
        merged.extend(rank_outputs)

    # Deterministic ordering makes saved JSONs easier to diff/reproduce.
    merged.sort(key=lambda x: (str(x.get("video_hashcode", "")), str(x.get("clip_names", ""))))
    return merged


def mlflow_validation_metrics(metrics):
    """Return finite scalar metrics with MLflow-safe names."""
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            value = float(value)
            if math.isfinite(value):
                safe_key = key.replace("@", "_at_").replace(" ", "_")
                out[f"val/{safe_key}"] = value
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="../../data/MM_AU")
    parser.add_argument("--metadata-json", type=str, default="../../data/MM_AU/video_metadata.json")
    parser.add_argument("--subset", type=str, default="CAP", choices=["CAP", "DADA", "Nexar", "DAD"])
    parser.add_argument("--backbone-type", type=str, default="videomae", choices=["dinov2", "videomae", "xclip", "vjepa2", "cosmos", "x3d", "vavim", "vipra"])
    parser.add_argument("--backbone-name", type=str, default="")
    parser.add_argument("--pretrained-model", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="./outputs/PRE-ACT_videomae")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--temporal-hidden-dim", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--use_augmentations", action="store_true", help="Clip-level RandomResizedCrop + horizontal flip during training. Off keeps the original resize-only transform.")
    parser.add_argument("--snippet-len", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train-stride", type=int, default=5)
    parser.add_argument("--anticipation-horizon-sec", type=float, default=2.0)
    parser.add_argument("--progress-alpha", type=float, default=1.0)
    parser.add_argument("--pair-gap-sec-min", type=float, default=0.5)
    parser.add_argument("--pair-gap-sec-max", type=float, default=1.5)
    parser.add_argument("--preference-random-pos-neg-sampling", action="store_true")
    parser.add_argument("--full_video_progress_risk", action="store_true")
    parser.add_argument("--no_bce_ablation", action="store_true")

    parser.add_argument("--lambda-bce", type=float, default=1.0)
    parser.add_argument("--lambda-prog", type=float, default=1.0)
    parser.add_argument("--lambda-pref", type=float, default=0.2)
    parser.add_argument("--num-classes", type=int, default=1, help="Number of classes for BCE loss. PRE-ACT uses 1.")

    parser.add_argument("--custom_risk_mode", type=str, default="mild")
    parser.add_argument("--transformer_head", action="store_true")
    parser.add_argument("--videomae_categorical", action="store_true")
    parser.add_argument("--pref-loss-type", type=str, default="logsigmoid", choices=["logsigmoid", "margin_ranking"])

    parser.add_argument("--transformer-dim", type=int, default=256)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)

    parser.add_argument("--unfreeze-last-n-blocks", type=int, default=0)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)

    parser.add_argument("--dataset-fraction", type=float, default=1.0, help="Fraction of the training dataset to use. Must be in (0, 1].")

    parser.add_argument("--eval-every-n-epochs", type=int, default=1,
                        help="Run validation at the end of every N epochs. Use <=0 to disable.")
    parser.add_argument("--eval-mode", type=str, default="sliding_window", choices=["binary_clips", "sliding_window"], help="Validation sampling mode. Use sliding_window to match run_mmau.py offline evaluation.")
    parser.add_argument("--eval-score-key", type=str, default="risk_score", choices=["score", "risk_score", "fused_score"], help="Prediction field consumed by run_mmau.evaluate_predictions.")
    parser.add_argument("--eval-anno-json", type=str, default=None, help="Ground-truth annotation JSON used by run_mmau.py. For CAP/DADA, defaults to ./annotations/mm_au_<subset>_anno.json.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Per-GPU validation batch size. Defaults to --batch-size.")
    parser.add_argument("--eval-fpr-max", type=float, default=0.1)
    parser.add_argument("--save-eval-predictions", action="store_true", help="Save gathered validation predictions JSON at each evaluation point.")
    parser.add_argument("--video-slice-idx", type=int, default=0)
    parser.add_argument("--video-slice-count", type=int, default=1)

    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--mlflow-experiment", type=str, default="accident_anticipation_encoder_videomae")
    parser.add_argument("--mlflow-run-name", type=str, default=None)
    parser.add_argument("--disable-mlflow", action="store_true")

    args = parser.parse_args()

    distributed, rank, local_rank, world_size = setup_distributed()
    is_main_process = rank == 0

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print(args)
        print(
            f"Distributed: {distributed} | rank={rank} | "
            f"local_rank={local_rank} | world_size={world_size} | device={device}"
        )

    # rank-specific setup
    use_mlflow = (not args.disable_mlflow) and is_main_process

    if use_mlflow:
        tracking_uri = args.mlflow_tracking_uri
        if tracking_uri is None:
            mlflow_db_path = os.path.abspath(
                os.path.join(args.output_dir, "mlflow.db")
            )
            tracking_uri = f"sqlite:///{mlflow_db_path}"

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)

    if distributed:
        dist.barrier()

    # CRITICAL:
    # Reset RNG state identically immediately before dataset construction.
    set_seed(args.seed)

    if args.backbone_type == "vipra":
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    if args.backbone_type in ["vavim"]:
        image_size = (288, 512)
    else:
        image_size = (args.image_size, args.image_size)

    if args.use_augmentations:
        # Clip-level transform: one crop box / flip per clip, augmenting only in
        # anticipation_train mode. dataloader._load_clip dispatches on clip_level.
        transform = ClipTransform(size=image_size, mean=mean, std=std)
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size[1], image_size[0])),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    if args.subset in ["CAP", "DADA"]:
        cfg = MMAUConfig(
            root=args.root,
            metadata_json=args.metadata_json,
            subset=args.subset,
            split_name="train",
            seed=args.seed,
            transform=transform,
            snippet_len=args.snippet_len,
            stride=args.stride,
            train_stride=args.train_stride,
            anticipation_horizon_sec=args.anticipation_horizon_sec,
            progress_alpha=args.progress_alpha,
            pair_gap_sec_min=args.pair_gap_sec_min,
            pair_gap_sec_max=args.pair_gap_sec_max,
            custom_risk_mode=args.custom_risk_mode,
            random_pos_neg_sampling=args.preference_random_pos_neg_sampling,
            full_video_progress_risk=args.full_video_progress_risk,
            fraction=args.dataset_fraction,
        )

        train_ds = MMAUAnticipationDataset(cfg, mode="anticipation_train")

        # Test loader
        test_cfg = MMAUConfig(
            root=args.root,
            metadata_json=args.metadata_json,
            subset=args.subset,
            split_name='test',
            seed=args.seed,
            transform=transform,
            snippet_len=args.snippet_len,
            stride=args.stride,
            video_slice_idx=args.video_slice_idx,
            video_slice_count=args.video_slice_count,
            inference_on_train=False,
        )

        binary_ds = MMAUAnticipationDataset(test_cfg, mode=args.eval_mode)

    elif args.subset == "Nexar":

        cfg = NexarConfig(
            root=args.root,
            subset="train",
            seed=args.seed,
            snippet_len=args.snippet_len,
            stride=args.stride,
            transform=transform,
            fps=30,
            train_stride=args.train_stride,
            anticipation_horizon_sec=args.anticipation_horizon_sec,
            progress_alpha=args.progress_alpha,
            pair_gap_sec_min=args.pair_gap_sec_min,
            pair_gap_sec_max=args.pair_gap_sec_max,
            custom_risk_mode=args.custom_risk_mode,
            random_pos_neg_sampling=args.preference_random_pos_neg_sampling,
            fraction=args.dataset_fraction,
        )

        train_ds = NexarAnticipationDataset(cfg, mode="anticipation_train")

        test_cfg = NexarConfig(
            root=args.root,
            subset="test",
            seed=args.seed,
            snippet_len=args.snippet_len,
            stride=args.stride,
            transform=transform,
            fps=30,
            inference_on_train=False,
            anticipation_horizon_sec=args.anticipation_horizon_sec,
        )

        binary_ds = NexarAnticipationDataset(test_cfg, mode=args.eval_mode)

    elif args.subset == "DAD":

        train_cfg = DADConfig(
            root=args.root,
            split_name="training",
            seed=args.seed,
            transform=transform,
            snippet_len=args.snippet_len,
            stride=args.stride,
            train_stride=args.train_stride,
            anticipation_horizon_sec=args.anticipation_horizon_sec,
            progress_alpha=args.progress_alpha,
            pair_gap_sec_min=args.pair_gap_sec_min,
            pair_gap_sec_max=args.pair_gap_sec_max,
            custom_risk_mode=args.custom_risk_mode,
            random_pos_neg_sampling=args.preference_random_pos_neg_sampling,
            full_video_progress_risk=args.full_video_progress_risk,
            no_bce_ablation=args.no_bce_ablation,
            fraction=args.dataset_fraction,
        )

        train_ds = DADAnticipationDataset(
            train_cfg,
            mode="anticipation_train",
        )

        test_cfg = DADConfig(
            root=args.root,
            split_name="testing",
            seed=args.seed,
            transform=transform,
            snippet_len=args.snippet_len,
            stride=args.stride,
            video_slice_idx=args.video_slice_idx,
            video_slice_count=args.video_slice_count,
            inference_on_train=False,
        )

        binary_ds = DADAnticipationDataset(
            test_cfg,
            mode=args.eval_mode,
        )

    # Create data loaders with custom collate_fn for padding variable-length sequences
    collate_fn = partial(anticipation_collate_fn_pad, backbone_type=args.backbone_type)
    collate_fn_test = partial(anticipation_eval_collate_fn_pad, backbone_type=args.backbone_type)

    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    if distributed:
        local_len = torch.tensor(
            [len(train_ds)],
            device=device,
            dtype=torch.long,
        )

        all_lens = [
            torch.zeros_like(local_len)
            for _ in range(world_size)
        ]

        dist.all_gather(all_lens, local_len)

        all_lens = [x.item() for x in all_lens]

        if is_main_process:
            print("Dataset lengths across ranks:", all_lens)

        assert len(set(all_lens)) == 1, (
            f"DDP ranks constructed different datasets: {all_lens}"
        )

    # Distributed validation: shard the validation dataset without padding.
    # We intentionally do NOT use DistributedSampler here because its padding
    # can duplicate validation samples when len(binary_ds) % world_size != 0.
    if distributed:
        eval_indices = list(range(rank, len(binary_ds), world_size))
        eval_ds = Subset(binary_ds, eval_indices)
    else:
        eval_ds = binary_ds

    eval_batch_size = args.eval_batch_size or args.batch_size
    binary_loader = DataLoader(
        eval_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_test,
    )

    if is_main_process:
        print(
            f"Validation dataset: {len(binary_ds)} clips | "
            f"mode={args.eval_mode} | "
            f"world_size={world_size} | "
            f"per-GPU batch={eval_batch_size}"
        )

    # run_mmau.py only needs the GT annotation dict on rank 0.
    anno_dict = None
    eval_anno_json = args.eval_anno_json
    if args.eval_every_n_epochs > 0 and is_main_process:
        if eval_anno_json is None:
            if args.subset.lower() in ["cap", "dada"]:
                eval_anno_json = os.path.join(
                    ".", "annotations", f"mm_au_{args.subset.lower()}_anno.json"
                )
            else:
                raise ValueError(
                    "--eval-anno-json is required for this subset when "
                    "--eval-every-n-epochs > 0."
                )

        if not os.path.exists(eval_anno_json):
            raise FileNotFoundError(
                f"Validation annotation JSON not found: {eval_anno_json}. "
                "Pass the correct path with --eval-anno-json."
            )

        with open(eval_anno_json, "r") as f:
            anno_dict = json.load(f)

        print(f"Loaded validation annotations: {eval_anno_json}")

    # Build model independently on each rank.
    model = build_model(args).to(device)

    if args.pretrained_model:
        load_weights(checkpoint_path=args.pretrained_model, model=model)

    # Build optimizer before wrapping with DDP so any name-based parameter
    # grouping in build_optimizer() sees the original module names.
    optimizer, scheduler = build_optimizer(model, args)

    if distributed:
        # find_unused_parameters=True is safer here because PRE-ACT has
        # multiple auxiliary heads / ablations that may not contribute to
        # every batch. Set it to False later if every trainable parameter is
        # always used, for slightly less DDP overhead.
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    # else:
    #     optimizer = torch.optim.AdamW(
    #         model.parameters(),
    #         lr=args.lr,
    #         weight_decay=args.weight_decay,
    #     )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process:
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Trainable ratio: {100.0 * trainable_params / total_params:.2f}%")

        for name, p in unwrap_model(model).named_parameters():
            if p.requires_grad:
                print("TRAINABLE:", name)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    run_ctx = mlflow.start_run(run_name=args.mlflow_run_name) if use_mlflow else nullcontext()

    with run_ctx:
        if use_mlflow:
            mlflow.log_params({
                "output_dir": args.output_dir,
                "custom_risk_mode": args.custom_risk_mode,
                "subset": args.subset,
                "backbone_type": args.backbone_type,
                "backbone_name": args.backbone_name,
                "pretrained_model": args.pretrained_model,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "epochs": args.epochs,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "freeze_backbone": args.freeze_backbone,
                "temporal_hidden_dim": args.temporal_hidden_dim,
                "image_size": image_size,
                "snippet_len": args.snippet_len,
                "stride": args.stride,
                "train_stride": args.train_stride,
                "anticipation_horizon_sec": args.anticipation_horizon_sec,
                "progress_alpha": args.progress_alpha,
                "pair_gap_sec_min": args.pair_gap_sec_min,
                "pair_gap_sec_max": args.pair_gap_sec_max,
                "lambda_bce": args.lambda_bce,
                "lambda_prog": args.lambda_prog,
                "lambda_pref": args.lambda_pref,
                "num_train_samples": len(train_ds),
                "unfreeze_last_n_blocks": args.unfreeze_last_n_blocks,
                "backbone_lr": args.backbone_lr,
                "head_lr": args.head_lr,
                "transformer_dim": args.transformer_dim,
                "transformer_heads": args.transformer_heads,
                "transformer_layers": args.transformer_layers,
                "transformer_dropout": args.transformer_dropout,
                "videomae_categorical": args.videomae_categorical,
                "pref_loss_type": args.pref_loss_type,
                "random_pos_neg_sampling": args.preference_random_pos_neg_sampling,
                "full_video_progress_risk": args.full_video_progress_risk,
                "no_bce_ablation": args.no_bce_ablation,
                "eval_every_n_epochs": args.eval_every_n_epochs,
                "eval_mode": args.eval_mode,
                "eval_score_key": args.eval_score_key,
                "eval_batch_size": args.eval_batch_size or args.batch_size,
                "eval_fpr_max": args.eval_fpr_max,
            })

        best_mauc_0_1 = 0.0
        for epoch in range(args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{args.epochs}",
                disable=not is_main_process,
            )

            running = {
                "loss": 0.0,
                "loss_bce": 0.0,
                "loss_prog": 0.0,
                "loss_pref": 0.0,
            }

            for step, batch in enumerate(pbar):
                batch["frames"] = batch["frames"].to(device, non_blocking=True)
                if batch["future_frames"] is not None:
                    batch["future_frames"] = batch["future_frames"].to(device, non_blocking=True)
                batch["binary_target"] = batch["binary_target"].to(device, non_blocking=True)
                batch["risk_target"] = batch["risk_target"].to(device, non_blocking=True)
                batch["valid_progress"] = batch["valid_progress"].to(device, non_blocking=True)
                batch["pref_valid"] = batch["pref_valid"].to(device, non_blocking=True)
                batch["has_future"] = batch["has_future"].to(device, non_blocking=True)
                batch["future_risk_target"] = batch["future_risk_target"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):

                    if not args.videomae_categorical:
                        loss_dict = compute_losses(
                            model,
                            batch,
                            lambda_bce=args.lambda_bce,
                            lambda_prog=args.lambda_prog,
                            lambda_pref=args.lambda_pref,
                            pref_loss_type=args.pref_loss_type,
                        )
                    else:
                        loss_dict = compute_losses_ordinal_ttc(
                            model,
                            batch,
                            lambda_bce=args.lambda_bce,
                            lambda_ttc=args.lambda_prog
                        )

                    loss = loss_dict["loss"]

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                for k in running:
                    running[k] += float(loss_dict[k])

                global_step = epoch * len(train_loader) + step
                if use_mlflow and global_step % 50 == 0:
                    mlflow.log_metrics(
                        {
                            "train/loss": float(loss_dict["loss"]),
                            "train/loss_bce": float(loss_dict["loss_bce"]),
                            "train/loss_prog": float(loss_dict["loss_prog"]),
                            "train/loss_pref": float(loss_dict["loss_pref"]),
                        },
                        step=global_step,
                    )

                if is_main_process:
                    pbar.set_postfix({
                        "loss": f"{running['loss'] / (step + 1):.4f}",
                        "bce": f"{running['loss_bce'] / (step + 1):.4f}",
                        "prog": f"{running['loss_prog'] / (step + 1):.4f}",
                        "pref": f"{running['loss_pref'] / (step + 1):.4f}",
                    })

            # Aggregate epoch loss sums across all ranks for correct logging.
            running_global = reduce_epoch_running(running, device, world_size)
            epoch_loss = running_global["loss"] / len(train_loader)
            epoch_bce = running_global["loss_bce"] / len(train_loader)
            epoch_prog = running_global["loss_prog"] / len(train_loader)
            epoch_pref = running_global["loss_pref"] / len(train_loader)

            if use_mlflow:
                mlflow.log_metric("epoch/loss", epoch_loss, step=epoch + 1)
                mlflow.log_metric("epoch/loss_bce", epoch_bce, step=epoch + 1)
                mlflow.log_metric("epoch/loss_prog", epoch_prog, step=epoch + 1)
                mlflow.log_metric("epoch/loss_pref", epoch_pref, step=epoch + 1)

            # ------------------------------------------------------------
            # Epoch-end distributed validation
            # ------------------------------------------------------------
            test_metrics = None
            should_eval = (
                args.eval_every_n_epochs > 0
                and (epoch + 1) % args.eval_every_n_epochs == 0
            )

            if should_eval:
                # All ranks have completed the full training epoch before
                # starting validation.
                if distributed:
                    dist.barrier()

                eval_model = unwrap_model(model)

                # global_step is reported as the number of completed optimizer
                # steps, not a zero-based index.
                eval_global_step = (epoch + 1) * len(train_loader)

                if is_main_process:
                    print(
                        f"\n[VAL] Starting distributed epoch-end validation "
                        f"after epoch {epoch + 1}/{args.epochs} "
                        f"(global_step={eval_global_step}) ..."
                    )

                # Each rank evaluates its own disjoint, non-padded shard.
                local_clip_outputs = run_binary_clip_eval(
                    binary_loader,
                    eval_model,
                    device,
                    show_progress=is_main_process,
                )

                # Gather all prediction dictionaries onto rank 0.
                clip_outputs = gather_eval_outputs(
                    local_clip_outputs,
                    distributed=distributed,
                    rank=rank,
                    world_size=world_size,
                )

                if is_main_process:
                    if len(clip_outputs) != len(binary_ds):
                        raise RuntimeError(
                            "Distributed validation gathered an unexpected "
                            f"number of predictions: got {len(clip_outputs)}, "
                            f"expected {len(binary_ds)}."
                        )

                    if args.save_eval_predictions:
                        pred_path = os.path.join(
                            args.output_dir,
                            f"val_predictions_epoch{epoch+1}.json",
                        )
                        with open(pred_path, "w") as f:
                            json.dump(clip_outputs, f)
                        print(f"[VAL] Saved predictions: {pred_path}")

                    if args.subset in ["CAP", "DADA"]:
                        test_metrics, _ = evaluate_predictions_mmau(
                            clip_outputs=clip_outputs,
                            anno_dict=anno_dict,
                            score_key=args.eval_score_key,
                            snippet_len=args.snippet_len,
                            base_fps=10,
                            fpr_max=args.eval_fpr_max,
                            plot_path=None,
                            match_neg_pos_numbers=False,
                        )
                    elif args.subset == "Nexar":
                        test_metrics, _ = evaluate_predictions_nexar(
                                clip_outputs=clip_outputs,
                                anno_dict=anno_dict,
                                score_key=args.eval_score_key,
                                snippet_len=args.snippet_len,
                                base_fps=10,
                                fpr_max=args.eval_fpr_max,
                                plot_path=None,
                            )
                    elif args.subset == "DAD":
                        test_metrics, _ = evaluate_predictions_dad(
                                clip_outputs=clip_outputs,
                                anno_dict=anno_dict,
                                score_key=args.eval_score_key,
                                snippet_len=args.snippet_len,
                                base_fps=10,
                                fpr_max=args.eval_fpr_max,
                                plot_path=None,
                            )

                    print_validation_metrics(
                        test_metrics,
                        epoch=epoch,
                        num_epochs=args.epochs,
                        global_step=eval_global_step,
                        score_key=args.eval_score_key,
                    )

                    if use_mlflow:
                        val_mlflow = mlflow_validation_metrics(test_metrics)
                        if val_mlflow:
                            mlflow.log_metrics(
                                val_mlflow,
                                step=epoch + 1,
                            )

                    current_mauc = test_metrics.get("mAUC@0.1", float("nan"))
                    if (
                        isinstance(current_mauc, (int, float, np.integer, np.floating))
                        and math.isfinite(float(current_mauc))
                        and float(current_mauc) > best_mauc_0_1
                    ):
                        best_mauc_0_1 = float(current_mauc)
                        best_path = os.path.join(
                            args.output_dir,
                            "best_mauc_0_1.pt",
                        )
                        torch.save(
                            {
                                "model": eval_model.state_dict(),
                                "args": vars(args),
                                "epoch": epoch + 1,
                                "global_step": eval_global_step,
                                "val_metrics": test_metrics,
                            },
                            best_path,
                        )
                        print(
                            f"[VAL] New best mAUC@0.1={best_mauc_0_1:.4f} "
                            f"-> {best_path}"
                        )

                # Rank 0 computes CPU metrics / writes files while the other
                # ranks wait. Resume together afterwards.
                if distributed:
                    dist.barrier()

                model.train()

            # Save one normal checkpoint per epoch. If this epoch was evaluated,
            # also store its validation metrics in that checkpoint.
            if is_main_process:
                ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch+1}.pt")
                checkpoint = {
                    "model": unwrap_model(model).state_dict(),
                    "args": vars(args),
                    "epoch": epoch + 1,
                }
                if test_metrics is not None:
                    checkpoint["val_metrics"] = test_metrics

                torch.save(checkpoint, ckpt_path)

                if use_mlflow:
                    mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")

                print(f"Saved {ckpt_path}")

            scheduler.step()

            if distributed:
                dist.barrier()

        # Save final model on rank 0.
        if is_main_process:
            final_model_path = os.path.join(args.output_dir, "final_model.pt")
            torch.save(
                {
                    "model": unwrap_model(model).state_dict(),
                    "args": vars(args),
                    "epoch": args.epochs,
                },
                final_model_path,
            )
            if use_mlflow:
                mlflow.log_artifact(final_model_path, artifact_path="final_model")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()