import os
import json
import argparse

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from functools import partial

from dataloaders.dataloader import MMAUConfig, MMAUAnticipationDataset
from dataloaders.dataloader_nexar import NexarConfig, NexarAnticipationDataset
from dataloaders.dataloader_dad import DADConfig, DADAnticipationDataset

from models import build_model, remap_videomae_qkv_bias_keys
from engine.collate import anticipation_eval_collate_fn_pad

from evaluation.eval_utils import print_validation_metrics
from evaluation.run_mmau import evaluate_predictions

def fuse_scores(score, risk_score, lam=0.15, k=8.0, b=0.10):
    risk_prob = 1.0 / (1.0 + np.exp(-k * (risk_score - b)))
    return (1.0 - lam) * score + lam * risk_prob


# -----------------------------------------------------------------------------
# Distributed helpers
# -----------------------------------------------------------------------------

def setup_distributed():
    """Initialize torch.distributed when launched with torchrun."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed GPU inference requested, but CUDA is unavailable.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

    return distributed, rank, local_rank, world_size


def gather_list_outputs(local_outputs, distributed, rank, world_size, sort_key=None):
    """Gather variable-length Python lists from every rank onto rank 0."""
    if not distributed:
        return local_outputs

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_outputs)

    if rank != 0:
        return None

    merged = []
    for rank_outputs in gathered:
        merged.extend(rank_outputs)

    if sort_key is not None:
        merged.sort(key=sort_key)

    return merged


def rank0_file_exists(path, distributed, rank, device):
    """Check a shared output path on rank 0 and broadcast the result."""
    flag = torch.zeros(1, dtype=torch.int32, device=device)
    if rank == 0:
        flag[0] = int(os.path.exists(path))
    if distributed:
        dist.broadcast(flag, src=0)
    return bool(flag.item())


def make_clip_shard(dataset, distributed, rank, world_size):
    """Disjoint validation shard with no padding and therefore no duplicates."""
    if not distributed:
        return dataset
    indices = list(range(rank, len(dataset), world_size))
    return Subset(dataset, indices)


def make_video_shard(dataset, distributed, rank, world_size):
    """Keep all clips from one video on the same rank.

    This is used by --run-full because run_full_video_eval reconstructs a complete
    per-video sequence before returning. Splitting one video across ranks would
    make the local forward-fill incomplete.
    """
    if not distributed:
        return dataset

    if not hasattr(dataset, "samples"):
        raise AttributeError("Dataset must expose .samples for video-level sharding.")

    video_to_indices = {}
    video_order = []
    for idx, sample in enumerate(dataset.samples):
        key = str(sample["video_hashcode"])
        if key not in video_to_indices:
            video_to_indices[key] = []
            video_order.append(key)
        video_to_indices[key].append(idx)

    local_videos = video_order[rank::world_size]
    local_indices = []
    for key in local_videos:
        local_indices.extend(video_to_indices[key])

    return Subset(dataset, local_indices)


# -----------------------------------------------------------------------------
# Build / load model
# -----------------------------------------------------------------------------

def load_checkpoint_model(checkpoint_path, device, verbose=True):
    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    train_args = ckpt["args"]

    if verbose:
        print(train_args)

    model = build_model(train_args)

    try:  # transformer-v4 trained models
        model.load_state_dict(ckpt["model"], strict=True)
    except RuntimeError:
        state_dict = remap_videomae_qkv_bias_keys(ckpt["model"])
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if verbose:
            print("Missing:", missing)
            print("Unexpected:", unexpected)

    model.to(device)
    model.eval()
    return model, train_args


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------

@torch.no_grad()
def run_binary_clip_eval(loader, model, device, show_progress=True):
    outputs = []
    model.eval()

    for batch in tqdm(
        loader,
        desc="Validation inference",
        disable=not show_progress,
    ):
        frames = batch["frames"].to(device, non_blocking=True)
        out = model(frames)

        if out["risk_logit"] is not None: # BCE Head
            scores = torch.sigmoid(out["risk_logit"]) # BCE Head
        else:
            scores = torch.zeros(batch["frames"].shape[0], device=batch["frames"].device)

        if scores.ndim > 1:
            scores = scores.amax(dim=-1)

        if out["progress_logit"] is not None: # Progress Head
            scores_risk = out["progress_logit"].detach().cpu().numpy()
        else:
            scores_risk = torch.zeros_like(scores).detach().cpu().numpy()

        scores = scores.detach().cpu().numpy()

        for i, item in enumerate(batch["meta"]):
            if "frame_paths" in item and item["frame_paths"] is not None:
                clip_start = int(item["frame_paths"][0].stem)
            elif "frame_indices" in item and item["frame_indices"] is not None:
                clip_start = int(item["frame_indices"][0]) + 1
            elif "current_frame_idx_1based" in item:
                clip_start = int(item["current_frame_idx_1based"])
            else:
                clip_start = -1

            outputs.append({
                "video_hashcode": item["video_hashcode"],
                "label": int(item["label"]),
                "score": float(scores[i]),
                "risk_score": float(scores_risk[i]),
                "horizon_sec": item["horizon_sec"],
                "fused_score": float(fuse_scores(scores[i], scores_risk[i])),
                "clip_names": str(f"{item['video_folder']}_{clip_start}"),
            })

    return outputs


@torch.no_grad()
def run_full_video_eval(loader, model, device, score_key="progress_logit", show_progress=True):
    by_video = {}
    model.eval()

    for batch in tqdm(loader, desc="Full video eval", disable=not show_progress):
        frames = batch["frames"].to(device, non_blocking=True)
        out = model(frames)

        if score_key not in out:
            raise KeyError(
                f"{score_key} not found in model output. Available keys: {list(out.keys())}"
            )

        if score_key == "progress_logit":
            # raw scores for Progress Head
            scores = out[score_key].detach().cpu().numpy()
        else:
            # sigmoid for BCE head
            scores = torch.sigmoid(out[score_key]).detach().cpu().numpy()

        for i, item in enumerate(batch["meta"]):
            key = item["video_hashcode"]
            total_frames = int(item["total_frames"])

            if key not in by_video:
                by_video[key] = {
                    "scores": np.zeros(total_frames, dtype=np.float32),
                    "filled": np.zeros(total_frames, dtype=np.uint8),
                    "t_ai": int(item["t_ai"]),
                    "t_co": int(item["t_co"]),
                    "t_ae": int(item["t_ae"]),
                    "fps": float(item.get("fps", 10.0)),
                    "clip_names": str(
                        f"{item['video_folder']}_{int(item['frame_paths'][0].stem)}"
                    ),
                    "label": np.zeros(total_frames, dtype=np.uint8),
                }

            idx = int(item["current_frame_idx_1based"]) - 1
            by_video[key]["scores"][idx] = float(scores[i])
            by_video[key]["filled"][idx] = 1
            by_video[key]["label"][idx] = item["label"]

    video_predictions = []
    for key, value in by_video.items():
        scores = value["scores"]
        filled = value["filled"]

        last = 0.0
        for j in range(len(scores)):
            if filled[j]:
                last = scores[j]
            else:
                scores[j] = last

        video_predictions.append({
            "video_hashcode": key,
            "scores": scores.tolist(),
            "t_ai": value["t_ai"],
            "t_co": value["t_co"],
            "t_ae": value["t_ae"],
            "fps": value["fps"],
            "clip_names": value["clip_names"],
            "label": value["label"].tolist(),
        })

    return video_predictions


# -----------------------------------------------------------------------------
# Dataset construction
# -----------------------------------------------------------------------------

def get_checkpoint_arg(train_args, name, default):
    if isinstance(train_args, dict):
        return train_args.get(name, default)
    return getattr(train_args, name, default)


def build_eval_dataset(args, train_args, transform, mode):
    image_size = get_checkpoint_arg(train_args, "image_size", 224)
    del image_size  # transform is already built; retained for checkpoint compatibility
    snippet_len = get_checkpoint_arg(train_args, "snippet_len", 5)
    stride = 1
    anticipation_horizon_sec = get_checkpoint_arg(
        train_args, "anticipation_horizon_sec", 2.0
    )

    if args.subset == "Nexar":
        cfg = NexarConfig(
            root=args.root,
            subset="test",
            snippet_len=snippet_len,
            stride=stride,
            transform=transform,
            fps=30,
            inference_on_train=False,
            anticipation_horizon_sec=anticipation_horizon_sec,
            video_slice_idx=args.video_slice_idx,
            video_slice_count=args.video_slice_count,
        )
        return NexarAnticipationDataset(cfg, mode=mode)

    if args.subset in ["CAP", "DADA"]:
        cfg = MMAUConfig(
            root=args.root,
            metadata_json=args.metadata_json,
            subset=args.subset,
            split_name=args.split_name,
            transform=transform,
            snippet_len=snippet_len,
            stride=stride,
            video_slice_idx=args.video_slice_idx,
            video_slice_count=args.video_slice_count,
            inference_on_train=False,
            anticipation_horizon_sec=anticipation_horizon_sec,
        )
        return MMAUAnticipationDataset(cfg, mode=mode)

    if args.subset == "DAD":
        cfg = DADConfig(
            root=args.root,
            split_name="testing",
            transform=transform,
            snippet_len=snippet_len,
            stride=stride,
            video_slice_idx=args.video_slice_idx,
            video_slice_count=args.video_slice_count,
            inference_on_train=False,
            anticipation_horizon_sec=anticipation_horizon_sec,
        )
        return DADAnticipationDataset(cfg, mode=mode)

    raise ValueError(f"Unsupported subset: {args.subset}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--metadata-json", type=str, required=True, default=None)
    parser.add_argument(
        "--subset",
        type=str,
        default="CAP",
        choices=["CAP", "DADA", "Nexar", "DAD"],
    )
    parser.add_argument("--split-name", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=256, help="Per-GPU batch size.")
    parser.add_argument("--num-workers", type=int, default=8, help="Workers per GPU/process.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Single-process device. Ignored under multi-GPU torchrun.",
    )
    parser.add_argument("--save-dir", type=str, default="outputs_encoder")
    parser.add_argument("--run-binary", action="store_true")
    parser.add_argument("--run-full", action="store_true")
    parser.add_argument("--run-sliding-window", action="store_true")
    parser.add_argument("--video-slice-idx", type=int, default=0)
    parser.add_argument("--video-slice-count", type=int, default=1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run inference even when the output JSON already exists.",
    )
    parser.add_argument("--snippet-len", type=int, default=5)

    parser.add_argument("--eval-score-key", type=str, default="risk_score", choices=["score", "risk_score", "fused_score"], help="Prediction field consumed by run_mmau.evaluate_predictions.")
    parser.add_argument("--eval-anno-json", type=str, default=None, help="Ground-truth annotation JSON used by run_mmau.py. For CAP/DADA, defaults to ./annotations/mm_au_<subset>_anno.json.")
    parser.add_argument("--eval-fpr-max", type=float, default=0.1)

    return parser.parse_args()


def main():
    args = parse_args()

    if not (args.run_binary or args.run_sliding_window or args.run_full):
        raise ValueError("Select at least one of --run-binary, --run-sliding-window, --run-full.")

    distributed, rank, local_rank, world_size = setup_distributed()
    is_main_process = rank == 0

    print(
        f"PID={os.getpid()} "
        f"rank={rank} local_rank={local_rank} world={world_size} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"current_gpu={torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'}",
        flush=True,
    )

    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    elif torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    if torch.cuda.is_available():
        print(
            f"[rank {rank}] "
            f"device={torch.cuda.current_device()} "
            f"name={torch.cuda.get_device_name()} "
            f"allocated={torch.cuda.memory_allocated(device)/1024**3:.2f} GB "
            f"reserved={torch.cuda.memory_reserved(device)/1024**3:.2f} GB",
            flush=True,
        )

    if is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)
        print(
            f"Distributed: {distributed} | rank={rank} | local_rank={local_rank} | "
            f"world_size={world_size} | device={device}"
        )

    if distributed:
        dist.barrier()

    model, train_args = load_checkpoint_model(
        args.checkpoint,
        device=device,
        verbose=is_main_process,
    )

    image_size = get_checkpoint_arg(train_args, "image_size", 224)
    backbone_type = get_checkpoint_arg(train_args, "backbone_type", "videomae")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    collate_fn_test = partial(
        anticipation_eval_collate_fn_pad,
        backbone_type=backbone_type,
    )

    if is_main_process:
        print(f"Using {args.subset} dataset for evaluation on {world_size} process(es).")
        print(f"Per-GPU batch size: {args.batch_size}")

    # ------------------------------------------------------------------
    # Binary / sliding-window clip inference
    # ------------------------------------------------------------------
    clip_modes = []
    if args.run_binary:
        clip_modes.append(("binary_clips", "clip_outputs_"))
    if args.run_sliding_window:
        clip_modes.append(("sliding_window", "sliding_window_outputs_"))

    for mode, heading in clip_modes:
        binary_ds = build_eval_dataset(args, train_args, transform, mode=mode)
        local_ds = make_clip_shard(binary_ds, distributed, rank, world_size)

        binary_loader = DataLoader(
            local_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=1 if args.num_workers > 0 else None,
            collate_fn=collate_fn_test,
        )

        if is_main_process:
            print(
                f"[{mode}] global clips={len(binary_ds)} | world_size={world_size} | "
                f"rank0 local clips={len(local_ds)}"
            )

        clip_path = os.path.join(
            args.save_dir,
            f"{heading}{args.subset.lower()}_slice{args.video_slice_idx}.json",
        )

        already_exists = (
            rank0_file_exists(clip_path, distributed, rank, device)
            if not args.overwrite
            else False
        )

        if already_exists:
            if is_main_process:
                print(f"Output already exists, skipping inference: {clip_path}")
            if distributed:
                dist.barrier()
            continue

        local_outputs = run_binary_clip_eval(
            binary_loader,
            model,
            device,
            show_progress=is_main_process,
        )

        clip_outputs = gather_list_outputs(
            local_outputs,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            sort_key=lambda x: (
                str(x.get("video_hashcode", "")),
                str(x.get("clip_names", "")),
            ),
        )

        if is_main_process:
            if len(clip_outputs) != len(binary_ds):
                raise RuntimeError(
                    f"Gathered {len(clip_outputs)} predictions, expected {len(binary_ds)}."
                )
            with open(clip_path, "w") as f:
                json.dump(clip_outputs, f)
            print(f"Saved {len(clip_outputs)} predictions: {clip_path}")

            # run_mmau.py only needs the GT annotation dict on rank 0.
            anno_dict = None
            eval_anno_json = args.eval_anno_json
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

            test_metrics, _ = evaluate_predictions(
                clip_outputs=clip_outputs,
                anno_dict=anno_dict,
                score_key=args.eval_score_key,
                snippet_len=args.snippet_len,
                base_fps=10,
                fpr_max=args.eval_fpr_max,
                plot_path=None,
                match_neg_pos_numbers=False,
            )

            print_validation_metrics(
                test_metrics,
                epoch=0,
                num_epochs=30,
                global_step=0,
                score_key=args.eval_score_key,
            )

        if distributed:
            dist.barrier()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
