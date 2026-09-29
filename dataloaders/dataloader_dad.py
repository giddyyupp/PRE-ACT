import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

import torch
from torch.utils.data import Dataset

from dataloaders.helpers import deterministic_record_subsets, print_subset_stats
from engine.risk_targets import get_risk_target_fn, full_video_progress_risk_fn


@dataclass
class DADConfig:
    root: str
    fps: int = 20
    image_size: int = 224
    snippet_len: int = 5
    stride: int = 1
    transform: Optional[Any] = None
    split_name: str = "testing"  # "training" or "testing"
    seed: int = 42
    video_slice_idx: int = 0
    video_slice_count: int = 1
    base_fps: int = 10
    full_video_progress_risk: bool = False
    accident_frame: int = 90

    anticipation_horizon_sec: float = 2.0
    progress_alpha: float = 5.0
    pair_gap_sec_min: float = 0.5
    pair_gap_sec_max: float = 1.5
    include_post_collision: bool = False
    custom_risk_mode: str = "exp_above"
    random_pos_neg_sampling: bool = False

    train_stride: int = 5
    neg_keep_prob: float = 0.8
    pref_keep_prob: float = 0.9
    max_samples_per_video: Optional[int] = 50
    inference_on_train: bool = False
    no_bce_ablation: bool = False

    fraction: Optional[float] = 1.0  # For subset selection, e.g., 0.01, 0.05, 0.10


class DADAnticipationDataset(Dataset):
    """
    DAD loader with the same modes and returned fields as
    MMAUAnticipationDataset.

    Expected layout:

        root/videos/training/positive/<video_id>/images/*.jpg
        root/videos/training/negative/<video_id>/images/*.jpg
        root/videos/testing/positive/<video_id>/images/*.jpg
        root/videos/testing/negative/<video_id>/images/*.jpg

    No metadata JSON is required.

    Labels and event frames are inferred from the folder name:
        positive: t_ai=50, t_co=90, t_ae=90
        negative: t_ai=t_co=t_ae=-1

    FPS is fixed to 20.
    """

    def __init__(
        self,
        cfg: DADConfig,
        mode: str = "test_full",
        horizons_sec: Tuple[float, ...] = (0.5, 1.0, 1.5),
        clip_duration_sec: float = 0.5,
    ):
        self.cfg = cfg
        self.mode = mode
        self.horizons_sec = horizons_sec
        self.clip_duration_sec = clip_duration_sec
        self.root = Path(cfg.root)

        print(
            f"Initializing DADAnticipationDataset with cfg={cfg} with mode={mode}",
            flush=True,
        )

        self.records = self._discover_records()

        if self.cfg.split_name == "testing":
            self.records = self._slice_records(self.records)

        if mode == "test_full":
            self.samples = self._tta_test()
        elif mode == "binary_clips":
            if self.cfg.inference_on_train:
                self.samples = self._build_binary_clip_samples_fps_fix()
            else:
                self.samples = self._build_binary_clip_samples_fps_fix_TOP_like()
        elif mode == "sliding_window":
            self.samples = self._build_binary_clip_samples_fps_fix_TOP_like_sliding_window()
        elif mode == "anticipation_train":
            # SUBSET SELECTION LOGIC -- Applies only to training mode, not test/eval modes
            self.subsets = deterministic_record_subsets(
                self.records,
                fractions=(self.cfg.fraction,),
                seed=42,
                stratify_by_label=True,
                id_key="video_hashcode",
                label_fn=lambda r: int(r["is_positive"]),
            )

            print_subset_stats(
                self.subsets,
                label_fn=lambda r: int(r["is_positive"]),
            )

            self.records = self.subsets[self.cfg.fraction]

            if self.cfg.random_pos_neg_sampling:
                self.samples = self._build_anticipation_train_samples_fps_fix_random_pos_neg_pairs()
            else:
                self.samples = self._build_anticipation_train_samples_fps_fix_subsample()
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        positive_videos = sum(int(r["is_positive"]) for r in self.records)
        positive_samples = sum(
            int(s.get("label", s.get("binary_target", 0.0)) > 0)
            for s in self.samples
        )
        print(
            f"[DAD] split={self._split_folder_name()}, fps={cfg.fps}, "
            f"videos={len(self.records)}, positive_videos={positive_videos}, "
            f"samples={len(self.samples)}, positive_samples={positive_samples}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Discovery and metadata
    # ------------------------------------------------------------------

    def _split_folder_name(self) -> str:
        split = self.cfg.split_name.lower()
        if split not in {"training", "testing"}:
            raise ValueError(
                f"DAD split_name must be 'training' or 'testing', got {self.cfg.split_name!r}"
            )
        return split

    def _discover_records(self) -> List[Dict[str, Any]]:
        split_folder = self._split_folder_name()
        split_root = self.root / "videos" / split_folder

        if not split_root.exists():
            raise FileNotFoundError(f"DAD split directory not found: {split_root}")

        records: List[Dict[str, Any]] = []

        for class_name, is_positive in (("positive", True), ("negative", False)):
            class_root = split_root / class_name

            if not class_root.exists():
                raise FileNotFoundError(f"DAD class directory not found: {class_root}")

            for video_dir in sorted(class_root.iterdir()):
                if not video_dir.is_dir():
                    continue

                images_dir = video_dir / "images"
                if not images_dir.exists():
                    continue

                frames = self._list_frames(images_dir)
                num_images = len(frames)
                video_id = video_dir.name

                video_fps = int(round(self.cfg.fps))  # or rec["fps"] if you ever support variable FPS

                if is_positive:
                    t_co = min(self.cfg.accident_frame, num_images)

                    anticipation_frames = int(round(
                        self.cfg.anticipation_horizon_sec * video_fps
                    ))

                    t_ai = max(1, t_co - anticipation_frames)
                    t_ae = t_co
                else:
                    t_ai = -1
                    t_co = -1
                    t_ae = -1

                records.append({
                    "video_hashcode": video_id,
                    "video_name": video_id,
                    "video_folder": video_id,
                    "id": 1 if is_positive else 0,
                    "type": 1 if is_positive else 0,
                    "is_positive": is_positive,
                    "class_name": class_name,
                    "split": split_folder,
                    "image_dir": images_dir,
                    "total_frames": num_images,
                    "num_images": num_images,
                    "fps": 20.0,
                    "t_ai": t_ai,
                    "t_co": t_co,
                    "t_ae": t_ae,
                    "abnormal_start_frame": t_ai if is_positive else None,
                    "accident_frame": t_co if is_positive else num_images,
                    "abnormal_end_frame": t_ae if is_positive else num_images,
                    "accident_type": "Clear",
                    "time_to_accident_sec": self.cfg.anticipation_horizon_sec if is_positive else None,
                    "texts": "",
                    "causes": "",
                    "measures": "",
                })

        if not records:
            raise RuntimeError(f"No usable DAD videos found below {split_root}")

        return records

    def _slice_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        slice_count = int(self.cfg.video_slice_count)
        slice_idx = int(self.cfg.video_slice_idx)
        if slice_count < 1:
            raise ValueError(f"video_slice_count must be >= 1, got {slice_count}")
        if not 0 <= slice_idx < slice_count:
            raise ValueError(
                f"video_slice_idx must be in [0, {slice_count - 1}], got {slice_idx}"
            )
        if slice_count == 1:
            return records
        n = len(records)
        start = (n * slice_idx) // slice_count
        end = (n * (slice_idx + 1)) // slice_count
        return records[start:end]

    def _has_frames(self, folder: Path) -> bool:
        return any(folder.glob("*.jpg")) or any(folder.glob("*.jpeg")) or any(folder.glob("*.png"))

    def _resolve_video_dir(self, rec: Dict[str, Any]) -> Path:
        image_dir = Path(rec["image_dir"])
        if not image_dir.exists():
            raise FileNotFoundError(
                f"Could not find DAD images for {rec['video_hashcode']}: {image_dir}"
            )
        return image_dir

    def _get_fps(self, rec: Dict[str, Any]) -> int:
        return int(rec.get("fps", self.cfg.fps))

    def _list_frames(self, image_dir: Path) -> List[Path]:
        frames = (
            list(image_dir.glob("*.jpg"))
            + list(image_dir.glob("*.jpeg"))
            + list(image_dir.glob("*.png"))
        )
        frames = sorted(frames)
        if not frames:
            raise FileNotFoundError(f"No frames found in {image_dir}")
        return frames

    # ------------------------------------------------------------------
    # Frame sampling
    # ------------------------------------------------------------------

    def _sample_frame_paths_for_duration(
        self,
        frames: List[Path],
        end_idx: int,
        video_fps: int,
    ) -> Optional[List[Path]]:
        S = self.cfg.snippet_len
        sample_stride = max(1, int(round(video_fps / self.cfg.base_fps)))
        start_idx = end_idx - (S - 1) * sample_stride
        if start_idx < 0 or end_idx >= len(frames):
            return None
        idxs = [start_idx + i * sample_stride for i in range(S)]
        if idxs[-1] != end_idx:
            idxs[-1] = end_idx
        return [frames[i] for i in idxs]

    # ------------------------------------------------------------------
    # Test/evaluation modes
    # ------------------------------------------------------------------

    def _tta_test(self) -> List[Dict[str, Any]]:
        samples = []
        S = self.cfg.snippet_len

        for rec in self.records:
            if not rec["is_positive"]:
                continue

            frames = self._list_frames(self._resolve_video_dir(rec))
            video_fps = self._get_fps(rec)
            t_ai, t_co, t_ae = int(rec["t_ai"]), int(rec["t_co"]), int(rec["t_ae"])
            horizon_frames = int(round(self.cfg.anticipation_horizon_sec * video_fps))
            sample_stride = max(1, int(round(video_fps / self.cfg.base_fps)))
            first_valid_end_1b = (S - 1) * sample_stride + 1
            start_end_1b = max(first_valid_end_1b, t_ai)
            end_end_1b = min(t_co, len(frames))

            for end_idx_1b in range(start_end_1b, end_end_1b + 1, self.cfg.stride):
                frame_paths = self._sample_frame_paths_for_duration(
                    frames, end_idx_1b - 1, video_fps
                )
                if frame_paths is None:
                    continue
                ttc_frames = t_co - end_idx_1b
                ttc_sec = ttc_frames / float(video_fps)
                label = int(ttc_frames <= horizon_frames)
                samples.append(
                    self._make_eval_sample(
                        rec,
                        frame_paths,
                        label,
                        ttc_sec if label == 1 else None,
                        end_idx_1b,
                        ttc_frames,
                        ttc_sec,
                        video_fps,
                        t_ai,
                        t_co,
                        t_ae,
                    )
                )
        return samples

    def _build_binary_clip_samples_fps_fix_TOP_like_sliding_window(self) -> List[Dict[str, Any]]:
        samples = []
        S = self.cfg.snippet_len

        for rec in self.records:
            frames = self._list_frames(self._resolve_video_dir(rec))
            video_fps = self._get_fps(rec)
            t_ai, t_co, t_ae = int(rec["t_ai"]), int(rec["t_co"]), int(rec["t_ae"])
            horizon_frames = int(round(self.cfg.anticipation_horizon_sec * video_fps))
            sample_stride = max(1, int(round(video_fps / self.cfg.base_fps)))
            first_valid_end_1b = (S - 1) * sample_stride + 1
            end_end_1b = min(t_co, len(frames)) if rec["is_positive"] else len(frames)

            for end_idx_1b in range(
                first_valid_end_1b,
                end_end_1b + 1,
                sample_stride,
            ):
                frame_paths = self._sample_frame_paths_for_duration(
                    frames, end_idx_1b - 1, video_fps
                )
                if frame_paths is None:
                    continue

                if rec["is_positive"]:
                    ttc_frames = t_co - end_idx_1b
                    ttc_sec = ttc_frames / float(video_fps)
                    label = int(ttc_frames <= horizon_frames)
                    horizon_sec = ttc_sec if label == 1 else None
                else:
                    ttc_frames, ttc_sec, label, horizon_sec = -1, -1.0, 0, None

                samples.append(
                    self._make_eval_sample(
                        rec,
                        frame_paths,
                        label,
                        horizon_sec,
                        end_idx_1b,
                        ttc_frames,
                        ttc_sec,
                        video_fps,
                        t_ai,
                        t_co,
                        t_ae,
                    )
                )
        return samples

    def _build_binary_clip_samples_fps_fix_TOP_like(self) -> List[Dict[str, Any]]:
        samples = []
        S = self.cfg.snippet_len

        for rec in self.records:
            frames = self._list_frames(self._resolve_video_dir(rec))
            video_fps = self._get_fps(rec)
            t_ai, t_co, t_ae = int(rec["t_ai"]), int(rec["t_co"]), int(rec["t_ae"])
            sample_stride = max(1, int(round(video_fps / self.cfg.base_fps)))
            first_valid_end_1b = (S - 1) * sample_stride + 1

            if rec["is_positive"]:
                neg_end_max = (
                    t_co
                    - int(round(self.cfg.anticipation_horizon_sec * video_fps))
                    - 1
                )
            else:
                neg_end_max = len(frames)

            if neg_end_max >= first_valid_end_1b:
                for end_idx_1b in range(
                    first_valid_end_1b,
                    neg_end_max + 1,
                    self.cfg.stride,
                ):
                    frame_paths = self._sample_frame_paths_for_duration(
                        frames, end_idx_1b - 1, video_fps
                    )
                    if frame_paths is None:
                        continue
                    if rec["is_positive"]:
                        ttc_frames = t_co - end_idx_1b
                        ttc_sec = ttc_frames / float(video_fps)
                    else:
                        ttc_frames, ttc_sec = -1, -1.0
                    samples.append(
                        self._make_eval_sample(
                            rec,
                            frame_paths,
                            0,
                            None,
                            end_idx_1b,
                            ttc_frames,
                            ttc_sec,
                            video_fps,
                            t_ai,
                            t_co,
                            t_ae,
                        )
                    )

            if not rec["is_positive"]:
                continue

            for h in self.horizons_sec:
                target_frame_1b = t_co - int(round(h * video_fps))
                if target_frame_1b < first_valid_end_1b:
                    continue
                frame_paths = self._sample_frame_paths_for_duration(
                    frames, target_frame_1b - 1, video_fps
                )
                if frame_paths is None:
                    continue
                ttc_frames = t_co - target_frame_1b
                samples.append(
                    self._make_eval_sample(
                        rec,
                        frame_paths,
                        1,
                        h,
                        target_frame_1b,
                        ttc_frames,
                        ttc_frames / float(video_fps),
                        video_fps,
                        t_ai,
                        t_co,
                        t_ae,
                    )
                )
        return samples

    def _build_binary_clip_samples_fps_fix(self) -> List[Dict[str, Any]]:
        samples = []
        S = self.cfg.snippet_len

        for rec in self.records:
            frames = self._list_frames(self._resolve_video_dir(rec))
            video_fps = self._get_fps(rec)
            t_ai, t_co, t_ae = int(rec["t_ai"]), int(rec["t_co"]), int(rec["t_ae"])
            sample_stride = max(1, int(round(video_fps / self.cfg.base_fps)))
            clip_span_frames = max(S, int(round(self.clip_duration_sec * video_fps)))
            first_valid_end_1b = (S - 1) * sample_stride + 1
            neg_end_max = t_ai - 1 if rec["is_positive"] else len(frames)

            for end_idx_1b in range(
                first_valid_end_1b,
                neg_end_max + 1,
                clip_span_frames,
            ):
                frame_paths = self._sample_frame_paths_for_duration(
                    frames, end_idx_1b - 1, video_fps
                )
                if frame_paths is None:
                    continue
                if rec["is_positive"]:
                    ttc_frames = t_co - end_idx_1b
                    ttc_sec = ttc_frames / float(video_fps)
                else:
                    ttc_frames, ttc_sec = -1, -1.0
                samples.append(
                    self._make_eval_sample(
                        rec,
                        frame_paths,
                        0,
                        None,
                        end_idx_1b,
                        ttc_frames,
                        ttc_sec,
                        video_fps,
                        t_ai,
                        t_co,
                        t_ae,
                    )
                )

            if not rec["is_positive"]:
                continue

            for h in self.horizons_sec:
                target_frame_1b = t_co - int(round(h * video_fps))
                if target_frame_1b < first_valid_end_1b:
                    continue
                frame_paths = self._sample_frame_paths_for_duration(
                    frames, target_frame_1b - 1, video_fps
                )
                if frame_paths is None:
                    continue
                ttc_frames = t_co - target_frame_1b
                samples.append(
                    self._make_eval_sample(
                        rec,
                        frame_paths,
                        1,
                        h,
                        target_frame_1b,
                        ttc_frames,
                        ttc_frames / float(video_fps),
                        video_fps,
                        t_ai,
                        t_co,
                        t_ae,
                    )
                )
        return samples

    def _make_eval_sample(
        self,
        rec: Dict[str, Any],
        frame_paths: List[Path],
        label: int,
        horizon_sec: Optional[float],
        current_frame_idx_1based: int,
        ttc_frames: int,
        ttc_sec: float,
        fps: int,
        t_ai: int,
        t_co: int,
        t_ae: int,
    ) -> Dict[str, Any]:
        return {
            "video_hashcode": rec["video_hashcode"],
            "record": rec,
            "frame_paths": frame_paths,
            "label": label,
            "horizon_sec": horizon_sec,
            "video_folder": rec["video_folder"],
            "current_frame_idx_1based": current_frame_idx_1based,
            "ttc_frames": ttc_frames,
            "ttc_sec": ttc_sec,
            "fps": fps,
            "t_ai": t_ai,
            "t_co": t_co,
            "t_ae": t_ae,
            "is_after_anomaly": int(rec["is_positive"] and current_frame_idx_1based >= t_ai),
            "is_before_collision": int(rec["is_positive"] and current_frame_idx_1based <= t_co),
        }

    # ------------------------------------------------------------------
    # Training modes
    # ------------------------------------------------------------------

    def _build_anticipation_train_samples_fps_fix_subsample(self) -> List[Dict[str, Any]]:
        samples = []
        S = self.cfg.snippet_len
        stride = self.cfg.train_stride
        rng = random.Random(self.cfg.seed)
        risk_target_fn = get_risk_target_fn(self.cfg)

        for rec in self.records:
            frames = self._list_frames(self._resolve_video_dir(rec))
            video_fps = self._get_fps(rec)
            t_co = int(rec["t_co"])
            is_accident = bool(rec["is_positive"])
            horizon_frames = int(round(self.cfg.anticipation_horizon_sec * video_fps))
            sample_stride = max(1, int(round(video_fps / self.cfg.base_fps)))
            first_valid_end_idx = (S - 1) * sample_stride
            video_samples = []

            endpoint_stride = stride * sample_stride

            for end_idx in range(
                first_valid_end_idx,
                len(frames),
                endpoint_stride,
            ):
            # for end_idx in range(first_valid_end_idx, len(frames), stride):
                end_frame_1b = end_idx + 1
                if (
                    is_accident
                    and not self.cfg.include_post_collision
                    and end_frame_1b > t_co
                ):
                    continue

                frame_paths = self._sample_frame_paths_for_duration(
                    frames, end_idx, video_fps
                )
                if frame_paths is None:
                    continue

                if is_accident and end_frame_1b <= t_co:
                    ttc_frames = t_co - end_frame_1b
                    binary_target = 1.0 if ttc_frames <= horizon_frames else 0.0
                    if not self.cfg.full_video_progress_risk:
                        ttc_frames_10fps = ttc_frames * self.cfg.base_fps / video_fps
                        risk_target = float(risk_target_fn(ttc_frames_10fps))
                    else:
                        risk_target = float(
                            full_video_progress_risk_fn(
                                current_frame_idx_1based=end_frame_1b,
                                t_co=t_co,
                                first_valid_frame_1based=first_valid_end_idx + 1,
                                mode=self.cfg.custom_risk_mode,
                                alpha=self.cfg.progress_alpha,
                            )
                        )
                    valid_progress = 1.0
                else:
                    ttc_frames = -1
                    binary_target = 0.0
                    risk_target = 0.0
                    # With BCE off, the progress loss must supply the downward
                    # anchor BCE used to provide, so negatives enter it with
                    # target 0 instead of being masked out.
                    valid_progress = 1.0 # if self.cfg.no_bce_ablation else 0.0

                if binary_target == 0.0 and rng.random() > self.cfg.neg_keep_prob:
                    continue

                future_frame_paths = None
                pref_valid = 0.0
                future_risk_target = 0.0

                if is_accident and end_frame_1b <= t_co:
                    min_gap = max(1, int(round(self.cfg.pair_gap_sec_min * video_fps)))
                    max_gap = max(min_gap, int(round(self.cfg.pair_gap_sec_max * video_fps)))
                    pair_gap_frames = rng.randint(min_gap, max_gap)
                    future_end_frame_1b = min(t_co, end_frame_1b + pair_gap_frames)
                    future_frame_paths = self._sample_frame_paths_for_duration(
                        frames, future_end_frame_1b - 1, video_fps
                    )

                    if future_frame_paths is not None and future_end_frame_1b > end_frame_1b:
                        future_ttc_frames = t_co - future_end_frame_1b
                        if not self.cfg.full_video_progress_risk:
                            future_ttc_frames_10fps = (
                                future_ttc_frames * self.cfg.base_fps / video_fps
                            )
                            future_risk_target = float(
                                risk_target_fn(future_ttc_frames_10fps)
                            )
                        else:
                            future_risk_target = float(
                                full_video_progress_risk_fn(
                                    current_frame_idx_1based=future_end_frame_1b,
                                    t_co=t_co,
                                    first_valid_frame_1based=first_valid_end_idx + 1,
                                    mode=self.cfg.custom_risk_mode,
                                    alpha=self.cfg.progress_alpha,
                                )
                            )

                        if future_risk_target > risk_target and rng.random() <= self.cfg.pref_keep_prob:
                            pref_valid = 1.0
                        else:
                            future_frame_paths = None
                            future_risk_target = 0.0

                video_samples.append(
                    {
                        "video_hashcode": rec["video_hashcode"],
                        "record": rec,
                        "frame_paths": frame_paths,
                        "future_frame_paths": future_frame_paths,
                        "binary_target": binary_target,
                        "risk_target": risk_target,
                        "valid_progress": valid_progress,
                        "pref_valid": pref_valid,
                        "future_risk_target": future_risk_target,
                        "ttc_frames": ttc_frames,
                        "current_frame_idx_1based": end_frame_1b,
                        "fps": video_fps,
                    }
                )

            video_samples = self._cap_video_samples(video_samples, rng)
            samples.extend(video_samples)

        self._print_train_stats(samples)
        return samples

    def _build_anticipation_train_samples_fps_fix_random_pos_neg_pairs(self) -> List[Dict[str, Any]]:
        samples = []
        S = self.cfg.snippet_len
        stride = self.cfg.train_stride
        rng = random.Random(self.cfg.seed)
        risk_target_fn = get_risk_target_fn(self.cfg)

        for rec in self.records:
            frames = self._list_frames(self._resolve_video_dir(rec))
            video_fps = self._get_fps(rec)
            t_co = int(rec["t_co"])
            is_accident = bool(rec["is_positive"])
            horizon_frames = int(round(self.cfg.anticipation_horizon_sec * video_fps))
            sample_stride = max(1, int(round(video_fps / self.cfg.base_fps)))
            first_valid_end_idx = (S - 1) * sample_stride
            video_samples = []

            for end_idx in range(first_valid_end_idx, len(frames), stride):
                end_frame_1b = end_idx + 1
                if (
                    is_accident
                    and not self.cfg.include_post_collision
                    and end_frame_1b > t_co
                ):
                    continue

                frame_paths = self._sample_frame_paths_for_duration(
                    frames, end_idx, video_fps
                )
                if frame_paths is None:
                    continue

                if is_accident and end_frame_1b <= t_co:
                    ttc_frames = t_co - end_frame_1b
                    binary_target = 1.0 if ttc_frames <= horizon_frames else 0.0
                    ttc_frames_10fps = ttc_frames * self.cfg.base_fps / video_fps
                    risk_target = float(risk_target_fn(ttc_frames_10fps))
                    valid_progress = 1.0
                else:
                    ttc_frames = -1
                    binary_target = 0.0
                    risk_target = 0.0
                    valid_progress = 1.0

                if binary_target == 0.0 and rng.random() > self.cfg.neg_keep_prob:
                    continue

                video_samples.append(
                    {
                        "video_hashcode": rec["video_hashcode"],
                        "record": rec,
                        "frame_paths": frame_paths,
                        "future_frame_paths": None,
                        "binary_target": binary_target,
                        "risk_target": risk_target,
                        "valid_progress": valid_progress,
                        "pref_valid": 0.0,
                        "future_risk_target": 0.0,
                        "ttc_frames": ttc_frames,
                        "current_frame_idx_1based": end_frame_1b,
                        "fps": video_fps,
                    }
                )

            pos_pool = [x for x in video_samples if x["binary_target"] > 0.0]
            neg_pool = [x for x in video_samples if x["binary_target"] == 0.0]
            if pos_pool and neg_pool:
                for sample in neg_pool:
                    if rng.random() > self.cfg.pref_keep_prob:
                        continue
                    partner = rng.choice(pos_pool)
                    sample["future_frame_paths"] = partner["frame_paths"]
                    sample["future_risk_target"] = partner["risk_target"]
                    sample["pref_valid"] = 1.0

            video_samples = self._cap_video_samples(video_samples, rng)
            samples.extend(video_samples)

        self._print_train_stats(samples)
        return samples

    def _cap_video_samples(
        self,
        video_samples: List[Dict[str, Any]],
        rng: random.Random,
    ) -> List[Dict[str, Any]]:
        max_per_video = self.cfg.max_samples_per_video
        if max_per_video is None or len(video_samples) <= max_per_video:
            return video_samples

        pos_samples = [x for x in video_samples if x["binary_target"] > 0.0]
        neg_samples = [x for x in video_samples if x["binary_target"] == 0.0]
        if len(pos_samples) >= max_per_video:
            rng.shuffle(pos_samples)
            return pos_samples[:max_per_video]
        remaining = max_per_video - len(pos_samples)
        rng.shuffle(neg_samples)
        return pos_samples + neg_samples[:remaining]

    def _print_train_stats(self, samples: List[Dict[str, Any]]) -> None:
        total = len(samples)
        pos = sum(int(s["binary_target"] > 0.0) for s in samples)
        pref = sum(int(s["pref_valid"] > 0.0) for s in samples)
        ratio = pref / total if total else 0.0
        print(
            f"[DAD train] total={total} pos={pos} pref={pref} pref_ratio={ratio:.4f}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Dataset output -- deliberately matches MMAUAnticipationDataset
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, p: Path):
        img = Image.open(p).convert("RGB")
        if self.cfg.transform is not None:
            return self.cfg.transform(img)
        return img

    def _load_clip(self, paths, augment):
        if getattr(self.cfg.transform, "clip_level", False):
            return self.cfg.transform(
                [Image.open(p).convert("RGB") for p in paths],
                augment=augment,
            )

        return [self._load_image(p) for p in paths]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        s = self.samples[index]

        if self.mode == "anticipation_train":
            n_cur = len(s["frame_paths"])

            paths = list(s["frame_paths"])
            if s["future_frame_paths"] is not None:
                paths += list(s["future_frame_paths"])

            loaded = self._load_clip(paths, augment=True)

            images = loaded[:n_cur]
            future_images = loaded[n_cur:] or None

            return {
                "frames": (
                    torch.stack(images, dim=0)
                    if isinstance(images[0], torch.Tensor)
                    else images
                ),
                "future_frames": (
                    torch.stack(future_images, dim=0)
                    if future_images is not None
                    and isinstance(future_images[0], torch.Tensor)
                    else future_images
                ),
                "binary_target": torch.tensor(
                    s["binary_target"], dtype=torch.float32
                ),
                "risk_target": torch.tensor(s["risk_target"], dtype=torch.float32),
                "valid_progress": torch.tensor(
                    s["valid_progress"], dtype=torch.float32
                ),
                "pref_valid": torch.tensor(s["pref_valid"], dtype=torch.float32),
                "future_risk_target": torch.tensor(
                    s["future_risk_target"], dtype=torch.float32
                ),
                "video_hashcode": s["video_hashcode"],
                "current_frame_idx_1based": s["current_frame_idx_1based"],
                "t_ai": s["record"]["t_ai"],
                "t_co": s["record"]["t_co"],
                "ttc_frames": torch.tensor(s["ttc_frames"], dtype=torch.float32),
                "fps": torch.tensor(s["fps"], dtype=torch.float32),
            }

        images = [self._load_image(p) for p in s["frame_paths"]]

        # Same key structure as MMAUAnticipationDataset. Your existing
        # anticipation_eval_collate_fn_pad therefore receives transformed
        # tensors in out["images"], not untransformed PIL images.
        out = {
            "images": images,
            "video_hashcode": s["video_hashcode"],
            "video_name": s["record"]["video_name"],
            "id": s["record"]["id"],
            "t_ai": s["record"]["t_ai"],
            "t_co": s["record"]["t_co"],
            "t_ae": s["record"]["t_ae"],
            "total_frames": s["record"]["total_frames"],
            "texts": s["record"].get("texts", ""),
            "causes": s["record"].get("causes", ""),
            "measures": s["record"].get("measures", ""),
            "label": s["label"],
            "horizon_sec": s["horizon_sec"],
            "frame_paths": s["frame_paths"],
            "video_folder": s["video_folder"],
        }

        # Include these for all evaluation modes. Existing code can ignore them.
        out["current_frame_idx_1based"] = s.get("current_frame_idx_1based")
        out["ttc_frames"] = s.get("ttc_frames")
        out["ttc_sec"] = s.get("ttc_sec")
        out["fps"] = s.get("fps")
        out["t_ai"] = s.get("t_ai", out["t_ai"])
        out["t_co"] = s.get("t_co", out["t_co"])
        out["t_ae"] = s.get("t_ae", out["t_ae"])
        out["is_after_anomaly"] = s.get("is_after_anomaly")
        out["is_before_collision"] = s.get("is_before_collision")
        return out
