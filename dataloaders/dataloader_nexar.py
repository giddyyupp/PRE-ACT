import csv
import random
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset

from engine.risk_targets import get_risk_target_fn
from dataloaders.helpers import deterministic_record_subsets, print_subset_stats


@dataclass
class NexarConfig:
    root: str
    subset: str = "train"          # train / test-public
    image_size: int = 224
    snippet_len: int = 5
    stride: int = 1
    transform: Optional[Any] = None
    seed: int = 42
    base_fps: int = 10
    video_slice_idx: int = 0
    video_slice_count: int = 1

    anticipation_horizon_sec: float = 2.0
    progress_alpha: float = 5.0
    pair_gap_sec_min: float = 0.5
    pair_gap_sec_max: float = 1.5
    include_post_collision: bool = False
    custom_risk_mode: str = "exp_above"
    random_pos_neg_sampling: bool = False

    train_stride: int = 3
    neg_keep_prob: float = 0.8
    pref_keep_prob: float = 0.9
    max_samples_per_video: Optional[int] = 50 # max negative samples per video, to avoid domination by huge negatives.

    fps: int = 30
    inference_on_train: bool = False

    fraction: Optional[float] = 1.0  # For subset selection, e.g., 0.01, 0.05, 0.10


class NexarAnticipationDataset(Dataset):
    def __init__(
        self,
        cfg: NexarConfig,
        mode: str = "anticipation_train",
        horizons_sec: Tuple[float, ...] = (0.5, 1.0, 1.5),
        clip_duration_sec: float = 0.5,
    ):
        self.cfg = cfg
        self.mode = mode
        self.horizons_sec = horizons_sec
        self.clip_duration_sec = clip_duration_sec
        self.root = Path(cfg.root)

        self.records = self._load_records()
        self.video_index = {}
        
        if self.cfg.subset == 'train':
            self.video_index = self._build_video_index(self.cfg.subset)
        else:
            self.video_index.update(self._build_video_index('test-public'))
            self.video_index.update(self._build_video_index('test-private'))

        self.records = self._filter_existing_records(self.records)

        if not self.cfg.subset == "train":
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
                label_key="label",
            )
            print_subset_stats(
                self.subsets,
                label_fn=lambda r: r["label"],
            )

            self.records = self.subsets[self.cfg.fraction]

            if self.cfg.random_pos_neg_sampling:
                self.samples = self._build_anticipation_train_samples_fps_fix_random_pos_neg_pairs()
            else:
                self.samples = self._build_anticipation_train_samples_fps_fix_subsample()
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------

    def _load_pos_records(self, pos_meta_csv) -> List[Dict[str, Any]]:

        records = []
        with open(pos_meta_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_name = str(row["file_name"]).strip()
                vid = Path(file_name).stem
                fps = self.cfg.fps
                time_of_event = float(row["time_of_event"]) if row.get("time_of_event") else None
                time_of_alert = float(row["time_of_alert"]) if row.get("time_of_alert") else None

                t_co = int(round(time_of_event * fps)) if time_of_event is not None else -1
                t_ai = int(round(time_of_alert * fps)) if time_of_alert is not None else -1

                records.append({
                    "video_hashcode": vid,
                    "video_name": vid,
                    "video_id": vid,
                    "label": 1,
                    "fps": fps,
                    "t_ai": t_ai,
                    "t_co": t_co,
                    "t_ae": -1,
                    "total_frames": -1,
                    "light_conditions": row.get("light_conditions", ""),
                    "weather": row.get("weather", ""),
                    "scene": row.get("scene", ""),
                    "time_to_accident_sec": float(row["time_to_accident"]) if row.get("time_to_accident") not in [None, ""] else None,
                })

        return records

    def _load_neg_records(self, neg_meta_csv) -> List[Dict[str, Any]]:

        records = []
        with open(neg_meta_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_name = str(row["file_name"]).strip()
                vid = Path(file_name).stem
                fps = self.cfg.fps

                records.append({
                    "video_hashcode": vid,
                    "video_name": vid,
                    "video_id": vid,
                    "label": 0,
                    "fps": fps,
                    "t_ai": -1,
                    "t_co": -1,
                    "t_ae": -1,
                    "total_frames": -1,
                    "light_conditions": row.get("light_conditions", ""),
                    "weather": row.get("weather", ""),
                    "scene": row.get("scene", ""),
                    "time_to_accident_sec": float(row["time_to_accident"]) if row.get("time_to_accident") not in [None, ""] else None,
                })

        return records

    def _load_records(self) -> List[Dict[str, Any]]:

        records = []

        if self.cfg.subset == 'train':
            split_root = self.root / self.cfg.subset
            pos_meta_csv = split_root / "positive" / "metadata.csv"
            neg_meta_csv = split_root / "negative" / "metadata.csv"

            if not pos_meta_csv.exists():
                raise FileNotFoundError(f"Missing file: {pos_meta_csv}")
            if not neg_meta_csv.exists():
                raise FileNotFoundError(f"Missing file: {neg_meta_csv}")
            
            records.extend(self._load_pos_records(pos_meta_csv))
            records.extend(self._load_neg_records(neg_meta_csv))
        else:
            # public split
            split_root_public = self.root / "test-public"
            pos_meta_csv = split_root_public / "positive" / "metadata.csv"
            neg_meta_csv = split_root_public / "negative" / "metadata.csv"

            if not pos_meta_csv.exists():
                raise FileNotFoundError(f"Missing file: {pos_meta_csv}")
            if not neg_meta_csv.exists():
                raise FileNotFoundError(f"Missing file: {neg_meta_csv}")
            
            records.extend(self._load_pos_records(pos_meta_csv))
            records.extend(self._load_neg_records(neg_meta_csv))

            # private split
            split_root_private = self.root / "test-private"
            pos_meta_csv = split_root_private / "positive" / "metadata.csv"
            neg_meta_csv = split_root_private / "negative" / "metadata.csv"

            if not pos_meta_csv.exists():
                raise FileNotFoundError(f"Missing file: {pos_meta_csv}")
            if not neg_meta_csv.exists():
                raise FileNotFoundError(f"Missing file: {neg_meta_csv}")
            
            records.extend(self._load_pos_records(pos_meta_csv))
            records.extend(self._load_neg_records(neg_meta_csv))

        return records

    # --------------------------------------------------
    # Video indexing
    # --------------------------------------------------

    def _get_eval_horizon_sec(self, rec: Dict[str, Any]) -> float:
        """
        For Nexar test metadata, prefer per-video time_to_accident if available.
        Otherwise fall back to the global config horizon.
        """
        tta_meta = rec.get("time_to_accident_sec", None)
        if tta_meta is not None:
            return float(tta_meta)
        return float(self.cfg.anticipation_horizon_sec)

    def _build_video_index(self, subset: str) -> Dict[str, Path]:
        index = {}
        split_root = self.root / subset

        for cls_name in ["positive", "negative"]:
            cls_root = split_root / cls_name
            if not cls_root.exists():
                continue

            for video_dir in sorted(cls_root.iterdir()):
                if not video_dir.is_dir():
                    continue

                images_dir = video_dir / "images"
                if images_dir.exists():
                    index[video_dir.name] = images_dir

        return index

    def _filter_existing_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered, missing = [], []
        for r in records:
            vid = r["video_id"]
            if vid in self.video_index:
                filtered.append(r)
            else:
                missing.append(vid)

        print(f"[Nexar {self.cfg.subset}] kept {len(filtered)} records, dropped {len(missing)} missing")
        if missing:
            print(f"[Nexar {self.cfg.subset}] first 10 missing:", missing[:10])
        return filtered

    def _resolve_video_dir(self, rec: Dict[str, Any]) -> Path:
        vid = rec["video_id"]
        if vid not in self.video_index:
            raise FileNotFoundError(f"Could not find frame folder for {vid}")
        return self.video_index[vid]

    def _list_frames(self, image_dir: Path) -> List[Path]:
        frames = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        frames = sorted(frames)
        if not frames:
            raise FileNotFoundError(f"No frames found in {image_dir}")
        return frames

    def _sample_frame_paths_for_duration(
        self,
        frames: List[Path],
        end_idx: int,
        video_fps: float,
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

    def _slice_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        slice_count = int(self.cfg.video_slice_count)
        slice_idx = int(self.cfg.video_slice_idx)

        if slice_count < 1:
            raise ValueError(f"video_slice_count must be >= 1, got {slice_count}")
        if not (0 <= slice_idx < slice_count):
            raise ValueError(
                f"video_slice_idx must be in [0, {slice_count - 1}], got {slice_idx}"
            )

        if slice_count == 1:
            return records

        n = len(records)
        start = (n * slice_idx) // slice_count
        end = (n * (slice_idx + 1)) // slice_count
        return records[start:end]
    
    # --------------------------------------------------
    # Training builders
    # --------------------------------------------------

    def _build_anticipation_train_samples_fps_fix_subsample(self) -> List[Dict[str, Any]]:
        samples = []
        stride = getattr(self.cfg, "train_stride", self.cfg.stride)
        rng = random.Random(self.cfg.seed)
        risk_target_fn = get_risk_target_fn(self.cfg)

        for rec in self.records:
            image_dir = self._resolve_video_dir(rec)
            frames = self._list_frames(image_dir)
            total_frames = len(frames)
            current_fps = float(rec.get("fps", self.cfg.fps))

            rec["total_frames"] = total_frames
            rec["fps"] = current_fps

            horizon_frames = int(round(self.cfg.anticipation_horizon_sec * current_fps))
            t_co = int(rec["t_co"])
            is_accident = int(t_co >= 0)

            sample_stride = max(1, int(round(current_fps / self.cfg.base_fps)))
            first_valid_end_idx = (self.cfg.snippet_len - 1) * sample_stride

            video_samples = []

            endpoint_stride = stride * sample_stride

            for end_idx in range(
                first_valid_end_idx,
                total_frames,
                endpoint_stride,
            ):
                end_frame_1b = end_idx + 1

                if is_accident and (not self.cfg.include_post_collision) and end_frame_1b > t_co:
                    continue

                frame_paths = self._sample_frame_paths_for_duration(
                    frames=frames,
                    end_idx=end_idx,
                    video_fps=current_fps,
                )
                if frame_paths is None:
                    continue

                if is_accident and end_frame_1b <= t_co:
                    ttc_frames = t_co - end_frame_1b
                    binary_target = 1.0 if ttc_frames <= horizon_frames else 0.0
                    ttc_frames_10fps = ttc_frames * self.cfg.base_fps / current_fps
                    risk_target = risk_target_fn(ttc_frames_10fps)
                    valid_progress = 1.0
                else:
                    ttc_frames = -1
                    binary_target = 0.0
                    risk_target = 0.0
                    valid_progress = 1.0

                if binary_target == 0.0 and rng.random() > getattr(self.cfg, "neg_keep_prob", 1.0):
                    continue

                future_frame_paths = None
                pref_valid = 0.0
                future_risk_target = 0.0

                pair_gap_frames = rng.randint(
                    int(round(self.cfg.pair_gap_sec_min * current_fps)),
                    int(round(self.cfg.pair_gap_sec_max * current_fps)),
                )

                if is_accident and end_frame_1b <= t_co:
                    future_end_frame_1b = min(t_co, end_frame_1b + pair_gap_frames)
                    future_end_idx = future_end_frame_1b - 1

                    future_frame_paths = self._sample_frame_paths_for_duration(
                        frames=frames,
                        end_idx=future_end_idx,
                        video_fps=current_fps,
                    )

                    if future_frame_paths is not None and future_end_frame_1b > end_frame_1b:
                        future_ttc_frames = t_co - future_end_frame_1b
                        future_ttc_frames_10fps = future_ttc_frames * self.cfg.base_fps / current_fps
                        future_risk_target = risk_target_fn(future_ttc_frames_10fps)

                        if future_risk_target > risk_target:
                            if rng.random() <= getattr(self.cfg, "pref_keep_prob", 1.0):
                                pref_valid = 1.0
                            else:
                                future_frame_paths = None
                                future_risk_target = 0.0
                                pref_valid = 0.0

                video_samples.append({
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
                    "fps": current_fps,
                })

            max_per_video = getattr(self.cfg, "max_samples_per_video", 50)
            if max_per_video is not None and len(video_samples) > max_per_video:
                pos_samples = [x for x in video_samples if x["binary_target"] > 0.0]
                neg_samples = [x for x in video_samples if x["binary_target"] == 0.0]

                if len(neg_samples) > max_per_video:
                    rng.shuffle(neg_samples)
                    neg_samples = neg_samples[:max_per_video]

                video_samples = pos_samples + neg_samples

            samples.extend(video_samples)

        num_total = len(samples)
        num_pos = sum(1 for s in samples if s["binary_target"] > 0.0)
        num_pref = sum(1 for s in samples if s["pref_valid"] > 0.0)
        print(f"total={num_total} pos={num_pos} pref={num_pref} pref_ratio={num_pref/num_total:.4f}")
        return samples

    def _build_anticipation_train_samples_fps_fix_random_pos_neg_pairs(self) -> List[Dict[str, Any]]:
        samples = []
        stride = getattr(self.cfg, "train_stride", self.cfg.stride)
        rng = random.Random(self.cfg.seed)
        risk_target_fn = get_risk_target_fn(self.cfg)

        for rec in self.records:
            image_dir = self._resolve_video_dir(rec)
            frames = self._list_frames(image_dir)
            total_frames = len(frames)
            current_fps = float(rec.get("fps", self.cfg.fps))

            rec["total_frames"] = total_frames
            rec["fps"] = current_fps

            horizon_frames = int(round(self.cfg.anticipation_horizon_sec * current_fps))
            t_co = int(rec["t_co"])
            is_accident = int(t_co >= 0)

            sample_stride = max(1, int(round(current_fps / self.cfg.base_fps)))
            first_valid_end_idx = (self.cfg.snippet_len - 1) * sample_stride

            video_samples = []

            for end_idx in range(first_valid_end_idx, total_frames, stride):
                end_frame_1b = end_idx + 1

                if is_accident and (not self.cfg.include_post_collision) and end_frame_1b > t_co:
                    continue

                frame_paths = self._sample_frame_paths_for_duration(
                    frames=frames,
                    end_idx=end_idx,
                    video_fps=current_fps,
                )
                if frame_paths is None:
                    continue

                if is_accident and end_frame_1b <= t_co:
                    ttc_frames = t_co - end_frame_1b
                    binary_target = 1.0 if ttc_frames <= horizon_frames else 0.0
                    ttc_frames_10fps = ttc_frames * self.cfg.base_fps / current_fps
                    risk_target = risk_target_fn(ttc_frames_10fps)
                    valid_progress = 1.0
                else:
                    ttc_frames = -1
                    binary_target = 0.0
                    risk_target = 0.0
                    valid_progress = 1.0

                if binary_target == 0.0 and rng.random() > getattr(self.cfg, "neg_keep_prob", 1.0):
                    continue

                video_samples.append({
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
                    "fps": current_fps,
                })

            pos_pool = [x for x in video_samples if x["binary_target"] > 0.0]
            neg_pool = [x for x in video_samples if x["binary_target"] == 0.0]

            if len(pos_pool) > 0 and len(neg_pool) > 0:
                for sample in neg_pool:
                    if rng.random() > getattr(self.cfg, "pref_keep_prob", 1.0):
                        continue

                    pos_partner = rng.choice(pos_pool)
                    sample["future_frame_paths"] = pos_partner["frame_paths"]
                    sample["future_risk_target"] = pos_partner["risk_target"]
                    sample["pref_valid"] = 1.0

            max_per_video = getattr(self.cfg, "max_samples_per_video", 50)
            if max_per_video is not None and len(video_samples) > max_per_video:
                pos_samples = [x for x in video_samples if x["binary_target"] > 0.0]
                neg_samples = [x for x in video_samples if x["binary_target"] == 0.0]

                if len(neg_samples) > max_per_video:
                    rng.shuffle(neg_samples)
                    neg_samples = neg_samples[:max_per_video]

                video_samples = pos_samples + neg_samples

            samples.extend(video_samples)

        num_total = len(samples)
        num_pos = sum(1 for s in samples if s["binary_target"] > 0.0)
        num_pref = sum(1 for s in samples if s["pref_valid"] > 0.0)
        print(f"total={num_total} pos={num_pos} pref={num_pref} pref_ratio={num_pref/num_total:.4f}")
        return samples

    # --------------------------------------------------
    # Eval builders
    # --------------------------------------------------

    def _tta_test(self) -> List[Dict[str, Any]]:
        samples = []

        for rec in self.records:
            image_dir = self._resolve_video_dir(rec)
            frames = self._list_frames(image_dir)
            total_frames = len(frames)
            video_fps = float(rec.get("fps", self.cfg.fps))

            rec["total_frames"] = total_frames
            rec["fps"] = video_fps

            t_ai = int(rec["t_ai"])
            t_co = int(rec["t_co"])
            t_ae = int(rec["t_ae"])

            if t_co < 0:
                continue

            horizon_sec = self._get_eval_horizon_sec(rec)
            horizon_frames = int(round(horizon_sec * video_fps))    

            sample_stride = max(1, int(round(video_fps / self.cfg.base_fps)))
            first_valid_end_1b = (self.cfg.snippet_len - 1) * sample_stride + 1
            end_end_1b = min(t_co, total_frames)

            for end_idx_1b in range(first_valid_end_1b, end_end_1b + 1, 1):
                end_idx = end_idx_1b - 1

                frame_paths = self._sample_frame_paths_for_duration(
                    frames=frames,
                    end_idx=end_idx,
                    video_fps=video_fps,
                )
                if frame_paths is None:
                    continue

                ttc_frames = t_co - end_idx_1b
                ttc_sec = ttc_frames / float(video_fps)
                label = 1 if ttc_frames <= horizon_frames else 0

                samples.append({
                    "video_hashcode": rec["video_hashcode"],
                    "record": rec,
                    "frame_paths": frame_paths,
                    "label": label,
                    "horizon_sec": horizon_sec,
                    "video_folder": str(rec["video_name"]),
                    "current_frame_idx_1based": end_idx_1b,
                    "ttc_frames": ttc_frames,
                    "ttc_sec": ttc_sec,
                    "fps": video_fps,
                    "t_ai": t_ai,
                    "t_co": t_co,
                    "t_ae": t_ae,
                    "is_after_anomaly": int(end_idx_1b >= t_ai) if t_ai >= 0 else None,
                    "is_before_collision": int(end_idx_1b <= t_co),
                })

        return samples

    def _build_binary_clip_samples_fps_fix_TOP_like_sliding_window(self) -> List[Dict[str, Any]]:
        samples = []

        eval_dict = {}
        """
        "000001": {
        "accident_type": 10,
        "abnormal_start_frame": 18,
        "accident_frame": 34,
        "abnormal_end_frame": 50,
        "num_images": 50,
        "fps": 10
        },
        """

        for rec in self.records:
            image_dir = self._resolve_video_dir(rec)
            frames = self._list_frames(image_dir)
            total_frames = len(frames)
            video_fps = float(rec.get("fps", self.cfg.fps))

            rec["total_frames"] = total_frames
            rec["fps"] = video_fps

            # t_ai = int(rec["t_ai"])
            # t_co = int(rec["t_co"])
            t_ae = int(rec.get("t_ae", -1))

            time_to_accident = rec.get("time_to_accident_sec", None)

            # t_ai = int(rec["t_ai"])
            if time_to_accident is not None and time_to_accident >= 0:
                t_ai = total_frames - (self.cfg.anticipation_horizon_sec - time_to_accident) * video_fps  # int(total_frames / 2)  # For Nexar use 2.0 sec before accident as proxy for anomaly point since t_ai is often missing or unreliable
            else: # no accident videos, t_ai could be after the end of the video, but we will set it to -1 to indicate no anomaly point
                t_ai = -1

            if rec["label"] == 1:
                t_co = total_frames + time_to_accident * video_fps if time_to_accident is not None else total_frames  # For positives, set t_co based on time_to_accident if available, otherwise use end of video. This allows sampling positive clips from the entire video up until the accident time.
            else:
                t_co = total_frames + video_fps * 5  # For negatives, set a pseudo t_co that's after the end of the video to allow sampling negative clips from the entire video        

            # horizon_sec = self._get_eval_horizon_sec(rec)
            # horizon_frames = int(round(horizon_sec * video_fps))

            eval_dict[rec["video_hashcode"]] = {
                "accident_type": rec["weather"],  # Using weather as a proxy for accident type since actual accident type is not provided in metadata
                "abnormal_start_frame": t_ai if rec["label"] == 1 else None,
                "accident_frame": t_co if t_co >= 0 else None,
                "abnormal_end_frame": t_co if t_co >= 0 else None,
                "num_images": total_frames,
                "fps": video_fps,
                "time_to_accident_sec": rec.get("time_to_accident_sec", None),
            }    

            sample_stride = max(1, int(round(video_fps / self.cfg.base_fps)))
            first_valid_end_1b = (self.cfg.snippet_len - 1) * sample_stride + 1

            end_end_1b = min(t_co, total_frames) if t_co >= 0 else total_frames

            for end_idx_1b in range(first_valid_end_1b, end_end_1b + 1, sample_stride):
                end_idx = end_idx_1b - 1
                frame_paths = self._sample_frame_paths_for_duration(
                    frames=frames,
                    end_idx=end_idx,
                    video_fps=video_fps,
                )
                if frame_paths is None:
                    continue

                if t_co >= 0:
                    ttc_frames = t_co - end_idx_1b
                    ttc_sec = ttc_frames / float(video_fps)
                    label = 1 if ttc_frames <= self.cfg.anticipation_horizon_sec * video_fps else 0
                else:
                    ttc_frames = -1
                    ttc_sec = -1.0
                    label = 0

                samples.append({
                    "video_hashcode": rec["video_hashcode"],
                    "record": rec,
                    "frame_paths": frame_paths,
                    "label": label,
                    "horizon_sec": ttc_sec,
                    "current_frame_idx_1based": end_idx_1b,
                    "video_folder": str(rec["video_name"]),
                    "ttc_frames": ttc_frames,
                    "ttc_sec": ttc_sec,
                    "fps": video_fps,
                    "t_ai": t_ai,
                    "t_co": t_co,
                    "t_ae": t_ae,
                    "is_after_anomaly": int(t_co >= 0 and t_ai >= 0 and end_idx_1b >= t_ai),
                    "is_before_collision": int(t_co >= 0 and end_idx_1b <= t_co),
                })

        with open("nexar_anno.json", "w") as f:
            json.dump(eval_dict, f)
        
        return samples

    def _build_binary_clip_samples_fps_fix_TOP_like(self) -> List[Dict[str, Any]]:
        samples = []

        for rec in self.records:
            image_dir = self._resolve_video_dir(rec)
            frames = self._list_frames(image_dir)
            total_frames = len(frames)
            video_fps = float(rec.get("fps", self.cfg.fps))

            rec["total_frames"] = total_frames
            rec["fps"] = video_fps
            time_to_accident = rec.get("time_to_accident_sec", None)

            t_ai = int(total_frames / 2)  # For Nexar use midpoint as proxy for anomaly point since t_ai is often missing or unreliable
            
            if rec["label"] == 1:
                t_co = total_frames + time_to_accident * video_fps if time_to_accident is not None else total_frames  # For positives, set t_co based on time_to_accident if available, otherwise use end of video. This allows sampling positive clips from the entire video up until the accident time.
            else:
                t_co = total_frames + video_fps * 5  # For negatives, set a pseudo t_co that's after the end of the video to allow sampling negative clips from the entire video                   

            sample_stride = max(1, int(round(video_fps / self.cfg.base_fps)))
            first_valid_end_1b = (self.cfg.snippet_len - 1) * sample_stride + 1

            # horizon_sec = self._get_eval_horizon_sec(rec)
            horizon_sec = self.cfg.anticipation_horizon_sec
            neg_end_max = t_co - int(round(horizon_sec * video_fps)) - 1 if rec["label"] > 0 else total_frames

            if neg_end_max >= first_valid_end_1b:
                for end_idx_1b in range(first_valid_end_1b, neg_end_max + 1, 1):
                    end_idx = end_idx_1b - 1

                    frame_paths = self._sample_frame_paths_for_duration(
                        frames=frames,
                        end_idx=end_idx,
                        video_fps=video_fps,
                    )
                    if frame_paths is None:
                        continue

                    samples.append({
                        "video_hashcode": rec["video_hashcode"],
                        "record": rec,
                        "frame_paths": frame_paths,
                        "label": 0,
                        "horizon_sec": None,
                        "current_frame_idx_1based": end_idx_1b,
                        "video_folder": str(rec["video_name"]),
                    })

            # if t_co < 0:
            #     continue

            if rec["label"] >= 1:
                # horizon_sec = self._get_eval_horizon_sec(rec)
                target_frame_1b = t_co - int(round(horizon_sec * video_fps))

                if target_frame_1b >= first_valid_end_1b:
                    end_idx = target_frame_1b - 1
                    frame_paths = self._sample_frame_paths_for_duration(
                        frames=frames,
                        end_idx=end_idx,
                        video_fps=video_fps,
                    )
                    if frame_paths is not None:
                        samples.append({
                            "video_hashcode": rec["video_hashcode"],
                            "record": rec,
                            "frame_paths": frame_paths,
                            "label": 1,
                            "horizon_sec": horizon_sec,
                            "video_folder": str(rec["video_name"]),
                        })

        return samples

    def _build_binary_clip_samples_fps_fix(self) -> List[Dict[str, Any]]:
        samples = []

        for rec in self.records:
            image_dir = self._resolve_video_dir(rec)
            frames = self._list_frames(image_dir)
            total_frames = len(frames)
            video_fps = float(rec.get("fps", self.cfg.fps))

            rec["total_frames"] = total_frames
            rec["fps"] = video_fps
            time_to_accident = rec.get("time_to_accident_sec", None)

            # t_ai = int(rec["t_ai"])
            t_ai = int(total_frames / 2)  # For Nexar use midpoint as proxy for anomaly point since t_ai is often missing or unreliable
            
            if rec["label"] == 1:
                t_co = total_frames + time_to_accident * video_fps if time_to_accident is not None else total_frames  # For positives, set t_co based on time_to_accident if available, otherwise use end of video. This allows sampling positive clips from the entire video up until the accident time.
            else:
                t_co = total_frames + video_fps * 5  # For negatives, set a pseudo t_co that's after the end of the video to allow sampling negative clips from the entire video                   

            sample_stride = max(1, int(round(video_fps / self.cfg.base_fps)))
            clip_span_frames = max(self.cfg.snippet_len, int(round(self.clip_duration_sec * video_fps)))
            first_valid_end_1b = (self.cfg.snippet_len - 1) * sample_stride + 1

            neg_end_max = t_ai - 1
            for end_idx_1b in range(first_valid_end_1b, neg_end_max + 1, clip_span_frames):
                end_idx = end_idx_1b - 1
                frame_paths = self._sample_frame_paths_for_duration(
                    frames=frames,
                    end_idx=end_idx,
                    video_fps=video_fps,
                )
                if frame_paths is None:
                    continue

                samples.append({
                    "video_hashcode": rec["video_hashcode"],
                    "record": rec,
                    "frame_paths": frame_paths,
                    "label": 0,
                    "horizon_sec": None,
                    "video_folder": str(rec["video_name"]),
                })

            for h in self.horizons_sec:
                target_frame_1b = t_co - int(round(h * video_fps))
                if target_frame_1b < first_valid_end_1b:
                    continue

                end_idx = target_frame_1b - 1
                frame_paths = self._sample_frame_paths_for_duration(
                    frames=frames,
                    end_idx=end_idx,
                    video_fps=video_fps,
                )
                if frame_paths is None:
                    continue

                samples.append({
                    "video_hashcode": rec["video_hashcode"],
                    "record": rec,
                    "frame_paths": frame_paths,
                    "label": 1,
                    "horizon_sec": h,
                    "video_folder": str(rec["video_name"]),
                })

        return samples

    # --------------------------------------------------
    # Dataset API
    # --------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def _apply_transform(self, img: Image.Image):
        if self.cfg.transform is not None:
            return self.cfg.transform(img)
        return img

    def _load_image(self, p):
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
                "frames": torch.stack(images, dim=0) if isinstance(images[0], torch.Tensor) else images,
                "future_frames": (
                    torch.stack(future_images, dim=0)
                    if (future_images is not None and isinstance(future_images[0], torch.Tensor))
                    else future_images
                ),
                "binary_target": torch.tensor(s["binary_target"], dtype=torch.float32),
                "risk_target": torch.tensor(s["risk_target"], dtype=torch.float32),
                "valid_progress": torch.tensor(s["valid_progress"], dtype=torch.float32),
                "pref_valid": torch.tensor(s["pref_valid"], dtype=torch.float32),
                "future_risk_target": torch.tensor(s["future_risk_target"], dtype=torch.float32),
                "video_hashcode": s["video_hashcode"],
                "current_frame_idx_1based": s["current_frame_idx_1based"],
                "t_ai": s["record"]["t_ai"],
                "t_co": s["record"]["t_co"],
                "ttc_frames": torch.tensor(s["ttc_frames"], dtype=torch.float32),
                "fps": torch.tensor(s["fps"], dtype=torch.float32),
            }

        # existing branches below
        images = [self._load_image(p) for p in s["frame_paths"]]

        out = {
            "images": images,
            "video_hashcode": s["video_hashcode"],
            "video_name": s["record"]["video_name"],
            "t_ai": s["record"]["t_ai"],
            "t_co": s["record"]["t_co"],
            "t_ae": s["record"]["t_ae"],
            "total_frames": s["record"]["total_frames"],
            "frame_paths": s["frame_paths"],
        }

        if self.mode == "test_full":
            out["label"] = s["label"]
            out["horizon_sec"] = s["horizon_sec"]
            out["current_frame_idx_1based"] = s.get("current_frame_idx_1based", None)
            out["ttc_frames"] = s.get("ttc_frames", None)
            out["ttc_sec"] = s.get("ttc_sec", None)
            out["fps"] = s.get("fps", None)
            out["t_ai"] = s.get("t_ai", out["t_ai"])
            out["t_co"] = s.get("t_co", out["t_co"])
            out["t_ae"] = s.get("t_ae", out["t_ae"])
            out["is_after_anomaly"] = s.get("is_after_anomaly", None)
            out["is_before_collision"] = s.get("is_before_collision", None)
            out["video_folder"] = s["video_folder"]
        else:
            out["label"] = s["label"]
            out["horizon_sec"] = s["horizon_sec"]
            out["video_folder"] = s["video_folder"]

        return out
