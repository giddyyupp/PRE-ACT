import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

import torch
from torch.utils.data import Dataset

from dataloaders.splits import cap_test, dada_test
from engine.risk_targets import get_risk_target_fn, full_video_progress_risk_fn
from dataloaders.helpers import deterministic_record_subsets, print_subset_stats


class QwenTrainWrapper(Dataset):
    def __init__(self, base_ds, prompt):
        self.base_ds = base_ds
        self.prompt = prompt

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x = self.base_ds[idx]
        return {
            "images": x["images"],
            "prompt": self.prompt,
            "answer": "Yes" if x["label"] == 1 else "No",
            "video_hashcode": x["video_hashcode"],
        }
    

@dataclass
class MMAUConfig:
    root: str
    metadata_json: str
    subset: str = "CAP"            # "CAP" or "DADA"
    fps: int = 10                  # 10 FPS
    image_size: int = 224          # 224x224
    snippet_len: int = 5           # S = 5
    stride: int = 1                # sliding window stride at test time
    transform: Optional[Any] = None
    split_json: Optional[str] = None   # optional official/custom split file
    split_name: str = "test"           # train / val / test
    seed: int = 42
    # rank: Optional[int] = None
    # world_size: Optional[int] = None
    video_slice_idx: int = 0
    video_slice_count: int = 1
    base_fps: int = 10
    full_video_progress_risk: bool = False

    # new for encoder training
    anticipation_horizon_sec: float = 2.0
    pair_gap_sec_min: float = 0.5
    pair_gap_sec_max: float = 1.5
    include_post_collision: bool = False
    custom_risk_mode: str = "exp_above"  # "linear", "exp_above", "exp_under"
    progress_alpha: float = 5.0
    random_pos_neg_sampling: bool = False

    # subsampling params for anticipation_train mode, to keep training set size manageable
    train_stride: int = 5          # use this for anticipation_train, separate from test stride
    neg_keep_prob: float = 0.8     # keep 80% of negatives
    pref_keep_prob: float = 0.9    # keep 90% of valid ranking pairs
    max_samples_per_video: int = 50  # max negative samples per video, to avoid domination by huge negatives.
    inference_on_train: bool = False          # if True, build test_full style samples even in train split for zero-shot evaluation on training videos

    fraction: Optional[float] = 1.0  # For subset selection, e.g., 0.01, 0.05, 0.10


class MMAUAnticipationDataset(Dataset):
    """
    MM-AU dataset for accident anticipation.

    Supports two main modes:
    1. mode='test_full':
       returns every sliding-window snippet in each video, along with metadata
       so you can run zero-shot Qwen2-VL and build frame/snippet scores.

    2. mode='binary_clips':
       builds positive and negative clip snippets for binary classification:
       - negatives from [0, t_ai)
       - positives from horizon windows before t_co
         e.g. 0.5s, 1.0s, 1.5s before collision.
    """
    def __init__(
        self,
        cfg: MMAUConfig,
        mode: str = "anticipation_train",
        horizons_sec: Tuple[float, ...] = (0.5, 1.0, 1.5),
        clip_duration_sec: float = 0.5,
    ):
        self.cfg = cfg
        print(f"Initializing MMAUAnticipationDataset with cfg={cfg} with mode={mode}")
        self.mode = mode
        self.horizons_sec = horizons_sec
        self.clip_duration_sec = clip_duration_sec
        self.root = Path(cfg.root)
        self.metadata = self._load_metadata(Path(cfg.metadata_json))
        self.records = self._select_subset(self.metadata, cfg.subset)
        # self.records = self._apply_split(self.records, cfg.split_json, cfg.split_name, cfg.seed)

        if not self.cfg.split_name == "train":
            self.records = self._slice_records(self.records)

        # if cfg.rank is not None and cfg.world_size is not None:
        #     self.records = [r for i, r in enumerate(self.records) if i % cfg.world_size == cfg.rank]

        self.cap_video_index = self._build_cap_video_index() if cfg.subset.upper() == "CAP" else None
        self.dada_video_index = self._build_dada_video_index() if cfg.subset.upper() == "DADA" else None

        if cfg.subset.upper() == "CAP":
            self.records = self._filter_existing_cap_records(self.records)
        elif cfg.subset.upper() == "DADA":
            self.records = self._filter_existing_dada_records(self.records)

        # FILTERING BASED ON ACCIDENT TYPE (ONLY KEEPING EGO ACCIDENTS)
        filtered_records = []
        for rec in self.records:
            if not rec.get("type") < 19:  # NOTE: filter out non-ego accidents
                continue
            filtered_records.append(rec)

        self.records = filtered_records

        if mode == "test_full":
            self.samples = self._tta_test()
        elif mode == "binary_clips":
            # self.samples = self._build_binary_clip_samples()
            if self.cfg.inference_on_train:
                self.samples = self._build_binary_clip_samples_fps_fix()  # few negative clips
            else:
                self.samples = self._build_binary_clip_samples_fps_fix_TOP_like()  #
        elif mode == "sliding_window":
            self.samples = self._build_binary_clip_samples_fps_fix_TOP_like_sliding_window()  # 
        
        elif mode == "anticipation_train":
            # SUBSET SELECTION LOGIC -- Applies only to training mode, not test/eval modes
            print(
                f"\n[SUBSET BEFORE] "
                f"records={len(self.records)} | "
                f"fraction={self.cfg.fraction} | "
                f"expected={round(len(self.records) * self.cfg.fraction)}"
            )
            self.subsets = deterministic_record_subsets(
                self.records,
                fractions=(self.cfg.fraction,),
                seed=42,
                stratify_by_label=False,
                id_key="video_hashcode",
            )
            print_subset_stats(self.subsets)     

            self.records = self.subsets[self.cfg.fraction]

            if self.cfg.random_pos_neg_sampling:
                self.samples = self._build_anticipation_train_samples_fps_fix_random_pos_neg_pairs()
            else:
                self.samples = self._build_anticipation_train_samples_fps_fix_subsample()
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # anticipation_train samples carry a scalar "binary_target";
        # binary_clips / sliding_window builders carry an int "label"; 
        # dataset carries a length-20 tensor. Handle all three.
        def _is_positive(sample):
            target = sample.get("binary_target", sample.get("label", 0))
            return float(target.sum() if hasattr(target, "sum") else target) > 0.0

        num_positive = sum(1 for sample in self.samples if _is_positive(sample))
        
        print(
            f"[MM-AU] subset={cfg.subset}, split={cfg.split_name}, "
            f"mode={mode}, videos={len(self.records)}, "
            f"samples={len(self.samples)}, positive_targets={num_positive}",
            flush=True,
        )
    
    def _cap_key_from_record(self, rec: Dict[str, Any]):
        video_name = str(rec["video_name"]).strip()
        parts = video_name.split("_")
        if len(parts) != 2:
            raise ValueError(f"Unexpected CAP video_name format: {video_name}")

        # Use the metadata type/category as currently stored in your JSON
        cap_category = str(int(rec["type"]))
        video_folder = f"{int(parts[1]):06d}"
        return cap_category, video_folder

    def _dada_key_from_record(self, rec: Dict[str, Any]):
        """
        DADA folder mapping:
        video_name like "61_20" -> category "61", folder "020"

        Filesystem example:
        .../DADA2000/61/020/images
        """
        video_name = str(rec["video_name"]).strip()
        parts = video_name.split("_")
        if len(parts) != 2:
            raise ValueError(f"Unexpected DADA video_name format: {video_name}")

        category_id = str(int(parts[0]))
        video_folder = f"{int(parts[1]):03d}"
        return category_id, video_folder

    def _get_fps(self, video_id, dataset_type="CAP"):
        assert dataset_type in ["CAP", "DADA"]  # 仅支持 'CAP' 和 'DADA' 数据集
        if dataset_type == "CAP":
            if "000001" <= video_id <= "006381":
                return 10
            elif "006382" <= video_id <= "007887":
                return 30
            elif "007888" <= video_id <= "009046":
                return 20
            elif "009047" <= video_id <= "011770":
                return 30
            elif "013001" <= video_id <= "014490":
                return 10
            else:
                return None
        else:  # dataset_type == 'DADA'
            return 30

    def _map_orig_to_stored_frame_idx(
        self,
        orig_idx_1b: int,
        orig_total_frames: int,
        stored_total_frames: int,
    ) -> int:
        """
        Map 1-based frame index from original video timeline to stored-frame timeline.
        """
        if orig_total_frames <= 1 or stored_total_frames <= 1:
            return 1

        mapped = 1 + round((orig_idx_1b - 1) * (stored_total_frames - 1) / (orig_total_frames - 1))
        return max(1, min(stored_total_frames, mapped))


    def _get_mapped_event_frames(self, rec: Dict[str, Any], stored_total_frames: int):
        orig_total = int(rec["total_frames"])

        t_ai = self._map_orig_to_stored_frame_idx(
            int(rec["t_ai"]), orig_total, stored_total_frames
        )

        t_co_raw = int(rec["t_co"])
        t_co = (
            self._map_orig_to_stored_frame_idx(t_co_raw, orig_total, stored_total_frames)
            if t_co_raw >= 0 else -1
        )

        t_ae = self._map_orig_to_stored_frame_idx(
            int(rec["t_ae"]), orig_total, stored_total_frames
        )

        return t_ai, t_co, t_ae

    def _build_anticipation_train_samples_fps_fix_subsample(self) -> List[Dict[str, Any]]:
        """
        Dense training windows for encoder-based accident anticipation, with
        redundancy reduction:
        - larger train stride
        - negative subsampling
        - ranking-pair subsampling
        - per-video sample cap

        Clips always contain exactly S frames, sampled according to per-video FPS
        so that they span roughly clip_duration_sec seconds.
        """
        samples = []
        S = self.cfg.snippet_len
        stride = getattr(self.cfg, "train_stride", self.cfg.stride)

        rng = random.Random(self.cfg.seed)

        risk_target_fn = get_risk_target_fn(self.cfg)

        risk_values = []
        risk_values_future = []

        all_ttas = []

        for rec in self.records:
            if self.cfg.subset.upper() == "CAP":
                _, video_folder = self._cap_key_from_record(rec)
            else:
                video_folder = str(rec["video_name"])

            # if not rec.get("type") < 19:  # NOTE: skip videos!
            #     continue

            image_dir = self._resolve_video_dir(rec)
            frames = self._list_frames(image_dir)

            current_fps = self._get_fps(video_folder, dataset_type=self.cfg.subset.upper())

            if current_fps != 10: # just to debug. has no effect.
                pass

            horizon_frames = int(round(self.cfg.anticipation_horizon_sec * current_fps))

            t_ai = int(rec["t_ai"])
            t_co = int(rec["t_co"])
            is_accident = int(t_co >= 0)

            sample_stride = max(1, int(round(current_fps / self.cfg.base_fps)))
            first_valid_end_idx = (S - 1) * sample_stride

            all_ttas.append((t_co - t_ai) / current_fps)

            video_samples = []

            endpoint_stride = stride * sample_stride

            for end_idx in range(first_valid_end_idx, len(frames), endpoint_stride):
                end_frame_1b = end_idx + 1

                # usually skip post-collision windows for accident videos
                if is_accident and (not self.cfg.include_post_collision) and end_frame_1b > t_co:
                    continue

                frame_paths = self._sample_frame_paths_for_duration(
                    frames=frames,
                    end_idx=end_idx,
                    video_fps=current_fps,
                )
                if frame_paths is None:
                    continue

                # binary target: accident within next horizon seconds
                if is_accident and end_frame_1b <= t_co:
                    ttc_frames = t_co - end_frame_1b
                    binary_target = 1.0 if ttc_frames <= horizon_frames else 0.0

                    if not self.cfg.full_video_progress_risk:
                        # NOTE: Default Version
                        ttc_frames_10fps = ttc_frames * self.cfg.base_fps / current_fps
                        risk_target = risk_target_fn(ttc_frames_10fps)
                    else:
                        risk_target = full_video_progress_risk_fn(
                            current_frame_idx_1based=end_frame_1b,
                            t_co=t_co,
                            first_valid_frame_1based=first_valid_end_idx + 1,
                            mode=self.cfg.custom_risk_mode,
                            alpha=self.cfg.progress_alpha,
                        )
                
                    valid_progress = 1.0
                else:
                    ttc_frames = -1
                    binary_target = 0.0
                    risk_target = 0.0
                    valid_progress = 0.0

                # negative subsampling: keep all positives, only some negatives
                if binary_target == 0.0:
                    if rng.random() > getattr(self.cfg, "neg_keep_prob", 1.0):
                        continue
            
                risk_values.append(risk_target)

                # build a later clip from same video for preference/ranking
                future_frame_paths = None
                pref_valid = 0.0
                future_risk_target = 0.0

                # Randomize the gap for ranking pairs to increase diversity, instead of fixed gap.
                pair_gap_frames = random.randint(int(round(self.cfg.pair_gap_sec_min * current_fps)), int(round(self.cfg.pair_gap_sec_max * current_fps)))

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

                        if not self.cfg.full_video_progress_risk:
                            # NOTE: default version
                            future_ttc_frames_10fps = future_ttc_frames * self.cfg.base_fps / current_fps
                            future_risk_target = risk_target_fn(future_ttc_frames_10fps)
                        else: 
                            future_risk_target = full_video_progress_risk_fn(
                                current_frame_idx_1based=future_end_frame_1b,
                                t_co=t_co,
                                first_valid_frame_1based=first_valid_end_idx + 1,
                                mode=self.cfg.custom_risk_mode,
                                alpha=self.cfg.progress_alpha,
                            )

                        if future_risk_target > risk_target:
                            # subsample ranking pairs too
                            if rng.random() <= getattr(self.cfg, "pref_keep_prob", 1.0):
                                pref_valid = 1.0
                                risk_values_future.append(future_risk_target)
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

            # cap total samples per video to reduce domination by long videos
            num_capped = 0
            max_per_video = getattr(self.cfg, "max_samples_per_video", 50)
            if max_per_video is not None and len(video_samples) > max_per_video:
                num_capped += 1
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
        print("Videos hitting 50-sample cap:", num_capped)

        xx = []
        for ss in samples:
            xx.append(ss['risk_target'])
        
        json.dump({"risk_scores": xx}, open(f'./{self.cfg.custom_risk_mode}_scores_{self.cfg.subset}.json', 'w'))

        return samples

    def _build_anticipation_train_samples_fps_fix_random_pos_neg_pairs(self) -> List[Dict[str, Any]]:
        """
        Dense training windows for encoder-based accident anticipation, with
        random same-video negative-positive pairing for preference loss.

        Difference from the original function:
        - build all candidate clips first
        - then create preference pairs by randomly matching a negative clip
        with a positive clip from the SAME video
        - "future_frame_paths" is the paired positive clip, not necessarily later in time

        This keeps the current preference-loss API unchanged:
        future_risk_target > risk_target
        """
        samples = []
        S = self.cfg.snippet_len
        stride = getattr(self.cfg, "train_stride", self.cfg.stride)

        rng = random.Random(self.cfg.seed)
        risk_target_fn = get_risk_target_fn(self.cfg)

        risk_values = []
        risk_values_future = []

        for rec in self.records:
            if self.cfg.subset.upper() == "CAP":
                _, video_folder = self._cap_key_from_record(rec)
            else:
                video_folder = str(rec["video_name"])

            # if not rec.get("type") < 19:  # NOTE: keep your original skip logic
            #     continue

            image_dir = self._resolve_video_dir(rec)
            frames = self._list_frames(image_dir)
            current_fps = self._get_fps(video_folder, dataset_type=self.cfg.subset.upper())

            horizon_frames = int(round(self.cfg.anticipation_horizon_sec * current_fps))
            t_co = int(rec["t_co"])
            is_accident = int(t_co >= 0)

            sample_stride = max(1, int(round(current_fps / self.cfg.base_fps)))
            first_valid_end_idx = (S - 1) * sample_stride

            video_samples = []

            endpoint_stride = stride * sample_stride

            # --------------------------------------------------
            # 1) Build all candidate clips for this video
            # --------------------------------------------------
            for end_idx in range(first_valid_end_idx, len(frames), endpoint_stride):
                end_frame_1b = end_idx + 1

                # usually skip post-collision windows for accident videos
                if is_accident and (not self.cfg.include_post_collision) and end_frame_1b > t_co:
                    continue

                frame_paths = self._sample_frame_paths_for_duration(
                    frames=frames,
                    end_idx=end_idx,
                    video_fps=current_fps,
                )
                if frame_paths is None:
                    continue

                # binary target: accident within next horizon seconds
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
                    valid_progress = 0.0

                # keep all positives, subsample negatives
                if binary_target == 0.0:
                    if rng.random() > getattr(self.cfg, "neg_keep_prob", 1.0):
                        continue

                risk_values.append(risk_target)

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

            # --------------------------------------------------
            # 2) Random same-video neg-pos pairing
            # --------------------------------------------------
            pos_pool = [x for x in video_samples if x["binary_target"] > 0.0]
            neg_pool = [x for x in video_samples if x["binary_target"] == 0.0]

            if len(pos_pool) > 0 and len(neg_pool) > 0:
                # only negatives become anchors for ranking pairs
                for sample in neg_pool:
                    if rng.random() > getattr(self.cfg, "pref_keep_prob", 1.0):
                        continue

                    pos_partner = rng.choice(pos_pool)

                    # enforce low-risk -> high-risk ordering
                    sample["future_frame_paths"] = pos_partner["frame_paths"]
                    sample["future_risk_target"] = pos_partner["risk_target"]
                    sample["pref_valid"] = 1.0

                    risk_values_future.append(pos_partner["risk_target"])

            # --------------------------------------------------
            # 3) Cap total samples per video if requested
            # --------------------------------------------------
            num_capped = 0
            max_per_video = getattr(self.cfg, "max_samples_per_video", 50)
            if max_per_video is not None and len(video_samples) > max_per_video:
                num_capped += 1
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
        print("Videos hitting 50-sample cap:", num_capped)

        return samples


    def _filter_existing_cap_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered = []
        missing = []

        for r in records:
            try:
                key = self._cap_key_from_record(r)
            except Exception:
                continue

            if self.cap_video_index is not None and key in self.cap_video_index:
                
                image_dir = self.cap_video_index[key]
                frames = self._list_frames(image_dir)
                total = len(frames)
                if total == r.get("total_frames", 0):
                    filtered.append(r)
                else:
                    print(f"[Warning] CAP record {key} has total_frames={r.get('total_frames')} in metadata but found {total} frames in folder. Skipping this record for consistency. If you recently added this video, please ensure the metadata is updated with the correct total_frames count.", flush=True)
            else:
                missing.append((r.get("video_name"), r.get("type"), key))

        print(f"[CAP] kept {len(filtered)} records with local folders, dropped {len(missing)} missing records")
        if missing:
            print("[CAP] first 10 missing examples:", missing[:10])

        return filtered

    def _filter_existing_dada_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered = []
        missing = []

        for r in records:
            try:
                key = self._dada_key_from_record(r)
            except Exception:
                missing.append((r.get("video_name"), r.get("id")))
                continue

            if self.dada_video_index is not None and key in self.dada_video_index:
                filtered.append(r)
            else:
                missing.append((r.get("video_name"), r.get("id"), key))

        print(f"[DADA] kept {len(filtered)} records with local folders, dropped {len(missing)} missing records")
        if missing:
            print("[DADA] first 10 missing examples:", missing[:10])

        return filtered

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

    def _load_metadata(self, path: Path) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        records = []
        for key, value in raw.items():
            item = dict(value)
            item["video_hashcode"] = key

            # normalize numeric fields
            for k in ["id", "type", "weather", "light", "scenes", "linear",
                      "abnormal_start_frame", "abnormal_end_frame", "accident_frame",
                      "total_frames", "t_ai", "t_co", "t_ae", "accident occurred"]:
                if k in item and item[k] not in [None, ""]:
                    try:
                        item[k] = int(item[k])
                    except Exception:
                        pass
            records.append(item)
        return records

    def _select_subset(self, records: List[Dict[str, Any]], subset: str) -> List[Dict[str, Any]]:
        subset = subset.upper()
        out = []

        if subset == "CAP":
            for r in records:
                try:
                    cap_category, video_folder = self._parse_cap_record(r)
                except Exception:
                    continue

                in_test = (
                    video_folder in cap_test
                    and int(cap_test[video_folder]) == int(cap_category)
                )

                if self.cfg.split_name == "test":
                    if in_test:
                        out.append(r)
                elif self.cfg.split_name == "train":
                    if not in_test:
                        out.append(r)
                else:
                    raise ValueError(f"Unsupported split_name for CAP: {self.cfg.split_name}")

        elif subset == "DADA":
            for r in records:
                # video_name = str(r.get("video_name", ""))
                # category_id = int(r.get("id", -1))

                category_id, video_folder = self._dada_key_from_record(r)

                in_test = (str(f"{category_id}_{video_folder}") in dada_test)

                if self.cfg.split_name == "test":
                    if in_test:
                        out.append(r)
                elif self.cfg.split_name == "train":
                    if not in_test:
                        out.append(r)
                else:
                    raise ValueError(f"Unsupported split_name for DADA: {self.cfg.split_name}")

        else:
            raise ValueError("subset must be 'CAP' or 'DADA'")

        return out

    def _apply_split(
        self,
        records: List[Dict[str, Any]],
        split_json: Optional[str],
        split_name: str,
        seed: int,
    ) -> List[Dict[str, Any]]:
        """
        If split_json is provided, it should be a dict:
        {
          "train": ["hash1", "hash2", ...],
          "val":   [...],
          "test":  [...]
        }

        If absent, we do a deterministic 80/20 split by video hash.
        """
        if split_json is not None:
            with open(split_json, "r", encoding="utf-8") as f:
                split = json.load(f)
            allowed = set(split[split_name])
            return [r for r in records if r["video_hashcode"] in allowed]

        # fallback deterministic split
        rng = random.Random(seed)
        keys = sorted([r["video_hashcode"] for r in records])
        rng.shuffle(keys)
        n_test = int(0.2 * len(keys))
        test_keys = set(keys[:n_test])
        train_keys = set(keys[n_test:])

        if split_name == "test":
            return [r for r in records if r["video_hashcode"] in test_keys]
        elif split_name in ("train", "val"):
            return [r for r in records if r["video_hashcode"] in train_keys]
        else:
            raise ValueError(f"Unknown split_name: {split_name}")

    def _parse_cap_record(self, rec: Dict[str, Any]):
        """
        CAP folder mapping:
        category folder <- rec["type"]
        video folder    <- numeric suffix from rec["video_name"], zero-padded to 6 digits

        Examples:
        video_name="1_1",   type=10 -> ("10", "000001")
        video_name="11_7495", type=11 -> ("11", "007495")
        """
        video_name = str(rec["video_name"]).strip()
        parts = video_name.split("_")
        if len(parts) != 2:
            raise ValueError(f"Unexpected CAP video_name format: {video_name}")

        cap_category = str(int(rec["type"]))
        video_folder = f"{int(parts[1]):06d}"
        return cap_category, video_folder

    def _build_cap_video_index(self):
        """
        Build lookup:
        (cap_category, folder_name_6d) -> images_dir

        Supports:
        root/CAP-DATA_chunks/<chunk>/CAP-DATA/<chunk>/<cap_category>/<video_folder>/images
        """
        base = self.root / "CAP-DATA_chunks"
        if not base.exists():
            return {}

        index = {}
        for chunk_dir in sorted(base.iterdir()):
            if not chunk_dir.is_dir():
                continue

            chunk_name = chunk_dir.name
            cap_root = chunk_dir / "CAP-DATA" / chunk_name
            if not cap_root.exists():
                continue

            for category_dir in sorted(cap_root.iterdir()):
                if not category_dir.is_dir():
                    continue
                cap_category = str(category_dir.name)

                for video_dir in sorted(category_dir.iterdir()):
                    if not video_dir.is_dir():
                        continue

                    images_dir = video_dir / "images"
                    if images_dir.exists():
                        index[(cap_category, str(video_dir.name))] = images_dir
                    else:
                        frame_files = list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png"))
                        if frame_files:
                            index[(cap_category, str(video_dir.name))] = video_dir

        return index

    def _build_dada_video_index(self):
        """
        Supports:
        root/DADA-DATA/<category>/<video>/images
        root/DADA-DATA/<category>/<video>
        root/DADA-2000_chunks/Origin/DADA2000/DADA2000/<category>/<video>/images
        root/DADA-2000_chunks/Origin/DADA2000/DADA2000/<category>/<video>
        """
        candidate_roots = [
            self.root / "DADA-DATA",
            self.root / "DADA-2000_chunks" / "Origin" / "DADA2000" / "DADA2000",
        ]

        index = {}

        for base in candidate_roots:
            if not base.exists():
                continue

            for category_dir in sorted(base.iterdir()):
                if not category_dir.is_dir():
                    continue
                category_id = str(category_dir.name)

                for video_dir in sorted(category_dir.iterdir()):
                    if not video_dir.is_dir():
                        continue

                    images_dir = video_dir / "images"
                    if images_dir.exists():
                        index[(category_id, str(video_dir.name))] = images_dir
                    else:
                        frame_files = list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png"))
                        if frame_files:
                            index[(category_id, str(video_dir.name))] = video_dir

        return index
  
    def _resolve_video_dir(self, rec: Dict[str, Any]) -> Path:
        subset = self.cfg.subset.upper()

        if subset == "CAP":
            key = self._cap_key_from_record(rec)

            if self.cap_video_index is None or key not in self.cap_video_index:
                available = [k for k in self.cap_video_index.keys() if k[0] == key[0]][:20]
                raise FileNotFoundError(
                    f"Could not find CAP images for video_name={rec['video_name']}, "
                    f"type={rec['type']}. Expected key={key}. "
                    f"Example available keys in category {key[0]}: {available}"
                )
            return self.cap_video_index[key]

        elif subset == "DADA":
            key = self._dada_key_from_record(rec)

            if self.dada_video_index is not None and key in self.dada_video_index:
                return self.dada_video_index[key]

            raise FileNotFoundError(
                f"Could not find DADA images for video_name={rec['video_name']}, "
                f"video_hashcode={rec['video_hashcode']}, expected key={key}"
            )
        else:
            raise ValueError(f"Unknown subset: {self.cfg.subset}")

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
        video_fps: int,
    ) -> Optional[List[Path]]:
        """
        Build a snippet of exactly S frames covering roughly clip_duration_sec,
        by sampling with a stride based on the video's FPS.

        Examples for S=5, clip_duration_sec=0.5:
        fps=10 -> stride=1  -> [t-4, t-3, t-2, t-1, t]
        fps=20 -> stride=2  -> [t-8, t-6, t-4, t-2, t]
        fps=30 -> stride=3  -> [t-12, t-9, t-6, t-3, t]
        """
        S = self.cfg.snippet_len

        # target temporal spacing between sampled frames
        sample_stride = max(1, int(round(video_fps / self.cfg.base_fps)))

        start_idx = end_idx - (S - 1) * sample_stride
        if start_idx < 0 or end_idx >= len(frames):
            return None

        idxs = [start_idx + i * sample_stride for i in range(S)]
        if idxs[-1] != end_idx:
            idxs[-1] = end_idx  # keep exact alignment at the end frame

        return [frames[i] for i in idxs]
    

    def _tta_test(self) -> List[Dict[str, Any]]:
        """
        Build dense clip-level samples from anomaly start to accident.

        For each accident video:
        - sample every valid clip endpoint from t_ai to t_co (inclusive)
        - label = 1 if accident is within anticipation_horizon_sec
        - label = 0 otherwise

        For non-accident videos:
        - no samples are generated here, because this mode is intended for
            anomaly-to-collision evaluation.

        Uses per-video FPS to sample exactly S frames over ~clip_duration_sec.
        """
        samples = []
        S = self.cfg.snippet_len

        all_ttas = []

        for rec in self.records:
            image_dir = self._resolve_video_dir(rec)
            frames = self._list_frames(image_dir)

            t_ai = int(rec["t_ai"])
            t_co = int(rec["t_co"])
            t_ae = int(rec["t_ae"])

            # this mode is only meaningful for accident videos
            if t_co < 0:
                continue

            if self.cfg.subset.upper() == "CAP":
                _, video_folder = self._cap_key_from_record(rec)
            else:
                video_folder = str(rec["video_name"])

            video_fps = self._get_fps(video_folder, dataset_type=self.cfg.subset.upper())
            horizon_frames = int(round(self.cfg.anticipation_horizon_sec * video_fps))

            sample_stride = max(1, int(round(video_fps / self.cfg.base_fps)))

            all_ttas.append((t_co - t_ai) / video_fps)

            # First valid 1-based endpoint for a clip of S sampled frames
            first_valid_end_1b = (S - 1) * sample_stride + 1

            if video_fps != 10:
                print(
                    f"Video {video_folder} has FPS={video_fps}, "
                    f"sampling clips with stride={sample_stride} "
                    f"to cover ~{self.clip_duration_sec} seconds per clip",
                    flush=True,
                )

            # start from anomaly start, end at accident
            start_end_1b = first_valid_end_1b
            end_end_1b = min(t_co, len(frames))

            for end_idx_1b in range(start_end_1b, end_end_1b + 1, 1):
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
                    "horizon_sec": self.cfg.anticipation_horizon_sec if label == 1 else None,
                    "video_folder": video_folder,

                    # extra metadata for evaluation / analysis
                    "current_frame_idx_1based": end_idx_1b,
                    "ttc_frames": ttc_frames,
                    "ttc_sec": ttc_sec,
                    "fps": video_fps,
                    "t_ai": t_ai,
                    "t_co": t_co,
                    "t_ae": t_ae,
                    "is_after_anomaly": int(end_idx_1b >= t_ai),
                    "is_before_collision": int(end_idx_1b <= t_co),
                })

        return samples

    def _build_binary_clip_samples_fps_fix_TOP_like_sliding_window(self) -> List[Dict[str, Any]]:
        """
        Build dense clip-level samples in a 10-FPS-equivalent sliding-window manner.

        For 10 FPS videos:
        - sample every frame endpoint.

        For 20 FPS videos:
        - sample every 2nd frame endpoint.

        For 30 FPS videos:
        - sample every 3rd frame endpoint.

        This keeps evaluation density equivalent to 10 FPS across videos.
        """
        samples = []
        S = self.cfg.snippet_len

        all_ttas = []

        for rec in self.records:
            image_dir = self._resolve_video_dir(rec)
            frames = self._list_frames(image_dir)

            t_ai = int(rec["t_ai"])
            t_co = int(rec["t_co"])
            t_ae = int(rec.get("t_ae", -1))

            if self.cfg.subset.upper() == "CAP":
                _, video_folder = self._cap_key_from_record(rec)
            else:
                video_folder = str(rec["video_name"])

            video_fps = self._get_fps(video_folder, dataset_type=self.cfg.subset.upper())
            horizon_frames = int(round(self.cfg.anticipation_horizon_sec * video_fps))

            # 10-FPS-equivalent endpoint stride
            sample_stride = max(1, int(round(video_fps / self.cfg.base_fps)))

            all_ttas.append((t_co - t_ai) / video_fps)

            # First valid 1-based endpoint for a clip of S sampled frames
            first_valid_end_1b = (S - 1) * sample_stride + 1

            if video_fps != self.cfg.base_fps:
                print(
                    f"Video {video_folder} has FPS={video_fps}, "
                    f"evaluating every {sample_stride} frame(s) "
                    f"to match {self.cfg.base_fps} FPS evaluation.",
                    flush=True,
                )

            if t_co >= 0:
                end_end_1b = min(t_co, len(frames))
            else:
                end_end_1b = len(frames)

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
                    label = 1 if ttc_frames <= horizon_frames else 0
                    horizon_sec = ttc_sec if label == 1 else None
                else:
                    ttc_frames = -1
                    ttc_sec = -1.0
                    label = 0
                    horizon_sec = None

                samples.append({
                    "video_hashcode": rec["video_hashcode"],
                    "record": rec,
                    "frame_paths": frame_paths,
                    "label": label,
                    "horizon_sec": horizon_sec,
                    "current_frame_idx_1based": end_idx_1b,
                    "video_folder": video_folder,
                    "ttc_frames": ttc_frames,
                    "ttc_sec": ttc_sec,
                    "fps": video_fps,
                    "t_ai": t_ai,
                    "t_co": t_co,
                    "t_ae": t_ae,
                    "is_after_anomaly": int(t_co >= 0 and end_idx_1b >= t_ai),
                    "is_before_collision": int(t_co >= 0 and end_idx_1b <= t_co),
                })

        return samples


    def _build_binary_clip_samples_fps_fix_TOP_like_sliding_window_orj(self) -> List[Dict[str, Any]]:
        """
        Build dense clip-level samples with stride 1.

        For accident videos:
        - sample every valid clip endpoint from the first valid endpoint to t_co
        - label = 1 if accident is within anticipation_horizon_sec
        - label = 0 otherwise

        For non-accident videos:
        - sample every valid clip endpoint over the whole video
        - all labels are 0

        Uses per-video FPS to sample exactly S frames over ~clip_duration_sec.
        """
        samples = []
        S = self.cfg.snippet_len

        for rec in self.records:
            image_dir = self._resolve_video_dir(rec)
            frames = self._list_frames(image_dir)

            t_ai = int(rec["t_ai"])
            t_co = int(rec["t_co"])
            t_ae = int(rec.get("t_ae", -1))

            if self.cfg.subset.upper() == "CAP":
                _, video_folder = self._cap_key_from_record(rec)
            else:
                video_folder = str(rec["video_name"])

            video_fps = self._get_fps(video_folder, dataset_type=self.cfg.subset.upper())
            horizon_frames = int(round(self.cfg.anticipation_horizon_sec * video_fps))

            sample_stride = max(1, int(round(video_fps / self.cfg.base_fps)))

            # First valid 1-based endpoint for a clip of S sampled frames
            first_valid_end_1b = (S - 1) * sample_stride + 1

            if video_fps != 10:
                print(
                    f"Video {video_folder} has FPS={video_fps}, "
                    f"sampling clips with stride={sample_stride} "
                    f"to cover ~{self.clip_duration_sec} seconds per clip",
                    flush=True,
                )

            # accident videos: evaluate until accident
            if t_co >= 0:
                end_end_1b = min(t_co, len(frames))
            else:
                # non-accident videos: use whole video
                end_end_1b = len(frames)

            for end_idx_1b in range(first_valid_end_1b, end_end_1b + 1, 1):
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
                    label = 1 if ttc_frames <= horizon_frames else 0
                    horizon_sec = self.cfg.anticipation_horizon_sec if label == 1 else None
                else:
                    ttc_frames = -1
                    ttc_sec = -1.0
                    label = 0
                    horizon_sec = None

                samples.append({
                    "video_hashcode": rec["video_hashcode"],
                    "record": rec,
                    "frame_paths": frame_paths,
                    "label": label,
                    "horizon_sec": ttc_sec,
                    "current_frame_idx_1based": end_idx_1b,
                    "video_folder": video_folder,
                    "ttc_frames": ttc_frames,
                    "ttc_sec": ttc_sec,
                    "fps": video_fps,
                    "t_ai": t_ai,
                    "t_co": t_co,
                    "t_ae": t_ae,
                    "is_after_anomaly": int(t_co >= 0 and end_idx_1b >= t_ai),
                    "is_before_collision": int(t_co >= 0 and end_idx_1b <= t_co),
                })

        return samples

    def _build_binary_clip_samples_fps_fix_TOP_like(self) -> List[Dict[str, Any]]:
        """
        Build clip-level positive and negative samples:

        Negatives:
        - stride = 1 over valid clip endpoints
        - any clip whose endpoint is more than anticipation_horizon_sec
            before the accident is a negative
        - this is independent of anomaly start t_ai

        Positives:
        - fixed horizons in self.horizons_sec before t_co

        Uses per-video FPS to sample exactly S frames over ~clip_duration_sec.
        """
        samples = []
        S = self.cfg.snippet_len

        for rec in self.records:
            image_dir = self._resolve_video_dir(rec)
            frames = self._list_frames(image_dir)

            t_co = int(rec["t_co"])

            if self.cfg.subset.upper() == "CAP":
                _, video_folder = self._cap_key_from_record(rec)
            else:
                video_folder = str(rec["video_name"])
            
            video_fps = self._get_fps(video_folder, dataset_type=self.cfg.subset.upper())

            sample_stride = max(1, int(round(video_fps / self.cfg.base_fps)))

            # First valid 1-based endpoint for a clip of S sampled frames
            first_valid_end_1b = (S - 1) * sample_stride + 1

            if video_fps != 10:
                print(
                    f"Video {video_folder} has FPS={video_fps}, "
                    f"sampling clips with stride={sample_stride} "
                    f"to cover ~{self.clip_duration_sec} seconds per clip",
                    flush=True,
                )

            # -------------------------
            # Negative clips
            # -------------------------
            if t_co >= 0:
                neg_end_max = t_co - int(round(self.cfg.anticipation_horizon_sec * video_fps)) - 1
            else:
                # for non-accident videos, all valid endpoints are negative
                neg_end_max = len(frames)

            if neg_end_max >= first_valid_end_1b:
                for end_idx_1b in range(first_valid_end_1b, neg_end_max + 1, 1):  # stride=1 for negatives
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
                        "video_folder": video_folder,
                    })

            # -------------------------
            # Positive clips
            # -------------------------
            if t_co < 0:
                continue

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
                    "video_folder": video_folder
                })

        return samples

    def _build_binary_clip_samples_fps_fix(self) -> List[Dict[str, Any]]:
        """
        Use during training!!
        Build clip-level positive and negative samples:
        negatives from [1, t_ai)
        positives from 0.5s segments ending at (t_co - horizon)

        Uses per-video FPS to sample exactly S frames over ~0.5 seconds.
        """
        samples = []
        S = self.cfg.snippet_len

        fps_vals = []

        horizon_skipped = []
        for rec in self.records:
            image_dir = self._resolve_video_dir(rec)
            frames = self._list_frames(image_dir)
            total = len(frames)

            t_ai = int(rec["t_ai"])
            t_co = int(rec["t_co"])

            # t_ai, t_co, t_ae = self._get_mapped_event_frames(rec, total)

            if self.cfg.subset.upper() == "CAP":
                _, video_folder = self._cap_key_from_record(rec)
            else:
                video_folder = str(rec["video_name"])

            video_fps = self._get_fps(video_folder, dataset_type=self.cfg.subset.upper())
            fps_vals.append(video_fps)

            if video_fps != 10:
                print(f"Video {video_folder} has FPS={video_fps}, sampling clips with stride={max(1, int(round(video_fps / self.cfg.base_fps)))} to cover ~{self.clip_duration_sec} seconds per clip", flush=True)

            # Number of original frames in 0.5 seconds for this video
            sample_stride = max(1, int(round(video_fps / self.cfg.base_fps)))
            clip_span_frames = max(S, int(round(self.clip_duration_sec * video_fps)))

            # First valid 1-based endpoint for a clip of S sampled frames
            first_valid_end_1b = (S - 1) * sample_stride + 1

            # Negative clips from pre-anomaly region
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
                    "video_folder": video_folder,
                    "frame_paths": frame_paths,
                })

            # Positive clips around horizons
            if t_co < 0:
                continue

            for h in self.horizons_sec:
                target_frame_1b = t_co - int(round(h * video_fps))
                end_idx = target_frame_1b - 1

                if target_frame_1b < first_valid_end_1b:
                    continue

                # make sure the clip endpoint is after anomaly appears
                # if end_idx + 1 < t_ai:
                #     horizon_skipped.append(h)
                #     continue

                frame_paths = self._sample_frame_paths_for_duration(
                    frames=frames,
                    end_idx=end_idx,
                    video_fps=video_fps,
                )
                if frame_paths is None:
                    horizon_skipped.append(h)
                    continue

                samples.append({
                    "video_hashcode": rec["video_hashcode"],
                    "record": rec,
                    "frame_paths": frame_paths,
                    "label": 1,
                    "horizon_sec": h,
                    "video_folder": video_folder,
                    "frame_paths": frame_paths,
                })

        return samples


    def _build_binary_clip_samples(self) -> List[Dict[str, Any]]:
        """
        Build clip-level positive and negative samples:
        negatives from [1, t_ai)
        positives from 0.5s segments ending at (t_co - horizon)
        """
        samples = []
        S = self.cfg.snippet_len
        clip_len = max(S, int(round(self.clip_duration_sec * self.cfg.fps)))

        for rec in self.records:
            image_dir = self._resolve_video_dir(rec)
            frames = self._list_frames(image_dir)
            total = len(frames)

            t_ai = int(rec["t_ai"])
            t_co = int(rec["t_co"])

            # negatives from pre-anomaly region
            neg_end_max = max(S, t_ai - 1)
            for end_idx_1b in range(S, neg_end_max + 1, clip_len):
                end_idx = end_idx_1b - 1
                start_idx = end_idx - (S - 1)
                if start_idx < 0 or end_idx >= total:
                    continue
                samples.append({
                    "video_hashcode": rec["video_hashcode"],
                    "record": rec,
                    "frame_paths": frames[start_idx:end_idx + 1],
                    "label": 0,
                    "horizon_sec": None,
                })

            # positives around horizons
            if t_co < 0:
                continue

            # positives around horizons
            for h in self.horizons_sec:
                target_frame_1b = t_co - int(round(h * self.cfg.fps))
                end_idx = target_frame_1b - 1
                start_idx = end_idx - (S - 1)
                if start_idx < 0 or end_idx >= total:
                    continue
                # make sure clip is after anomaly appears
                if end_idx + 1 < t_ai:
                    continue
                samples.append({
                    "video_hashcode": rec["video_hashcode"],
                    "record": rec,
                    "frame_paths": frames[start_idx:end_idx + 1],
                    "label": 1,
                    "horizon_sec": h,
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, p: Path):
        img = Image.open(p).convert("RGB")
        if self.cfg.transform is not None:
            return self.cfg.transform(img)
        return img

    def _load_clip(self, paths, augment):
        """Load one clip's frames.

        A clip-level transform (utils.ClipTransform) receives the whole list so a
        single crop box / flip is shared across frames. A plain per-image transform
        keeps the original frame-by-frame behaviour and ignores `augment`.
        """
        if getattr(self.cfg.transform, "clip_level", False):
            return self.cfg.transform(
                [Image.open(p).convert("RGB") for p in paths],
                augment=augment,
            )
        return [self._load_image(p) for p in paths]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        s = self.samples[index]

        if self.mode == "anticipation_train":
            # One augmentation draw for the current and future clip together, so the
            # preference loss compares two time windows and not two different crops.
            n_cur = len(s["frame_paths"])
            paths = list(s["frame_paths"])
            if s["future_frame_paths"] is not None:
                paths += list(s["future_frame_paths"])

            # augment=True only says "this is the training branch". Whether any
            # augmentation happens depends on cfg.transform being a ClipTransform,
            # i.e. on --use_augmentations; a plain Compose ignores it (_load_clip).
            loaded = self._load_clip(paths, augment=True)
            images = loaded[:n_cur]
            future_images = loaded[n_cur:] or None

            return {
                "frames": torch.stack(images, dim=0) if isinstance(images[0], torch.Tensor) else images,
                "future_frames": (
                    torch.stack(future_images, dim=0) if (future_images is not None and isinstance(future_images[0], torch.Tensor))
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
            "id": s["record"]["id"],
            "t_ai": s["record"]["t_ai"],
            "t_co": s["record"]["t_co"],
            "t_ae": s["record"]["t_ae"],
            "total_frames": s["record"]["total_frames"],
            "texts": s["record"].get("texts", ""),
            "causes": s["record"].get("causes", ""),
            "measures": s["record"].get("measures", ""),
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
            out["frame_paths"] = s["frame_paths"]
            out["video_folder"] = s["video_folder"]

        else:
            out["label"] = s["label"]
            out["horizon_sec"] = s["horizon_sec"]
            out["frame_paths"] = s["frame_paths"]
            out["video_folder"] = s["video_folder"]

        return out