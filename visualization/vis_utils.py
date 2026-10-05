import os
import cv2
from typing import Dict, List, Optional, Tuple
from math import isfinite
import numpy as np
import json
from pathlib import Path



def score_to_bgr(score: float) -> Tuple[int, int, int]:
    """
    Map score in [0, 1] to BGR color using OpenCV JET colormap:
    low -> blue
    mid -> green/yellow
    high -> red
    """
    score = float(np.clip(score, 0.0, 1.0))
    val = np.uint8(score * 255)
    color = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
    return int(color[0]), int(color[1]), int(color[2])


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_float(x, default=0.0):
    if x is None:
        return default
    return float(x)


def pad_or_hold_scores(frame_score_map: Dict[int, float], num_frames: int) -> List[float]:
    """
    Expand sparse 1-based frame->score dictionary into a dense per-frame list [1..num_frames].
    Missing frames are filled by holding the last known value.
    Before the first known frame, the first known score is used.
    """
    scores = np.zeros(num_frames, dtype=np.float32)

    if not frame_score_map:
        return scores.tolist()

    known_frames = sorted(frame_score_map.keys())
    first_known = known_frames[0]
    last_score = frame_score_map[first_known]

    for i in range(1, num_frames + 1):
        if i in frame_score_map:
            last_score = frame_score_map[i]
        scores[i - 1] = last_score

    # fill before first known frame with first known score
    scores[:first_known - 1] = frame_score_map[first_known]
    return scores.tolist()


def build_sparse_score_maps(
    preds_for_video: List[dict],
    fps: float,
    assign_to: str = "end",
    snippet_len: int = 5,
    base_fps: float = 10.0,
) -> Dict[str, Dict[int, float]]:
    """
    Map predictions to 1-based native-video frames.

    clip_names contains the snippet's starting frame.
    By default, assign predictions to the snippet's ending frame,
    matching the evaluation code.

    fps: native video FPS.
    base_fps: sampling rate used within each snippet.
    """
    if assign_to not in {"start", "end"}:
        raise ValueError("assign_to must be 'start' or 'end'")

    fps, base_fps = float(fps), float(base_fps)
    if not all(isfinite(x) and x > 0 for x in (fps, base_fps)):
        raise ValueError("fps and base_fps must be finite and positive")

    if snippet_len < 1 or int(snippet_len) != snippet_len:
        raise ValueError("snippet_len must be a positive integer")

    stride = max(1, int(round(fps / base_fps)))
    end_offset = (int(snippet_len) - 1) * stride

    maps = {
        "score": {},
        "risk_score": {},
        "fused_score": {},
    }

    for item in preds_for_video:
        explicit_end = item.get("current_frame_idx_1based")

        if assign_to == "end" and explicit_end is not None:
            frame_idx = int(explicit_end)
        else:
            clip_name = item.get("clip_names")
            if clip_name is None:
                continue

            try:
                start_frame = int(clip_name.rsplit("_", 1)[-1])
            except (AttributeError, ValueError):
                continue

            frame_idx = start_frame + (
                end_offset if assign_to == "end" else 0
            )

        for key in maps:
            value = item.get(key)
            maps[key][frame_idx] = (
                0.0 if value is None else float(value)
            )

    return maps


# ------------------------------------------------------------
# CAP DADA path resolver
# ------------------------------------------------------------

def find_cap_video_folder(cap_root: str, video_name: str) -> Path:
    """
    Locate CAP video folder based on video_name like "10_27", "11_7495", etc.

    CAP directory examples:
      MM_AU/CAP-DATA_chunks/1-10/CAP-DATA/1-10/10/000027
      MM_AU/CAP-DATA_chunks/11/CAP-DATA/11/11/007495
      MM_AU/CAP-DATA_chunks/12-42/CAP-DATA/12-42/12/000123
      MM_AU/CAP-DATA_chunks/43/CAP-DATA/43/43/000001
      MM_AU/CAP-DATA_chunks/44-62/CAP-DATA/44-62/44/000001

    """
    cap_root = Path(cap_root)
    cat_str, vid_str = video_name.split("_")
    cat = int(cat_str)
    vid_folder = f"{int(vid_str):06d}"

    if 1 <= cat <= 10:
        chunk = "1-10"
    elif cat == 11:
        chunk = "11"
    elif 12 <= cat <= 42:
        chunk = "12-42"
    elif cat == 43:
        chunk = "43"
    elif 44 <= cat <= 62:
        chunk = "44-62"
    else:
        raise ValueError(f"Unsupported CAP category: {cat}")

    folder = cap_root / "CAP-DATA_chunks" / chunk / "CAP-DATA" / chunk / str(cat) / vid_folder / 'images'
    if not folder.exists():
        raise FileNotFoundError(f"Could not find folder for {video_name}: {folder}")
    return folder


def find_dada_video_folder(dada_root: str, video_name: str) -> Path:
    """
    Locate DADA video folder based on video_name like "1_001", "12_034", etc.

    Supports:
      root/DADA-DATA/<category>/<video>/images
      root/DADA-DATA/<category>/<video>
      root/DADA-2000_chunks/Origin/DADA2000/DADA2000/<category>/<video>/images
      root/DADA-2000_chunks/Origin/DADA2000/DADA2000/<category>/<video>
    """
    dada_root = Path(dada_root)

    cat_str, vid_str = video_name.split("_")
    cat = str(int(cat_str))

    vid_int = int(vid_str)
    video_candidates = [
        vid_str,
        str(vid_int),
        f"{vid_int:03d}",
        f"{vid_int:04d}",
        f"{vid_int:06d}",
    ]

    base_candidates = [
        dada_root / "DADA-DATA",
        dada_root / "DADA-2000_chunks" / "Origin" / "DADA2000" / "DADA2000",
    ]

    for base in base_candidates:
        for vid in video_candidates:
            for folder in [
                base / cat / vid / "images",
                base / cat / vid,
            ]:
                if folder.exists():
                    return folder

    raise FileNotFoundError(
        f"Could not find DADA folder for {video_name} under {dada_root}"
    )


def list_frame_paths(video_folder: Path) -> List[Path]:
    """
    List image frames from CAP folder. Assumes images named like 000001.jpg/png.
    """
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    frames = []
    for e in exts:
        frames.extend(video_folder.glob(e))
    frames = sorted(frames)
    if not frames:
        raise FileNotFoundError(f"No frames found in {video_folder}")
    return frames