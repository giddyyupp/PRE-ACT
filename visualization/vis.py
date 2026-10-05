import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import subprocess
import shutil

from dataloaders.helpers import get_fps_mmau
from visualization.vis_utils import (build_sparse_score_maps, 
                                     pad_or_hold_scores, 
                                     safe_float, 
                                     load_json, 
                                     ensure_dir, 
                                     find_cap_video_folder, 
                                     find_dada_video_folder,
                                     list_frame_paths,
                                     score_to_bgr,)

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def reencode_for_web(input_path: str, output_path: str):
    """
    Re-encode video to a browser-friendly MP4:
    H.264 + yuv420p + faststart, no audio.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        output_path,
    ]
    subprocess.run(cmd, check=True)



def draw_text_block(img, lines, x, y, line_h=28, scale=0.7, color=(255, 255, 255), thickness=2):
    """
    Draw multiple text lines with a black rectangle background.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    widths = []
    for line in lines:
        (w, h), _ = cv2.getTextSize(line, font, scale, thickness)
        widths.append(w)
    block_w = max(widths) + 20
    block_h = len(lines) * line_h + 10

    cv2.rectangle(img, (x, y), (x + block_w, y + block_h), (0, 0, 0), thickness=-1)
    for i, line in enumerate(lines):
        yy = y + 25 + i * line_h
        cv2.putText(img, line, (x + 10, yy), font, scale, color, thickness, cv2.LINE_AA)



# ------------------------------------------------------------
# Prediction parsing
# ------------------------------------------------------------

def group_predictions_by_video(clip_outputs: List[dict]) -> Dict[str, List[dict]]:
    grouped = defaultdict(list)
    for item in clip_outputs:
        vh = item["video_hashcode"]
        grouped[vh].append(item)
    return grouped



# ------------------------------------------------------------
# Timeline / plot drawing
# ------------------------------------------------------------

def draw_timeline_panel(
    panel: np.ndarray,
    score_values: List[float],
    current_frame_1b: int,
    t_ai: int,
    t_co: int,
    t_ae: Optional[int] = None,
    title: str = "score",
):
    """
    Draw bottom timeline panel with:
      - score curve
      - current frame marker
      - anomaly start (t_ai)
      - accident (t_co)
      - anomaly end (t_ae if given)
    """
    h, w = panel.shape[:2]
    margin_left = 60
    margin_right = 30
    margin_top = 25
    margin_bottom = 35

    plot_x0 = margin_left
    plot_x1 = w - margin_right
    plot_y0 = margin_top
    plot_y1 = h - margin_bottom

    # Background
    panel[:] = (25, 25, 25)

    # Axes
    cv2.line(panel, (plot_x0, plot_y1), (plot_x1, plot_y1), (200, 200, 200), 1)
    cv2.line(panel, (plot_x0, plot_y0), (plot_x0, plot_y1), (200, 200, 200), 1)

    # y labels
    for yy, val in [(plot_y1, "0.0"), ((plot_y0 + plot_y1)//2, "0.5"), (plot_y0, "1.0")]:
        cv2.putText(panel, val, (8, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.line(panel, (plot_x0, yy), (plot_x1, yy), (60, 60, 60), 1)

    n = len(score_values)
    if n < 2:
        return

    def x_of(frame_1b):
        frac = (frame_1b - 1) / max(1, n - 1)
        return int(plot_x0 + frac * (plot_x1 - plot_x0))

    def y_of(score):
        score = np.clip(score, 0.0, 1.0)
        return int(plot_y1 - score * (plot_y1 - plot_y0))

    # curve
    pts = []
    for i, s in enumerate(score_values, start=1):
        pts.append((x_of(i), y_of(s)))
    pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(panel, [pts], isClosed=False, color=(0, 255, 255), thickness=2)

    # t_ai
    if t_ai is not None and t_ai > 0 and t_ai <= n:
        x = x_of(t_ai)
        cv2.line(panel, (x, plot_y0), (x, plot_y1), (0, 255, 0), 2)
        cv2.putText(panel, "t_ai", (x - 18, plot_y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # t_co
    if t_co is not None and t_co > 0 and t_co <= n:
        x = x_of(t_co)
        cv2.line(panel, (x, plot_y0), (x, plot_y1), (0, 0, 255), 2)
        cv2.putText(panel, "t_co", (x - 18, plot_y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # t_ae
    if t_ae is not None and t_ae > 0 and t_ae <= n:
        x = x_of(t_ae)
        cv2.line(panel, (x, plot_y0), (x, plot_y1), (255, 180, 0), 1)
        cv2.putText(panel, "t_ae", (x - 18, plot_y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1, cv2.LINE_AA)

    # current frame
    if current_frame_1b is not None and 1 <= current_frame_1b <= n:
        x = x_of(current_frame_1b)
        cv2.line(panel, (x, plot_y0), (x, plot_y1), (255, 255, 255), 2)
        cv2.putText(panel, "current", (x - 28, plot_y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # x-axis labels
    cv2.putText(panel, "1", (plot_x0 - 5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(panel, f"{n}", (plot_x1 - 15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    cv2.putText(panel, title, (w // 2 - 30, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)


# ------------------------------------------------------------
# Frame overlay
# ------------------------------------------------------------

def draw_score_meter(frame: np.ndarray, score: float, x: int = 10, y: int = 10, w: int = 28, h: int = 160):
    """
    Draw a vertical colored score bar.
    """
    color = score_to_bgr(score)

    # outer box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    fill_h = int(h * float(np.clip(score, 0.0, 1.0)))
    cv2.rectangle(frame, (x + 4, y + h - fill_h - 4), (x + w - 4, y + h - 4), color, thickness=-1)

    cv2.putText(frame, "1.0", (x + w + 5, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, "0.0", (x + w + 5, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)


def overlay_on_frame(
    frame: np.ndarray,
    frame_idx_1b: int,
    fps: float,
    score: float,
    risk_score: float,
    # fused_score: float,
    t_ai: int,
    t_co: int,
    t_ae: Optional[int],
    total_frames: int,
    timeline_scores: List[float],
    heat_key: str = "fused_score",
) -> np.ndarray:
    """
    Create annotated frame with heat tint, text block, score bar, and bottom timeline panel.
    """
    frame = frame.copy()
    h, w = frame.shape[:2]

    # choose which score controls heat tint
    if heat_key == "score":
        heat_score = score
    elif heat_key == "risk_score":
        heat_score = risk_score
    # else:
    #     heat_score = fused_score

    # full-frame heat tint
    color = score_to_bgr(heat_score)
    overlay = np.full_like(frame, color, dtype=np.uint8)
    frame = cv2.addWeighted(overlay, 0.20, frame, 0.80, 0)

    # left score bar
    draw_score_meter(frame, heat_score, x=10, y=20, w=26, h=160)

    # info text block
    t_sec = (frame_idx_1b - 1) / fps if fps > 0 else 0.0
    lines = [
        f"Frame: {frame_idx_1b}/{total_frames}",
        f"Time: {t_sec:.2f} s   FPS: {fps}",
        f"score:      {score:.3f}",
        # f"risk_score: {risk_score:.3f}",
        # f"fused:      {fused_score:.3f}",
        f"t_ai: {t_ai}   t_co: {t_co}" + (f"   t_ae: {t_ae}" if t_ae is not None else ""),
    ]
    draw_text_block(frame, lines, x=55, y=15, line_h=28, scale=0.65)

    # current event status
    status = []
    if frame_idx_1b < t_ai:
        status.append("pre-anomaly")
    elif frame_idx_1b < t_co:
        status.append("anomalous")
    elif frame_idx_1b == t_co:
        status.append("collision")
    elif t_ae is not None and frame_idx_1b <= t_ae:
        status.append("post-collision / abnormal end")
    else:
        status.append("post-collision")

    cv2.putText(frame, " | ".join(status), (55, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # bottom timeline panel
    panel_h = 180
    panel = np.zeros((panel_h, w, 3), dtype=np.uint8)
    draw_timeline_panel(
        panel,
        score_values=timeline_scores,
        current_frame_1b=frame_idx_1b,
        t_ai=t_ai,
        t_co=t_co,
        t_ae=t_ae,
        title=f"Timeline ({heat_key})"
    )

    out = np.vstack([frame, panel])
    return out


# ------------------------------------------------------------
# Main generation per video
# ------------------------------------------------------------

def make_visualization_video_for_video(
    video_hashcode: str,
    anno_dict: Dict[str, dict],
    preds_for_video: List[dict],
    cap_root: str,
    out_dir: str,
    subset: str,
    heat_key: str = "score",
    assign_to: str = "end",
):
    """
    Generate annotated video for one CAP video.
    """
    rec = anno_dict[video_hashcode]

    video_name = rec["video_name"]
    t_ai = int(rec["t_ai"])
    t_co = int(rec["t_co"])
    t_ae = int(rec.get("t_ae", -1))
    # fps = int(rec.get("fps", 10))

    if subset == "cap":
        video_folder = f"{int(video_name.split('_')[1]):06d}" 
        fps = get_fps_mmau(video_folder, dataset_type="CAP")
    elif subset == "dada":
        fps = get_fps_mmau(video_name, dataset_type="DADA")
    else:
        fps = 30

    if subset.lower() == 'dada':
        video_folder = find_dada_video_folder(cap_root, video_name)
    elif subset.lower() == 'cap':
        video_folder = find_cap_video_folder(cap_root, video_name)
    frame_paths = list_frame_paths(video_folder)
    total_frames = len(frame_paths)

    sparse_maps = build_sparse_score_maps(preds_for_video, fps=fps, assign_to=assign_to)

    score_dense = pad_or_hold_scores(sparse_maps["score"], total_frames)
    risk_dense = pad_or_hold_scores(sparse_maps["risk_score"], total_frames)
    fused_dense = pad_or_hold_scores(sparse_maps["fused_score"], total_frames)

    if heat_key == "score":
        timeline_scores = score_dense
    elif heat_key == "risk_score":
        timeline_scores = risk_dense
    else:
        timeline_scores = fused_dense

    # first frame size
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"Could not read frame: {frame_paths[0]}")
    frame_h, frame_w = first.shape[:2]
    out_h = frame_h + 180
    out_w = frame_w

    ensure_dir(out_dir)

    raw_out_path = str(Path(out_dir) / f"{video_name}_{video_hashcode}_{heat_key}_raw.mp4")
    out_path = str(Path(out_dir) / f"{video_name}_{video_hashcode}_{heat_key}.mp4")

    if os.path.exists(out_path):
        print(f"[INFO] Skipping {video_hashcode}: output already exists")
        return  

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_out_path, fourcc, fps, (out_w, out_h))

    print(f"[INFO] Writing raw video: {raw_out_path}")

    end_frame_1b = min(total_frames, int(round(t_co + 1 * fps)))
    trimmed_total_frames = end_frame_1b

    for i, frame_path in enumerate(frame_paths[:end_frame_1b], start=1):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        annotated = overlay_on_frame(
            frame=frame,
            frame_idx_1b=i,
            fps=fps,
            score=score_dense[i - 1],
            risk_score=risk_dense[i - 1],
            # fused_score=fused_dense[i - 1],
            t_ai=t_ai,
            t_co=t_co,
            t_ae=t_ae if t_ae > 0 else None,
            total_frames=trimmed_total_frames,
            timeline_scores=timeline_scores[:trimmed_total_frames],
            heat_key=heat_key,
        )

        writer.write(annotated)

    writer.release()

    print(f"[INFO] Re-encoding for web: {out_path}")
    reencode_for_web(raw_out_path, out_path)

    # remove temporary raw file if you want
    if os.path.exists(raw_out_path):
        os.remove(raw_out_path)

    print(f"[DONE] {out_path}")


# ------------------------------------------------------------
# Generate for all videos
# ------------------------------------------------------------

def make_videos_for_all(
    pred_json_path: str,
    anno_json_path: str,
    cap_root: str,
    out_dir: str,
    subset: str,
    heat_key: str = "score",
    assign_to: str = "end",
    limit: Optional[int] = None,
):
    """
    Generate annotated videos for all videos found in predictions.
    """
    clip_outputs = load_json(pred_json_path)
    anno_dict = load_json(anno_json_path)

    grouped = group_predictions_by_video(clip_outputs)
    video_hashcodes = sorted(grouped.keys())

    if limit is not None:
        video_hashcodes = video_hashcodes[:limit]

    print(f"[INFO] Number of videos in predictions: {len(video_hashcodes)}")

    CAP_selected_videos = [
        "232132ea",
        "ec56e498",
        "17dabd06",
        "12d85e02",
        "bd8bb5d9",
        "c8cc447b",
        "09e0304e",
        "37148f95",
        "913b6c65",
        "1efd8034",
        "41586ed3",
        "4ae0b749",
        "4cfdfe98",
        "8af676e7",
    ]


    for i, video_hashcode in enumerate(video_hashcodes, start=1):
        if video_hashcode not in anno_dict:
            print(f"[WARN] {video_hashcode} not in anno.json, skipping")
            continue
        
        # if video_hashcode not in CAP_selected_videos:
        #     print(f"[INFO] Skipping {video_hashcode} as it's not in selected videos")
        #     continue

        print(f"[{i}/{len(video_hashcodes)}] Processing {video_hashcode} ...")
        try:
            make_visualization_video_for_video(
                video_hashcode=video_hashcode,
                anno_dict=anno_dict,
                preds_for_video=grouped[video_hashcode],
                cap_root=cap_root,
                out_dir=out_dir,
                subset=subset,
                heat_key=heat_key,
                assign_to=assign_to,
            )
        except Exception as e:
            print(f"[ERROR] Failed for {video_hashcode}: {e}")


# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------

if __name__ == "__main__":
    anno_json_path = "../../data/MM_AU/video_metadata.json"
    cap_root = "../../data/MM_AU"
    subset = 'cap'

    res_folder = "./results_cap"
    results_folders = sorted(os.listdir(res_folder))

    for res in results_folders:

        if not res == "pre_act":
            continue
        
        pred_json_path = os.path.join(res_folder, res, "val_predictions.json")
        out_dir = f"./visuals/{subset.upper()}_videos_{res}"

        # if os.path.exists(out_dir):
        #     print(f"Skipping {res}: output folder already exists")
        #     continue

        if not os.path.exists(pred_json_path):
            print(f"Skipping {res}: file not found")
            continue

        os.makedirs(out_dir, exist_ok=True)

        make_videos_for_all(
            pred_json_path=pred_json_path,
            anno_json_path=anno_json_path,
            cap_root=cap_root,
            out_dir=out_dir,
            subset=subset,
            heat_key="score",   # "score", "risk_score", "fused_score"
            assign_to="end",        # use clip_names end frame
            limit=None,               # set 10 to test only first 10
        )