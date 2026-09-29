import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, auc
)
from evaluation.metrics import (
    partial_auc_normalized,
    _bin_segment,
    compute_interval_metrics,
    pad_repeat_first,
    sklearn_auc,
)

from evaluation.eval_utils import (
    fuse_scores_residual,
    fuse_scores_adaptive,
    fuse_scores_boost,
    fuse_scores_max,
)

from evaluation.eval_utils import print_validation_metrics


def _safe_partial_auc(y_true, y_score, max_fpr=0.1):
    """
    Returns normalized partial AUC in [0, 1], same style as sklearn.
    """
    return roc_auc_score(y_true, y_score, max_fpr=max_fpr)


def _safe_full_auc(y_true, y_score):
    return roc_auc_score(y_true, y_score)


def save_anticipation_roc_plot(
    preds,
    labels,
    save_path,
    title="Anticipation ROC",
    max_fpr=0.1,
    negative_mode="last5",
):
    """
    Save a multi-horizon ROC plot similar to the figure you shared.

    Args:
        preds: list of 1D numpy arrays, one sequence per video
        labels: list of bool, True for positive sequence, False for negative sequence
        save_path: output PNG path
        title: plot title
        max_fpr: highlighted FPR region, usually 0.1
        negative_mode:
            - "last5": use max(pred[-5:]) for negative sequences
            - "allmax": use max(pred) for negative sequences
    """

    # -------------------------
    # positive horizon scores
    # -------------------------
    preds_0 = [np.max(pred[-5:]) for pred, label in zip(preds, labels) if label]
    preds_5 = [np.max(pred[-10:-5]) for pred, label in zip(preds, labels) if label and len(pred) >= 10]
    preds_10 = [np.max(pred[-15:-10]) for pred, label in zip(preds, labels) if label and len(pred) >= 15]
    preds_15 = [np.max(pred[-20:-15]) for pred, label in zip(preds, labels) if label and len(pred) >= 20]

    # -------------------------
    # negative scores
    # -------------------------
    neg_preds_raw = [pred for pred, label in zip(preds, labels) if not label]

    if negative_mode == "last5":
        preds_n = [np.max(pred[-5:]) for pred in neg_preds_raw if len(pred) >= 5]
    elif negative_mode == "allmax":
        preds_n = [np.max(pred) for pred in neg_preds_raw if len(pred) > 0]
    else:
        raise ValueError(f"Unknown negative_mode: {negative_mode}")

    curve_defs = [
        ("0.0s before accidents", preds_0),
        ("0.5s before accidents", preds_5),
        ("1.0s before accidents", preds_10),
        ("1.5s before accidents", preds_15),
    ]

    plt.figure(figsize=(7, 5.5))

    for label_name, pos_scores in curve_defs:
        if len(pos_scores) == 0 or len(preds_n) == 0:
            print(f"[WARN] Skipping ROC curve for {label_name}, insufficient samples.")
            continue

        y_true = np.array([1] * len(pos_scores) + [0] * len(preds_n), dtype=np.int32)
        y_score = np.array(pos_scores + preds_n, dtype=np.float32)

        fpr, tpr, _ = roc_curve(y_true, y_score)

        auc_full = _safe_full_auc(y_true, y_score)
        auc_p01 = _safe_partial_auc(y_true, y_score, max_fpr=max_fpr)

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{label_name} (AUC={auc_full:.3f}, pAUC@{max_fpr:.1f}={auc_p01:.3f})"
        )

    # highlighted max_fpr box
    plt.axvline(max_fpr, linestyle="--", linewidth=1.5, color="gray")
    plt.axhline(1.0, linestyle="--", linewidth=1.0, color="gray", alpha=0.6)

    # small dashed rectangle like the paper figure
    plt.plot([0, max_fpr], [1, 1], linestyle="--", linewidth=1.5, color="gray")
    plt.plot([max_fpr, max_fpr], [0, 1], linestyle="--", linewidth=1.5, color="gray")

    # annotation
    plt.annotate(
        f"$\\lambda$ = {max_fpr}",
        xy=(max_fpr * 0.6, 0.08),
        xytext=(max_fpr * 1.2, 0.18),
        arrowprops=dict(arrowstyle="->", lw=1.5),
        fontsize=11
    )

    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.xticks(np.arange(0.0, 1.01, 0.1))
    plt.yticks(np.arange(0.0, 1.01, 0.1))
    plt.xlabel("False Positive Rate (False Alarm Rate)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title(title)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved ROC plot to: {save_path}")



# ------------------------------------------------------------
# Our Proposed Separation Score metric
# ------------------------------------------------------------

def save_step_profile_csv(profile, save_path):
    """
    Save one normalized pre/post profile to CSV.

    profile should be the dict returned by compute_binned_step_profile(...)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    x = np.asarray(profile["x"])
    y = np.asarray(profile["y"])
    std = np.asarray(profile["std"])
    ideal = np.asarray(profile["ideal"])

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "std", "lower", "upper", "ideal"])

        for xi, yi, si, ii in zip(x, y, std, ideal):
            lower = max(0.0, yi - si)
            upper = min(1.0, yi + si)
            writer.writerow([xi, yi, si, lower, upper, ii])



def save_all_step_profiles(model_profiles, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for model_name, profile in model_profiles.items():
        safe_name = model_name.lower().replace(" ", "_")
        save_step_profile_csv(profile, os.path.join(out_dir, f"{safe_name}.csv"))


def make_ideal_step_edges(n_pre_bins=20, n_post_bins=20):
    x_edges_pre = np.linspace(-1.0, 0.0, n_pre_bins + 1)
    x_edges_post = np.linspace(0.0, 1.0, n_post_bins + 1)[1:]
    x_edges = np.concatenate([x_edges_pre, x_edges_post])

    y_vals = np.concatenate([
        np.zeros(n_pre_bins, dtype=np.float32),
        np.ones(n_post_bins + 1, dtype=np.float32),
    ])

    return x_edges, y_vals


def make_ideal_step_profile(n_pre_bins=20, n_post_bins=20):
    x_pre = np.linspace(-1.0, 0.0, n_pre_bins, endpoint=False)
    x_post = np.linspace(0.0, 1.0, n_post_bins, endpoint=True)
    x = np.concatenate([x_pre, x_post], axis=0)

    y = np.concatenate([
        np.zeros(n_pre_bins, dtype=np.float32),
        np.ones(n_post_bins, dtype=np.float32),
    ])

    std = np.zeros_like(y, dtype=np.float32)

    return {
        "x": x,
        "y": y,
        "std": std,
        "all_profiles": y[None, :],
        "ideal": y.copy(),
        "pre_area": 0.0,
        "post_area": 1.0,
    }

def compute_binned_step_profile(
    preds,
    abnormal_start_inds,
    accident_inds,
    labels=None,
    n_pre_bins=20,
    n_post_bins=20,
):
    """
    Build a normalized binned score profile across positive sequences.

    Inputs
    ------
    preds : list of 1D arrays
        Unpadded prediction sequences for each video.
    abnormal_start_inds : list[int]
        Index of anomaly start inside each unpadded prediction sequence.
    accident_inds : list[int]
        Index of accident inside each unpadded prediction sequence.
    labels : optional list[bool or int]
        If provided, only positive sequences are used.
    n_pre_bins : int
        Number of bins in pre-anomaly region.
    n_post_bins : int
        Number of bins in post-anomaly region.

    Returns
    -------
    result : dict
        {
            "x": normalized x-axis,
            "y": mean profile,
            "std": std profile,
            "all_profiles": stacked profiles,
            "ideal": ideal step profile,
            "pre_area": normalized pre area,
            "post_area": normalized post area,
        }
    """
    profiles = []

    for i, seq in enumerate(preds):
        if labels is not None and not bool(labels[i]):
            continue

        seq = np.asarray(seq, dtype=np.float32)
        ai_idx = int(abnormal_start_inds[i])
        co_idx = int(accident_inds[i])

        if len(seq) == 0:
            continue
        if ai_idx < 0 or co_idx < 0:
            continue
        if ai_idx >= len(seq):
            continue
        if co_idx >= len(seq):
            co_idx = len(seq) - 1
        if co_idx < ai_idx:
            continue

        pre_seg = seq[:ai_idx]
        post_seg = seq[ai_idx:co_idx + 1]

        # keep empty pre-seg as NaNs, but post should exist
        pre_bins = _bin_segment(pre_seg, n_pre_bins)
        post_bins = _bin_segment(post_seg, n_post_bins)

        profile = np.concatenate([pre_bins, post_bins], axis=0)
        profiles.append(profile)

    if len(profiles) == 0:
        raise ValueError("No valid positive sequences found for profile computation.")

    profiles = np.stack(profiles, axis=0)  # [N, n_pre_bins + n_post_bins]

    y = np.nanmean(profiles, axis=0)
    std = np.nanstd(profiles, axis=0)

    # normalized x-axis:
    # [-1, 0) for pre-anomaly
    # [0, 1] for post-anomaly
    x_pre = np.linspace(-1.0, 0.0, n_pre_bins, endpoint=False)
    x_post = np.linspace(0.0, 1.0, n_post_bins, endpoint=False) + (1.0 / n_post_bins) / 2.0
    x = np.concatenate([x_pre, x_post], axis=0)

    ideal = np.concatenate([
        np.zeros(n_pre_bins, dtype=np.float32),
        np.ones(n_post_bins, dtype=np.float32),
    ])

    # normalized interval areas on the averaged profile
    pre_y = y[:n_pre_bins]
    post_y = y[n_pre_bins:]

    def _normalized_area(seg):
        seg = np.asarray(seg, dtype=np.float32)
        valid = ~np.isnan(seg)
        seg = seg[valid]
        if len(seg) == 0:
            return np.nan
        if len(seg) == 1:
            return float(seg[0])
        area = np.trapezoid(seg, dx=1.0)
        return float(area / (len(seg) - 1))

    pre_area = _normalized_area(pre_y)
    post_area = _normalized_area(post_y)

    return {
        "x": x,
        "y": y,
        "std": std,
        "all_profiles": profiles,
        "ideal": ideal,
        "pre_area": pre_area,
        "post_area": post_area,
    }


def plot_binned_step_profiles(
    model_profiles,
    title="Pre/Post-Anomaly Step Profile",
    ylabel="Predicted score",
    save_path=None,
    show_std=True,
    use_step=True,
):
    """
    model_profiles: dict
        {
            "model_name": output_of_compute_binned_step_profile(...),
            ...
        }
    """
    plt.figure(figsize=(10, 6))

    # plot model curves
    for model_name, res in model_profiles.items():
        x = res["x"]
        y = res["y"]
        std = res["std"]

        label = (
            f"{model_name} "
            f"(preAUC={res['pre_area']:.3f}, postAUC={res['post_area']:.3f})"
        )

        if use_step:
            plt.step(x, y, where="mid", linewidth=2, label=label)
        else:
            plt.plot(x, y, linewidth=2, label=label)

        if show_std:
            lower = np.clip(y - std, 0.0, 1.0)
            upper = np.clip(y + std, 0.0, 1.0)
            plt.fill_between(x, lower, upper, alpha=0.15)

    # plot ideal curve once
    first_key = next(iter(model_profiles.keys()))
    x = model_profiles[first_key]["x"]
    ideal = model_profiles[first_key]["ideal"]
    plt.step(x, ideal, where="mid", linestyle="--", linewidth=2, label="Ideal step")

    plt.axvline(0.0, linestyle="--", color="gray", alpha=0.8)
    plt.text(0.01, 0.03, "Video Start", transform=plt.gca().transAxes)

    plt.ylim(0.0, 1.05)
    plt.xlim(-1.0, 1.0)
    plt.xlabel("Normalized time (pre-anomaly  →  post-anomaly)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()



def metric_calculate(preds, labels, abnormal_start_inds, accident_inds, fpr_max=0.1):
    """
    preds: list of 1D numpy arrays
    labels: list of bool
    abnormal_start_inds: list of int
    accident_inds: list of int
    """
    eval_results = {}

    # If any pred is 2D, max over last dim
    if len(preds) > 0 and len(preds[0].shape) > 1:
        preds = [np.max(pred, axis=-1) for pred in preds]

    # negatives
    preds_n = [pred for pred, label in zip(preds, labels) if not label]
    preds_n = np.concatenate(preds_n)
    preds_n = -np.sort(-preds_n)

    eval_results[f"threshold@{fpr_max:.2f}"] = preds_n[int(len(preds_n) * fpr_max)]

    # TTA
    ttas = [
        (j - i - np.argmax(pred[i:j + 1, None] >= preds_n, axis=0)) / 10
        * np.any(pred[i:j + 1, None] >= preds_n, axis=0)
        for pred, label, i, j in zip(preds, labels, abnormal_start_inds, accident_inds)
        if label
    ]
    ttas = np.array(ttas).mean(axis=0)

    eval_results["tta@0.01"] = ttas[int(len(ttas) * 0.01)]
    eval_results["tta@0.05"] = ttas[int(len(ttas) * 0.05)]
    eval_results["tta@0.1"] = ttas[int(len(ttas) * fpr_max)]
    eval_results["mtta@0.1"] = ttas[: int(len(ttas) * fpr_max)].mean()
    eval_results["tta@1"] = ttas[int(len(ttas) * 0.99)]

    # AUC / AP windows
    preds_0 = [np.max(pred[-5:]) for pred, label in zip(preds, labels) if label]
    preds_5 = [np.max(pred[-10:-5]) for pred, label in zip(preds, labels) if label]
    preds_10 = [np.max(pred[-15:-10]) for pred, label in zip(preds, labels) if label]
    preds_15 = [np.max(pred[-20:-15]) for pred, label in zip(preds, labels) if label]
    preds_n = [np.max(pred[-5:]) for pred, label in zip(preds, labels) if not label]

    # preds_n = [np.max(pred) for pred, label in zip(preds, labels) if not label]  # NOTE: Considers all negatives!!! not just supposedly hard ones!

    y = [1] * len(preds_0) + [0] * len(preds_n)

    partial_auc_0 = partial_auc_normalized(np.array(y), np.array(preds_0 + preds_n), fpr_max)
    partial_auc_5 = partial_auc_normalized(np.array(y), np.array(preds_5 + preds_n), fpr_max)
    partial_auc_10 = partial_auc_normalized(np.array(y), np.array(preds_10 + preds_n), fpr_max)
    partial_auc_15 = partial_auc_normalized(np.array(y), np.array(preds_15 + preds_n), fpr_max)

    eval_results["AUC@0.0s"] = partial_auc_0
    eval_results["AUC@0.5s"] = partial_auc_5
    eval_results["AUC@1.0s"] = partial_auc_10
    eval_results["AUC@1.5s"] = partial_auc_15

    eval_results["mAUC@0.1"] = (
        eval_results["AUC@0.5s"]
        + eval_results["AUC@1.0s"]
        + eval_results["AUC@1.5s"]
    ) / 3

    partial_auc_0_001 = partial_auc_normalized(np.array(y), np.array(preds_0 + preds_n), 0.01)
    partial_auc_5_001 = partial_auc_normalized(np.array(y), np.array(preds_5 + preds_n), 0.01)
    partial_auc_10_001 = partial_auc_normalized(np.array(y), np.array(preds_10 + preds_n), 0.01)
    partial_auc_15_001 = partial_auc_normalized(np.array(y), np.array(preds_15 + preds_n), 0.01)

    eval_results["AUC_0.01@0.0s"] = partial_auc_0_001
    eval_results["AUC_0.01@0.5s"] = partial_auc_5_001
    eval_results["AUC_0.01@1.0s"] = partial_auc_10_001
    eval_results["AUC_0.01@1.5s"] = partial_auc_15_001

    eval_results["mAUC@0.01"] = (
        eval_results["AUC_0.01@0.5s"]
        + eval_results["AUC_0.01@1.0s"]
        + eval_results["AUC_0.01@1.5s"]
    ) / 3


    eval_results["AUC_full@0.0s"], _, _ = sklearn_auc(np.array(y), np.array(preds_0 + preds_n), 1.0)
    eval_results["AUC_full@0.5s"], _, _ = sklearn_auc(np.array(y), np.array(preds_5 + preds_n), 1.0)
    eval_results["AUC_full@1.0s"], _, _ = sklearn_auc(np.array(y), np.array(preds_10 + preds_n), 1.0)
    eval_results["AUC_full@1.5s"], _, _ = sklearn_auc(np.array(y), np.array(preds_15 + preds_n), 1.0)

    eval_results["mAUC"] = (
        eval_results["AUC_full@0.5s"]
        + eval_results["AUC_full@1.0s"]
        + eval_results["AUC_full@1.5s"]
    ) / 3

    eval_results["AP@0.0s"] = average_precision_score(np.array(y), np.array(preds_0 + preds_n))
    eval_results["AP@0.5s"] = average_precision_score(np.array(y), np.array(preds_5 + preds_n))
    eval_results["AP@1.0s"] = average_precision_score(np.array(y), np.array(preds_10 + preds_n))
    eval_results["AP@1.5s"] = average_precision_score(np.array(y), np.array(preds_15 + preds_n))

    eval_results["mAP"] = (
        eval_results["AP@0.5s"]
        + eval_results["AP@1.0s"]
        + eval_results["AP@1.5s"]
    ) / 3

    eval_results["num_samples"] = len(preds)
    return eval_results


# ------------------------------------------------------------
# Convert your dense clip file -> 10 FPS sequences (0.5sec snippets)
# ------------------------------------------------------------

def build_inputs_from_dense_clip_file(
    clip_outputs,
    anno_dict,
    score_key="score",
    snippet_len=5,
    base_fps=10,
    match_neg_pos_numbers=False,
    convert_raw_logit=False,
    ):
    """
    Build inputs from dense clip outputs.

    Important:
    - Metrics assume prediction sequences are sampled at 10 FPS.
    - Therefore, for videos with fps > 10, we must subsample clips onto the
      10-FPS-equivalent grid before building the sequence.

    We construct:
      - one positive sequence per video: clips up to accident
      - one negative sequence per video: safe clips with TTA > 2.0s
    """
    by_video = {}
    non_ego_accidents = 0

    for row in clip_outputs:
        clip_name = row["clip_names"]
        # vid_id, start_str = clip_name.split("_")
        splits = clip_name.split("_")
        if len(splits) == 2:
             vid_id, start_str = splits[0], splits[1]
        else:
            vid_id, start_str = f"{splits[0]}_{splits[1]}", splits[2]

        start_frame = int(start_str)

        if vid_id not in anno_dict:
            continue

        info = anno_dict[vid_id]

        if not (1 <= int(info["accident_type"]) <= 18):
            # lets skip non-ego vehicle accidents. 
            non_ego_accidents += 1
            continue

        fps = float(info["fps"])
        try:
            abnormal_start_frame = int(info["abnormal_start_frame"])
        except:
            abnormal_start_frame = int(info["accident_frame"])
        accident_frame = int(info["accident_frame"])

        if fps != base_fps:
            pass

        sample_stride = max(1, int(round(fps / base_fps)))

        # Keep only clips aligned to the 10-FPS-equivalent grid
        if (start_frame - 1) % sample_stride != 0:
            continue

        end_frame = start_frame + (snippet_len - 1) * sample_stride
        tta = (accident_frame - end_frame) / fps

        if score_key == 'fused_score':
            score = fuse_scores_residual(float(row["score"]), float(row["risk_score"]))
        else:
            score = float(row[score_key])

        if convert_raw_logit:
            score = np.log(score / (1 - score))  # if the score is a probability, convert to logit

        item = {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "score": score,
            "tta": float(tta),
            "sample_stride": sample_stride,
            "abnormal_start_frame": abnormal_start_frame,
            "accident_frame": accident_frame,
        }

        by_video.setdefault(vid_id, []).append(item)
    
    print(f"Total videos with non-ego accidents skipped: {non_ego_accidents}")

    preds = []
    labels = []
    abnormal_start_inds = []
    abnormal_start_inds_no_pad = []
    accident_inds = []
    accident_inds_no_pad = []
    preds_no_pad = []

    dropped_pos = 0
    dropped_neg = 0

    for vid_id, clips in by_video.items():

        clips = sorted(clips, key=lambda x: x["start_frame"])

        if len(clips) == 0:
            continue

        abnormal_start_frame = clips[0]["abnormal_start_frame"]
        accident_frame = clips[0]["accident_frame"]

        # -------------------------
        # positive sequence
        # -------------------------
        pos_clips = [c for c in clips if c["end_frame"] <= accident_frame]

        # if len(pos_clips) < 20:
        #     dropped_pos += 1
        #     continue

        if len(pos_clips) >= 0:
            pos_pred = np.array([c["score"] for c in pos_clips], dtype=np.float32)
            pos_end_frames = np.array([c["end_frame"] for c in pos_clips], dtype=np.int32)

            ai_candidates = np.where(pos_end_frames >= abnormal_start_frame)[0]

            # Prefer the first clip that reaches/passes accident_frame.
            # If none exists, fall back to the last available pre-accident clip.
            co_candidates = np.where(pos_end_frames >= accident_frame)[0]

            if len(ai_candidates) > 0:
                ai_idx = int(ai_candidates[0])

                if len(co_candidates) > 0:
                    co_idx = int(co_candidates[0])
                else:
                    co_idx = len(pos_end_frames) - 1

                # sanity: accident index should not be before anomaly-start index
                if co_idx < ai_idx:
                    dropped_pos += 1
                else:
                    old_len = len(pos_pred)
                    preds_no_pad.append(pos_pred)

                    pos_pred = pad_repeat_first(pos_pred, 20)
                    pad_len = len(pos_pred) - old_len

                    preds.append(pos_pred)
                    labels.append(True)
                    abnormal_start_inds.append(ai_idx + pad_len)
                    accident_inds.append(co_idx + pad_len)
                    abnormal_start_inds_no_pad.append(ai_idx)
                    accident_inds_no_pad.append(co_idx)

            else:
                dropped_pos += 1
        else:
            dropped_pos += 1

        # -------------------------
        # negative sequence
        # -------------------------
        # neg_clips = [c for c in clips if c["tta"] > 2.0]

        # if len(neg_clips) > 0:
        #     neg_pred = np.array([c["score"] for c in neg_clips], dtype=np.float32)
        #     neg_pred = pad_repeat_first(neg_pred, 5)

        #     preds.append(neg_pred)
        #     labels.append(False)
        #     abnormal_start_inds.append(0)
        #     accident_inds.append(0)
        # else:
        #     dropped_neg += 1


        # -------------------------
        # negative sequence
        # -------------------------
        # Use only pre-anomaly clips: clips whose endpoint is before abnormal start
        neg_clips = [
            c for c in clips
            if c["end_frame"] < abnormal_start_frame and c["tta"] > 2.0
        ]

        if len(neg_clips) > 0: #(len(pos_clips) - len(neg_clips)) > 20:  # ensure we have enough clips to form a positive sequence for this video, otherwise skip the negative as well since it won't be comparable
            neg_pred = np.array([c["score"] for c in neg_clips], dtype=np.float32)
            preds_no_pad.append(neg_pred)

            neg_pred = pad_repeat_first(neg_pred, 5)
            preds.append(neg_pred)
            labels.append(False)
            abnormal_start_inds.append(0)
            accident_inds.append(0)
            abnormal_start_inds_no_pad.append(0)
            accident_inds_no_pad.append(0)
        else:
            dropped_neg += 1

    if match_neg_pos_numbers:
        fill_negs = dropped_neg - dropped_pos
        if fill_negs > 0:
            preds_n = [pred for pred, label in zip(preds_no_pad, labels) if not label and len(pred) >= 10]
            # random.shuffle(preds_n)    
            preds_n = sorted(preds_n, key=len, reverse=True)
            # n = len(preds_n)  
            # seq_step = max(1, n // fill_negs)

            # if random_indices is None:
            #     random_indices = rng.permutation(n)
            #     print(random_indices)

            # for pred in preds_n[::1]:
            #     if fill_negs <= 0:
            #         break

            #     # preds.append(pred[-10:-5])
            #     # labels.append(False)
            #     # abnormal_start_inds.append(0)
            #     # accident_inds.append(0)
            #     # abnormal_start_inds_no_pad.append(0)
            #     # accident_inds_no_pad.append(0)

            #     # fill_negs -= 1

            #     # if fill_negs <= 0:
            #     #     break

            #     preds.append(pred[0:5])
            #     labels.append(False)
            #     abnormal_start_inds.append(0)
            #     accident_inds.append(0)
            #     abnormal_start_inds_no_pad.append(0)
            #     accident_inds_no_pad.append(0)

            #     fill_negs -= 1

            while True:
                if len(preds_n) == 0:
                    break
                pred = preds_n.pop()
                preds.append(pred[-10:-5])
                # preds.append(pred[0:5])
                labels.append(False)
                abnormal_start_inds.append(0)
                accident_inds.append(0)
                abnormal_start_inds_no_pad.append(0)
                accident_inds_no_pad.append(0)

                fill_negs -= 1
                if fill_negs <= 0:
                    break        

    info = {
        "num_sequences": len(preds),
        "num_positive_sequences": int(sum(labels)),
        "num_negative_sequences": int(len(labels) - sum(labels)),
        "dropped_pos": dropped_pos,
        "dropped_neg": dropped_neg,
    }

    return preds, labels, abnormal_start_inds, accident_inds, info, preds_no_pad, abnormal_start_inds_no_pad, accident_inds_no_pad


# ------------------------------------------------------------
# Main API
# ------------------------------------------------------------

def evaluate_predictions(
    clip_outputs,
    anno_dict,
    score_key="risk_score",
    snippet_len=5,
    base_fps=10,
    fpr_max=0.1,
    plot_path=None,
    plot_title=None,
    match_neg_pos_numbers=False,
    convert_raw_logit=False,
):
    
    preds, labels, abnormal_start_inds, accident_inds, info, preds_no_pad, abnormal_start_inds_no_pad, accident_inds_no_pad = build_inputs_from_dense_clip_file(
        clip_outputs=clip_outputs,
        anno_dict=anno_dict,
        score_key=score_key,
        snippet_len=snippet_len,
        base_fps=base_fps,
        match_neg_pos_numbers=match_neg_pos_numbers,  # ensure balanced positives/negatives for fair AUC computation
        convert_raw_logit=convert_raw_logit,
    )

    print("Sequence build info:", info)

    if info["num_positive_sequences"] == 0 or info["num_negative_sequences"] == 0:
        raise ValueError(
            f"Need both positive and negative sequences. Got: {info}"
        )

    results = metric_calculate(
        preds=preds,
        labels=labels,
        abnormal_start_inds=abnormal_start_inds,
        accident_inds=accident_inds,
        fpr_max=fpr_max,
    )

    # new interval metrics
    interval_results = compute_interval_metrics(
        preds=preds_no_pad,
        labels=labels,
        abnormal_start_inds=abnormal_start_inds_no_pad,
        accident_inds=accident_inds_no_pad,
    )
    results.update(interval_results)

    if plot_path is not None:
        save_anticipation_roc_plot(
            preds=preds,
            labels=labels,
            save_path=plot_path,
            title=plot_title or f"Anticipation ROC ({score_key})",
            max_fpr=fpr_max,
            negative_mode="last5",   # or "allmax" if you want that variant
        )

    step_profile = compute_binned_step_profile(
        preds=preds_no_pad,
        abnormal_start_inds=abnormal_start_inds_no_pad,
        accident_inds=accident_inds_no_pad,
        labels=labels,
        n_pre_bins=50,
        n_post_bins=50,
    )


    return results, step_profile


# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------


if __name__ == "__main__":

    subset = 'cap'  # or 'dada', 'cap'
    scoring_key = "score" # risk_score or score
    match_neg_pos_numbers = False
    snippet_len = 5

    res_folder = f"./results_{subset}"
    anno_file = f"./annotations/mm_au_{subset}_anno.json"

    results_folders = sorted(os.listdir(res_folder))

    # load annotation once
    with open(anno_file, "r") as f:
        anno_dict = json.load(f)

    all_results = []
    model_profiles = {}
    naming_map = {}

    for res in results_folders:
        res_file = os.path.join(res_folder, res, "val_predictions.json")

        if not os.path.exists(res_file):
            print(f"Skipping {res}: file not found")
            continue

        with open(res_file, "r") as f:
            clip_outputs = json.load(f)

        out_score, step_profile = evaluate_predictions(
            clip_outputs=clip_outputs,
            anno_dict=anno_dict,
            score_key=scoring_key,
            snippet_len=snippet_len,
            base_fps=10,
            fpr_max=0.1,
            plot_path=os.path.join(res_folder, res, f"roc_{scoring_key}.png"),
            plot_title="Anticipation ROC",
            match_neg_pos_numbers=match_neg_pos_numbers,
            convert_raw_logit=False,
        )

        print_validation_metrics(
            out_score,
            epoch=5,
            num_epochs=30,
            global_step=1000,
            score_key=scoring_key,
        )

        if naming_map:
            # store one row per result folder
            row = {"result_folder": res}
            row.update(out_score)
            all_results.append(row)
            model_profiles[naming_map.get(res, res)] = step_profile

    if model_profiles:
        save_all_step_profiles(model_profiles, f"tikz_data_{subset}_{scoring_key}.csv")

        # plot step profiles for all models together
        plot_binned_step_profiles(
            model_profiles,
            title="Pre/Post-Anomaly Score Profiles",
            save_path=f"step_profile_models_{subset}_{scoring_key}.png",
            show_std=False,
            use_step=False,
        )

    # convert to table
    df = pd.DataFrame(all_results)

    # pretty print
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 4)

    # save as csv
    save_path = os.path.join(res_folder, f"summary_{scoring_key}_{subset}_raw_auc_all_models.csv")
    df.to_csv(save_path, index=False)
    print(f"\nSaved summary to: {save_path}")
