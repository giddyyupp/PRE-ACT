import json
import os
import csv
from typing import Dict, List, Optional, Tuple
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, auc

from evaluation.run_mmau import plot_binned_step_profiles, save_all_step_profiles
from evaluation.metrics import (
    partial_auc_normalized,
    interval_auc_normalized,
    _bin_segment,
    compute_interval_metrics,
    pad_repeat_first,
)


# ------------------------------------------------------------
# Generic helpers copied/adapted from current evaluator
# ------------------------------------------------------------

def compute_binned_step_profile(
    preds,
    abnormal_start_inds,
    accident_inds,
    labels=None,
    n_pre_bins=50,
    n_post_bins=50,
):
    profiles = []

    for i, seq in enumerate(preds):
        if labels is not None and not bool(labels[i]):
            continue

        seq = np.asarray(seq, dtype=np.float32)
        ai_idx = int(abnormal_start_inds[i])
        co_idx = int(accident_inds[i])

        if len(seq) == 0 or ai_idx < 0 or co_idx < 0 or ai_idx >= len(seq):
            continue
        if co_idx >= len(seq):
            co_idx = len(seq) - 1
        if co_idx < ai_idx:
            continue

        pre_seg = seq[:ai_idx]
        post_seg = seq[ai_idx:co_idx + 1]

        pre_bins = _bin_segment(pre_seg, n_pre_bins)
        post_bins = _bin_segment(post_seg, n_post_bins)
        profile = np.concatenate([pre_bins, post_bins], axis=0)
        profiles.append(profile)

    if len(profiles) == 0:
        raise ValueError("No valid positive sequences found for profile computation.")

    profiles = np.stack(profiles, axis=0)
    y = np.mean(profiles, axis=0)
    std = np.std(profiles, axis=0)

    x_pre = np.linspace(-1.0, 0.0, n_pre_bins, endpoint=False)
    x_post = np.linspace(0.0, 1.0, n_post_bins, endpoint=False) + (1.0 / n_post_bins) / 2.0
    x = np.concatenate([x_pre, x_post], axis=0)

    ideal = np.concatenate([
        np.zeros(n_pre_bins, dtype=np.float32),
        np.ones(n_post_bins, dtype=np.float32),
    ])

    def _normalized_area(seg):
        seg = np.asarray(seg, dtype=np.float32)
        if len(seg) == 0:
            return np.nan
        if len(seg) == 1:
            return float(seg[0])
        area = np.trapezoid(seg, dx=1.0)
        return float(area / (len(seg) - 1))

    pre_area = _normalized_area(y[:n_pre_bins])
    post_area = _normalized_area(y[n_pre_bins:])

    return {
        "x": x,
        "y": y,
        "std": std,
        "all_profiles": profiles,
        "ideal": ideal,
        "pre_area": pre_area,
        "post_area": post_area,
    }



def save_anticipation_roc_plot(preds, labels, save_path, title="Anticipation ROC", max_fpr=0.1, negative_mode="last5"):
    preds_0 = [np.max(pred[-5:]) for pred, label in zip(preds, labels) if label]
    preds_5 = [np.max(pred[-10:-5]) for pred, label in zip(preds, labels) if label and len(pred) >= 10]
    preds_10 = [np.max(pred[-15:-10]) for pred, label in zip(preds, labels) if label and len(pred) >= 15]
    preds_15 = [np.max(pred[-20:-15]) for pred, label in zip(preds, labels) if label and len(pred) >= 20]

    neg_preds_raw = [pred for pred, label in zip(preds, labels) if not label]
    if negative_mode == "last5":
        preds_n = [np.max(pred[-5:]) for pred in neg_preds_raw if len(pred) >= 5]
    elif negative_mode == "allmax":
        preds_n = [np.max(pred) for pred in neg_preds_raw if len(pred) > 0]
    else:
        raise ValueError(f"Unknown negative_mode: {negative_mode}")

    curve_defs = [
        ("0.0s before accident", preds_0),
        ("0.5s before accident", preds_5),
        ("1.0s before accident", preds_10),
        ("1.5s before accident", preds_15),
    ]

    plt.figure(figsize=(7, 5.5))
    for label_name, pos_scores in curve_defs:
        if len(pos_scores) == 0 or len(preds_n) == 0:
            continue
        y_true = np.array([1] * len(pos_scores) + [0] * len(preds_n), dtype=np.int32)
        y_score = np.array(pos_scores + preds_n, dtype=np.float32)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_full = roc_auc_score(y_true, y_score)
        auc_p01 = partial_auc_normalized(y_true, y_score, max_fpr=max_fpr)
        plt.plot(fpr, tpr, linewidth=2, label=f"{label_name} (AUC={auc_full:.3f}, pAUC@{max_fpr:.1f}={auc_p01:.3f})")

    plt.axvline(max_fpr, linestyle="--", linewidth=1.5, color="gray")
    plt.plot([0, max_fpr], [1, 1], linestyle="--", linewidth=1.5, color="gray")
    plt.plot([max_fpr, max_fpr], [0, 1], linestyle="--", linewidth=1.5, color="gray")
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


# ------------------------------------------------------------
# DAD-specific builder/evaluator
# ------------------------------------------------------------


def _parse_clip_name_generic(clip_name: str) -> Tuple[str, int]:
    splits = clip_name.split("_")
    if len(splits) == 2:
        return splits[0], int(splits[1])
    return "_".join(splits[:-1]), int(splits[-1])


def build_inputs_from_dense_clip_file(
    clip_outputs: List[Dict],
    anno_dict: Dict,
    score_key: str = "score",
    snippet_len: int = 5,
    base_fps: int = 10,
    positive_tail_pad_mode: str = "repeat_last",
):
    """
    Build sequences for DAD.

    Positive videos:
      - anomaly begins inside the video
      - accident may happen after the video ends
      - we extend the sequence virtually from video-end to accident-time by
        repeating the last visible score; this keeps AUC/mTTA well defined

    Negative videos:
      - abnormal_start_frame is None
      - sequence stays as observed

    Notes
    -----
    - Metrics use 10-FPS-equivalent indexing, so we still align clips to
      base_fps and convert accident timing to virtual clip indices using the same stride.
    - For the custom pre/post metrics and the step-profile metric, we use the
      *observed* sequence only and define the post region as alert -> video end.
    """
    by_video: Dict[str, List[Dict]] = {}

    for row in clip_outputs:
        clip_name = row["clip_names"]
        vid_id, start_frame = _parse_clip_name_generic(clip_name)
        if vid_id not in anno_dict:
            continue

        info = anno_dict[vid_id]
        fps = float(info["fps"])
        sample_stride = max(1, int(round(fps / base_fps)))

        if (start_frame - 1) % sample_stride != 0:
            continue

        end_frame = start_frame + (snippet_len - 1) * sample_stride
        item = {
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "score": float(row[score_key]),
        }
        by_video.setdefault(vid_id, []).append(item)

    preds = []
    labels = []
    abnormal_start_inds = []
    accident_inds = []

    preds_no_pad = []
    abnormal_start_inds_no_pad = []
    accident_inds_no_pad = []

    dropped_pos = 0
    dropped_neg = 0

    for vid_id, clips in by_video.items():
        clips = sorted(clips, key=lambda x: x["start_frame"])
        if len(clips) == 0:
            continue

        info = anno_dict[vid_id]
        fps = float(info["fps"])
        sample_stride = max(1, int(round(fps / base_fps)))

        abnormal_start_frame_raw = info.get("abnormal_start_frame", None)
        accident_frame = int(float(info["accident_frame"]))
        num_images = int(info.get("num_images", clips[-1]["end_frame"]))

        seq_obs = np.array([c["score"] for c in clips], dtype=np.float32)
        end_frames_obs = np.array([c["end_frame"] for c in clips], dtype=np.int32)

        # Negative video: no anomaly, no accident after video end that should count as positive
        if abnormal_start_frame_raw is None:
            neg_seq = pad_repeat_first(seq_obs, 5)
            preds.append(neg_seq)
            labels.append(False)
            abnormal_start_inds.append(0)
            accident_inds.append(0)

            preds_no_pad.append(seq_obs)
            abnormal_start_inds_no_pad.append(0)
            accident_inds_no_pad.append(0)
            continue

        abnormal_start_frame = int(float(abnormal_start_frame_raw))

        # alert index on observed sequence
        ai_candidates = np.where(end_frames_obs >= abnormal_start_frame)[0]
        if len(ai_candidates) == 0:
            dropped_pos += 1
            continue
        ai_idx_obs = int(ai_candidates[0])

        # accident may be after video end; compute virtual accident index using the
        # same clip-end spacing as base_fps alignment.
        last_end_frame = int(end_frames_obs[-1])
        co_idx_obs_end = len(seq_obs) - 1

        if accident_frame <= last_end_frame:
            co_candidates = np.where(end_frames_obs >= accident_frame)[0]
            co_idx_virtual = int(co_candidates[0]) if len(co_candidates) > 0 else co_idx_obs_end
            seq_virtual = seq_obs.copy()
        else:
            extra_steps = int(np.ceil((accident_frame - last_end_frame) / float(sample_stride)))
            extra_steps = max(1, extra_steps)
            last_score = seq_obs[-1]
            tail = np.full(extra_steps, last_score, dtype=np.float32)
            seq_virtual = np.concatenate([seq_obs, tail], axis=0)
            co_idx_virtual = len(seq_virtual) - 1

        # Metrics use virtual sequence
        old_len = len(seq_virtual)
        seq_virtual_pad = pad_repeat_first(seq_virtual, 20)
        pad_len = len(seq_virtual_pad) - old_len

        preds.append(seq_virtual_pad)
        labels.append(True)
        abnormal_start_inds.append(ai_idx_obs + pad_len)
        accident_inds.append(co_idx_virtual + pad_len)

        # custom profile metric uses observed-only sequence, with post region = alert -> video end
        preds_no_pad.append(seq_obs)
        abnormal_start_inds_no_pad.append(ai_idx_obs)
        accident_inds_no_pad.append(co_idx_obs_end)

    info = {
        "num_sequences": len(preds),
        "num_positive_sequences": int(sum(labels)),
        "num_negative_sequences": int(len(labels) - sum(labels)),
        "dropped_pos": dropped_pos,
        "dropped_neg": dropped_neg,
    }

    return (
        preds,
        labels,
        abnormal_start_inds,
        accident_inds,
        info,
        preds_no_pad,
        abnormal_start_inds_no_pad,
        accident_inds_no_pad,
    )



def calculate_metrics(preds, labels, abnormal_start_inds, accident_inds, fpr_max=0.1):
    eval_results = {}

    if len(preds) > 0 and len(preds[0].shape) > 1:
        preds = [np.max(pred, axis=-1) for pred in preds]

    preds_n = [pred for pred, label in zip(preds, labels) if not label]
    preds_n = np.concatenate(preds_n)
    preds_n = -np.sort(-preds_n)
    eval_results[f"threshold@{fpr_max:.2f}"] = preds_n[int(len(preds_n) * fpr_max)]

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

    preds_0 = [np.max(pred[-5:]) for pred, label in zip(preds, labels) if label]
    preds_5 = [np.max(pred[-10:-5]) for pred, label in zip(preds, labels) if label]
    preds_10 = [np.max(pred[-15:-10]) for pred, label in zip(preds, labels) if label]
    preds_15 = [np.max(pred[-20:-15]) for pred, label in zip(preds, labels) if label]
    preds_n = [np.max(pred[-5:]) for pred, label in zip(preds, labels) if not label]

    y = [1] * len(preds_0) + [0] * len(preds_n)
    y = np.array(y)

    # Keep same reporting style as current script: partial AUC normalized @0.1
    eval_results["AUC@0.0s"] = partial_auc_normalized(y, np.array(preds_0 + preds_n), fpr_max)
    eval_results["AUC@0.5s"] = partial_auc_normalized(y, np.array(preds_5 + preds_n), fpr_max)
    eval_results["AUC@1.0s"] = partial_auc_normalized(y, np.array(preds_10 + preds_n), fpr_max)
    eval_results["AUC@1.5s"] = partial_auc_normalized(y, np.array(preds_15 + preds_n), fpr_max)

    eval_results["mAUC@0.1"] = (
        eval_results["AUC@0.5s"]
        + eval_results["AUC@1.0s"]
        + eval_results["AUC@1.5s"]
    ) / 3.0

    eval_results["AUC_0.01@0.0s"] = partial_auc_normalized(np.array(y), np.array(preds_0 + preds_n), 0.01)
    eval_results["AUC_0.01@0.5s"] = partial_auc_normalized(np.array(y), np.array(preds_5 + preds_n), 0.01)
    eval_results["AUC_0.01@1.0s"] = partial_auc_normalized(np.array(y), np.array(preds_10 + preds_n), 0.01)
    eval_results["AUC_0.01@1.5s"] = partial_auc_normalized(np.array(y), np.array(preds_15 + preds_n), 0.01)

    eval_results["mAUC@0.01"] = (
        eval_results["AUC_0.01@0.5s"]
        + eval_results["AUC_0.01@1.0s"]
        + eval_results["AUC_0.01@1.5s"]
    ) / 3


    eval_results["AUC_full@0.0s"] = roc_auc_score(y, np.array(preds_0 + preds_n))
    eval_results["AUC_full@0.5s"] = roc_auc_score(y, np.array(preds_5 + preds_n))
    eval_results["AUC_full@1.0s"] = roc_auc_score(y, np.array(preds_10 + preds_n))
    eval_results["AUC_full@1.5s"] = roc_auc_score(y, np.array(preds_15 + preds_n))

    eval_results["mAUC"] = (
        eval_results["AUC_full@0.5s"]
        + eval_results["AUC_full@1.0s"]
        + eval_results["AUC_full@1.5s"]
    ) / 3.0

    eval_results["AP@0.0s"] = average_precision_score(y, np.array(preds_0 + preds_n))
    eval_results["AP@0.5s"] = average_precision_score(y, np.array(preds_5 + preds_n))
    eval_results["AP@1.0s"] = average_precision_score(y, np.array(preds_10 + preds_n))
    eval_results["AP@1.5s"] = average_precision_score(y, np.array(preds_15 + preds_n))

    eval_results["mAP"] = (
        eval_results["AP@0.5s"]
        + eval_results["AP@1.0s"]
        + eval_results["AP@1.5s"]
    ) / 3.0

    eval_results["num_samples"] = len(preds)
    return eval_results



def evaluate_predictions_dad(
    clip_outputs,
    anno_dict,
    score_key="score",
    snippet_len=5,
    base_fps=10,
    fpr_max=0.1,
    plot_path: Optional[str] = None,
    plot_title: Optional[str] = None,
):
    (
        preds,
        labels,
        abnormal_start_inds,
        accident_inds,
        info,
        preds_no_pad,
        abnormal_start_inds_no_pad,
        accident_inds_no_pad,
    ) = build_inputs_from_dense_clip_file(
        clip_outputs=clip_outputs,
        anno_dict=anno_dict,
        score_key=score_key,
        snippet_len=snippet_len,
        base_fps=base_fps,
    )

    print("DAD sequence build info:", info)
    if info["num_positive_sequences"] == 0 or info["num_negative_sequences"] == 0:
        raise ValueError(f"Need both positive and negative sequences. Got: {info}")

    results = calculate_metrics(
        preds=preds,
        labels=labels,
        abnormal_start_inds=abnormal_start_inds,
        accident_inds=accident_inds,
        fpr_max=fpr_max,
    )

    interval_results = compute_interval_metrics(
        preds=preds_no_pad,
        labels=labels,
        abnormal_start_inds=abnormal_start_inds_no_pad,
        accident_inds=accident_inds_no_pad,
    )
    results.update(interval_results)

    step_profile = compute_binned_step_profile(
        preds=preds_no_pad,
        abnormal_start_inds=abnormal_start_inds_no_pad,
        accident_inds=accident_inds_no_pad,
        labels=labels,
        n_pre_bins=50,
        n_post_bins=50,
    )

    if plot_path is not None:
        save_anticipation_roc_plot(
            preds=preds,
            labels=labels,
            save_path=plot_path,
            title=plot_title or f"DAD Anticipation ROC ({score_key})",
            max_fpr=fpr_max,
            negative_mode="last5",
        )

    return results, step_profile


if __name__ == "__main__":

    subset = "dad"
    scoring_key = "score" # or risk_score, score

    res_folder = f"./results_{subset.upper()}"
    anno_file = f"./annotations/{subset}_anno.json"

    results_folders = sorted(os.listdir(res_folder))

    # load annotation once
    with open(anno_file, "r") as f:
        anno_dict = json.load(f)

    all_results = []
    model_profiles = {}

    plot_models = ["my_model"]
    naming_map = {"my_model": "Progress Loss Exp5",}

    for res in results_folders:
        
        res_file = os.path.join(res_folder, res, "val_predictions.json")

        if not os.path.exists(res_file):
            print(f"Skipping {res}: file not found")
            continue

        with open(res_file, "r") as f:
            clip_outputs = json.load(f)

        results, step_profile = evaluate_predictions_dad(
            clip_outputs=clip_outputs,
            anno_dict=anno_dict,
            score_key=scoring_key,
            snippet_len=5,
            base_fps=10,
            fpr_max=0.1,
            plot_path=os.path.join(res_folder, res, f"roc_{scoring_key}.png"),
        )

        # store one row per result folder
        row = {"result_folder": res}
        row.update(results)
        all_results.append(row)
        model_profiles[naming_map.get(res, res)] = step_profile

    
    save_all_step_profiles(model_profiles, f"tikz_data_{subset}")

    # plot step profiles for all models together
    plot_binned_step_profiles(
        model_profiles,
        title="Pre/Post-Anomaly Score Profiles",
        save_path=f"step_profile_models_{subset}.png",
        show_std=False,
        use_step=False,
    )

    # convert to table
    df = pd.DataFrame(all_results)

    # pretty print
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 4)

    # print("\n=== Summary table using score ===\n")
    # print(df.round(4).to_string(index=False))

    # save as csv
    save_path = os.path.join(res_folder, f"summary_{scoring_key}_{subset}_raw_auc_match_pos_neg_all.csv")
    df.to_csv(save_path, index=False)
    print(f"\nSaved summary to: {save_path}")
