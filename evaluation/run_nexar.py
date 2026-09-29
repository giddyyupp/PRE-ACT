import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, auc
from evaluation.metrics import (
    partial_auc_normalized,
    _bin_segment,
    compute_interval_metrics,
    pad_repeat_first,
    pad_repeat_last,
)


# ------------------------------------------------------------
# Generic helpers
# ------------------------------------------------------------

def _parse_clip_name_generic(clip_name: str):
    """
    Supports clip names like:
      video_start
      part1_part2_start
    Returns:
      vid_id, start_frame(int)
    """
    splits = clip_name.split("_")
    if len(splits) == 2:
        vid_id, start_str = splits[0], splits[1]
    else:
        vid_id, start_str = "_".join(splits[:-1]), splits[-1]
    return vid_id, int(start_str)


def _nearest_nexar_horizon(x, allowed=(0.5, 1.0, 1.5), tol=0.26):
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    best = min(allowed, key=lambda a: abs(a - x))
    return best if abs(best - x) <= tol else None


def _video_label_from_anno(info):
    return info.get("abnormal_start_frame", None) is not None


# ------------------------------------------------------------
# Your custom pre/post metric helpers
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
        profiles.append(np.concatenate([pre_bins, post_bins], axis=0))

    if len(profiles) == 0:
        raise ValueError("No valid positive sequences found for step profile computation.")

    profiles = np.stack(profiles, axis=0)
    y = np.mean(profiles, axis=0)
    std = np.std(profiles, axis=0)

    x_pre = np.linspace(-1.0, 0.0, n_pre_bins, endpoint=False)
    x_post = np.linspace(0.0, 1.0, n_post_bins, endpoint=False) + (1.0 / n_post_bins) / 2.0
    x = np.concatenate([x_pre, x_post], axis=0)
    ideal = np.concatenate([np.zeros(n_pre_bins, dtype=np.float32), np.ones(n_post_bins, dtype=np.float32)])

    def _normalized_area(seg):
        seg = np.asarray(seg, dtype=np.float32)
        if len(seg) <= 1:
            return float(seg[0]) if len(seg) == 1 else np.nan
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


# ------------------------------------------------------------
# Official Nexar protocol: video-end AP by bucket
# ------------------------------------------------------------

def build_nexar_video_end_examples(
    clip_outputs,
    anno_dict,
    score_key="score",
    snippet_len=5,
    base_fps=10,
):
    """
    Build one score per video using the last visible clip region.

    For positives, bucket by time-to-accident (0.5 / 1.0 / 1.5 sec).
    For negatives, keep one score per negative video.

    Also builds sequences for your custom metrics and a virtual post-video tail for mTTA.
    """
    by_video = defaultdict(list)

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
        by_video[vid_id].append({
            "start_frame": start_frame,
            "end_frame": end_frame,
            "score": float(row[score_key]),
        })

    pos_scores_by_horizon = {0.5: [], 1.0: [], 1.5: []}
    neg_scores_by_horizon = {0.5: [], 1.0: [], 1.5: []}

    # for custom metrics on observed sequence only
    preds_no_pad = []
    labels = []
    abnormal_start_inds_no_pad = []
    accident_inds_no_pad = []

    # for mTTA with virtual tail
    preds_virtual = []
    abnormal_start_inds_virtual = []
    accident_inds_virtual = []

    info_out = {
        "num_videos": 0,
        "num_positive_videos": 0,
        "num_negative_videos": 0,
        "num_pos_0.5": 0,
        "num_pos_1.0": 0,
        "num_pos_1.5": 0,
    }

    for vid_id, clips in by_video.items():
        clips = sorted(clips, key=lambda x: x["start_frame"])
        if len(clips) == 0:
            continue

        info = anno_dict[vid_id]
        fps = float(info["fps"])
        abnormal_start_frame = info.get("abnormal_start_frame", None)
        accident_frame = info.get("accident_frame", None)
        label = _video_label_from_anno(info)

        seq = np.asarray([c["score"] for c in clips], dtype=np.float32)
        end_frames = np.asarray([c["end_frame"] for c in clips], dtype=np.int32)
        last_end_frame = int(end_frames[-1])
        final_score = float(np.max(seq[-1:])) if len(seq) >= 1 else float(np.max(seq))  # 

        info_out["num_videos"] += 1

        time_to_accident = info.get("time_to_accident_sec", None)
        horizon_bucket = _nearest_nexar_horizon(time_to_accident)

        if label:
            info_out["num_positive_videos"] += 1
            if horizon_bucket is not None:
                pos_scores_by_horizon[horizon_bucket].append(final_score)
                info_out[f"num_pos_{horizon_bucket:.1f}"] += 1

            # observed sequence only: pre = video start -> alert, post = alert -> video end
            if abnormal_start_frame is not None:
                ai_candidates = np.where(end_frames >= int(abnormal_start_frame))[0]
                ai_idx = int(ai_candidates[0]) if len(ai_candidates) > 0 else len(seq) - 1
            else:
                ai_idx = len(seq) - 1
            co_idx_obs = len(seq) - 1

            preds_no_pad.append(seq)
            labels.append(True)
            abnormal_start_inds_no_pad.append(ai_idx)
            accident_inds_no_pad.append(co_idx_obs)

            # virtual extension so mTTA remains defined until official accident frame
            if accident_frame is not None and int(accident_frame) > last_end_frame:
                sample_stride = max(1, int(round(fps / base_fps)))
                future_steps = int(np.ceil((float(accident_frame) - float(last_end_frame)) / sample_stride))
                seq_virtual = pad_repeat_last(seq, len(seq) + future_steps)
                co_idx_virtual = len(seq_virtual) - 1
            else:
                seq_virtual = seq.copy()
                co_idx_virtual = len(seq_virtual) - 1

            seq_virtual = pad_repeat_first(seq_virtual, 20)
            pad_len = len(seq_virtual) - len(seq) - max(0, len(seq_virtual) - max(len(seq), len(seq_virtual)))
            # simpler / correct pad len for positive virtual sequences:
            pad_len = len(seq_virtual) - (len(seq_virtual) if False else len(seq_virtual))
            # recompute explicitly
            unpadded_virtual_len = len(seq_virtual) if False else None
            # easier: rebuild
            if accident_frame is not None and int(accident_frame) > last_end_frame:
                sample_stride = max(1, int(round(fps / base_fps)))
                future_steps = int(np.ceil((float(accident_frame) - float(last_end_frame)) / sample_stride))
                seq_virtual_unpadded = pad_repeat_last(seq, len(seq) + future_steps)
            else:
                seq_virtual_unpadded = seq.copy()
            old_len = len(seq_virtual_unpadded)
            seq_virtual = pad_repeat_first(seq_virtual_unpadded, 20)
            pad_len = len(seq_virtual) - old_len
            preds_virtual.append(seq_virtual)
            abnormal_start_inds_virtual.append(ai_idx + pad_len)
            accident_inds_virtual.append(len(seq_virtual_unpadded) - 1 + pad_len)

        else:
            info_out["num_negative_videos"] += 1
            if horizon_bucket is not None:
                neg_scores_by_horizon[horizon_bucket].append(final_score)

            preds_no_pad.append(seq)
            labels.append(False)
            abnormal_start_inds_no_pad.append(0)
            accident_inds_no_pad.append(0)

            seq_virtual = pad_repeat_first(seq, 5)
            preds_virtual.append(seq_virtual)
            abnormal_start_inds_virtual.append(0)
            accident_inds_virtual.append(0)

    return {
        "pos_scores_by_horizon": pos_scores_by_horizon,
        "neg_scores_by_horizon": neg_scores_by_horizon,
        "preds_no_pad": preds_no_pad,
        "labels": labels,
        "abnormal_start_inds_no_pad": abnormal_start_inds_no_pad,
        "accident_inds_no_pad": accident_inds_no_pad,
        "preds_virtual": preds_virtual,
        "abnormal_start_inds_virtual": abnormal_start_inds_virtual,
        "accident_inds_virtual": accident_inds_virtual,
        "info": info_out,
    }


def compute_nexar_official_ap_metrics(pos_scores_by_horizon, neg_scores_by_horizon):
    results = {}
    if len(neg_scores_by_horizon) == 0:
        raise ValueError("No negative videos available.")

    aps = []
    aucs = []

    for h in [0.5, 1.0, 1.5]:
        pos_scores = np.asarray(pos_scores_by_horizon[h], dtype=np.float32)
        neg_scores = np.asarray(neg_scores_by_horizon[h], dtype=np.float32)

        if len(pos_scores) == 0:
            results[f"AP@{h:.1f}s"] = np.nan
            results[f"AUC@{h:.1f}s"] = np.nan
            results[f"pAUC@0.1@{h:.1f}s"] = np.nan
            continue

        y_true = np.array([1] * len(pos_scores) + [0] * len(neg_scores), dtype=np.int32)
        y_score = np.concatenate([pos_scores, neg_scores], axis=0)

        ap = average_precision_score(y_true, y_score)
        full_auc = roc_auc_score(y_true, y_score)
        p_auc = partial_auc_normalized(y_true, y_score, max_fpr=0.1)

        results[f"AP@{h:.1f}s"] = float(ap)
        results[f"AUC@{h:.1f}s"] = float(full_auc)
        results[f"pAUC@0.1@{h:.1f}s"] = float(p_auc)
        aps.append(ap)
        aucs.append(full_auc)

    results["mAP"] = float(np.nanmean(aps)) if len(aps) else np.nan
    results["mAUC"] = float(np.nanmean(aucs)) if len(aucs) else np.nan
    results["num_negative_videos"] = int(len(neg_scores))
    results["num_positive_0.5"] = int(len(pos_scores_by_horizon[0.5]))
    results["num_positive_1.0"] = int(len(pos_scores_by_horizon[1.0]))
    results["num_positive_1.5"] = int(len(pos_scores_by_horizon[1.5]))
    return results


# ------------------------------------------------------------
# Optional: mTTA on Nexar virtual tail
# ------------------------------------------------------------

def mtta_only(preds, labels, abnormal_start_inds, accident_inds, fpr_max=0.1):
    out = {}
    preds_n = [pred for pred, label in zip(preds, labels) if not label]
    if len(preds_n) == 0:
        raise ValueError("No negative sequences for mTTA computation.")
    preds_n = np.concatenate(preds_n)
    preds_n = -np.sort(-preds_n)
    threshold = preds_n[int(len(preds_n) * fpr_max)]
    out[f"threshold@{fpr_max:.2f}"] = float(threshold)

    ttas = [
        (j - i - np.argmax(pred[i:j + 1, None] >= preds_n, axis=0)) / 10
        * np.any(pred[i:j + 1, None] >= preds_n, axis=0)
        for pred, label, i, j in zip(preds, labels, abnormal_start_inds, accident_inds)
        if label
    ]
    ttas = np.array(ttas).mean(axis=0)

    out["tta@0.01"] = float(ttas[int(len(ttas) * 0.01)])
    out["tta@0.05"] = float(ttas[int(len(ttas) * 0.05)])
    out["tta@0.1"] = float(ttas[int(len(ttas) * fpr_max)])
    out["mtta@0.1"] = float(ttas[:int(len(ttas) * fpr_max)].mean())
    out["tta@1"] = float(ttas[int(len(ttas) * 0.99)])
    return out


# ------------------------------------------------------------
# ROC plotting
# ------------------------------------------------------------

def save_nexar_official_roc_plot(pos_scores_by_horizon, neg_scores_by_horizon, save_path, title="Nexar official ROC"):
    plt.figure(figsize=(7, 5.5))
    # neg_scores_arr = np.asarray(neg_scores, dtype=np.float32)

    for h in [0.5, 1.0, 1.5]:
        pos_scores = np.asarray(pos_scores_by_horizon[h], dtype=np.float32)
        neg_scores = np.asarray(neg_scores_by_horizon[h], dtype=np.float32)
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            continue

        y_true = np.array([1] * len(pos_scores) + [0] * len(neg_scores), dtype=np.int32)
        y_score = np.concatenate([pos_scores, neg_scores], axis=0)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        plt.plot(fpr, tpr, linewidth=2, label=f"{h:.1f}s (AP={ap:.3f})")

    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.xticks(np.arange(0.0, 1.01, 0.1))
    plt.yticks(np.arange(0.0, 1.01, 0.1))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------
# Main API
# ------------------------------------------------------------

def evaluate_nexar_official(
    clip_outputs,
    anno_dict,
    score_key="score",
    snippet_len=5,
    base_fps=10,
    fpr_max=0.1,
    plot_path=None,
    plot_title=None,
):
    pack = build_nexar_video_end_examples(
        clip_outputs=clip_outputs,
        anno_dict=anno_dict,
        score_key=score_key,
        snippet_len=snippet_len,
        base_fps=base_fps,
    )

    results = {}
    results.update(pack["info"])
    results.update(compute_nexar_official_ap_metrics(pack["pos_scores_by_horizon"], pack["neg_scores_by_horizon"]))
    results.update(mtta_only(
        preds=pack["preds_virtual"],
        labels=pack["labels"],
        abnormal_start_inds=pack["abnormal_start_inds_virtual"],
        accident_inds=pack["accident_inds_virtual"],
        fpr_max=fpr_max,
    ))
    results.update(compute_interval_metrics(
        preds=pack["preds_no_pad"],
        labels=pack["labels"],
        abnormal_start_inds=pack["abnormal_start_inds_no_pad"],
        accident_inds=pack["accident_inds_no_pad"],
    ))

    step_profile = compute_binned_step_profile(
        preds=pack["preds_no_pad"],
        labels=pack["labels"],
        abnormal_start_inds=pack["abnormal_start_inds_no_pad"],
        accident_inds=pack["accident_inds_no_pad"],
        n_pre_bins=50,
        n_post_bins=50,
    )

    if plot_path is not None:
        save_nexar_official_roc_plot(
            pos_scores_by_horizon=pack["pos_scores_by_horizon"],
            neg_scores_by_horizon=pack["neg_scores_by_horizon"],
            save_path=plot_path,
            title=plot_title or "Nexar official ROC/AP",
        )

    return results, step_profile


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    subset = "nexar"
    scoring_key = "score" # or risk_score, score
    snippet_len = 5

    res_folder = f"./results_{subset}"
    anno_file = f"./annotations/{subset}_anno.json"

    results_folders = sorted(os.listdir(res_folder))

    # load annotation once
    with open(anno_file, "r") as f:
        anno_dict = json.load(f)

    for res in results_folders:
        
        res_file = os.path.join(res_folder, res, "val_predictions.json")

        if not os.path.exists(res_file):
            print(f"Skipping {res}: file not found")
            continue

        with open(res_file, "r") as f:
            clip_outputs = json.load(f)
    
        results, _ = evaluate_nexar_official(
            clip_outputs=clip_outputs,
            anno_dict=anno_dict,
            score_key=scoring_key,
            snippet_len=snippet_len,
            base_fps=10,
            fpr_max=0.1,
            plot_path=os.path.join(res_folder, res, f"roc_{scoring_key}.png"),
        )

        print(res)

        df = pd.DataFrame([results])
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
