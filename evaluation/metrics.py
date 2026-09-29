import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve, 
    auc
)


def collect_horizon_scores_with_shared_negatives(eval_outputs, horizons=(0.5, 1.0, 1.5)):
    negatives = []
    positives = {h: [] for h in horizons}

    for x in eval_outputs:
        if x["label"] == 0:
            negatives.append(x)
        else:
            h = x["horizon_sec"]
            if h in positives:
                positives[h].append(x)

    result = {}
    for h in horizons:
        y_true, y_score = [], []

        for p in positives[h]:
            y_true.append(1)
            y_score.append(float(p["score"]))

        for n in negatives:
            y_true.append(0)
            y_score.append(float(n["score"]))

        result[h] = (y_true, y_score)

    return result


def auc_lambda(y_true, y_score, lambda_far=0.1):
    """
    AUC restricted to FPR <= lambda_far.
    Equivalent to AUC_λ in the TOP paper.

    y_true: 0/1 labels
    y_score: continuous risk scores
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if len(np.unique(y_true)) < 2:
        return 0.0

    fpr, tpr, _ = roc_curve(y_true, y_score)

    # keep portion up to lambda_far, including interpolation at boundary
    if lambda_far <= fpr[0]:
        return 0.0

    if lambda_far >= fpr[-1]:
        return float(auc(fpr, tpr))

    idx = np.searchsorted(fpr, lambda_far, side="right")
    fpr_keep = fpr[:idx].tolist()
    tpr_keep = tpr[:idx].tolist()

    # interpolate TPR at lambda_far
    x0, x1 = fpr[idx - 1], fpr[idx]
    y0, y1 = tpr[idx - 1], tpr[idx]
    if x1 == x0:
        y_lambda = y1
    else:
        y_lambda = y0 + (lambda_far - x0) * (y1 - y0) / (x1 - x0)

    if fpr_keep[-1] != lambda_far:
        fpr_keep.append(lambda_far)
        tpr_keep.append(y_lambda)

    # return float(auc(np.array(fpr_keep), np.array(tpr_keep)))
    return float(auc(np.array(fpr_keep), np.array(tpr_keep)) / lambda_far)


def mean_auc_lambda(horizon_to_labels_scores, lambda_far=0.1):
    """
    horizon_to_labels_scores:
    {
      0.5: (y_true_05, y_score_05),
      1.0: (y_true_10, y_score_10),
      1.5: (y_true_15, y_score_15),
    }
    """
    vals = []
    for h in sorted(horizon_to_labels_scores.keys()):
        y_true, y_score = horizon_to_labels_scores[h]
        vals.append(auc_lambda(y_true, y_score, lambda_far=lambda_far))
    return float(np.mean(vals)) if vals else 0.0


def compute_far_for_threshold(video_predictions, threshold):
    """
    video_predictions: list of dicts, each containing:
      {
        "scores": np.ndarray shape [num_frames], score per frame/snippet endpoint
        "t_ai": int,
        "t_co": int,
      }

    FAR here is implemented as frame-level FPR over pre-anomaly frames,
    matching the paper's use of the segment before t_ai as negatives.
    """
    false_alarms = 0
    total_negative_frames = 0

    for vp in video_predictions:
        scores = np.asarray(vp["scores"])
        t_ai = int(vp["t_ai"])

        neg_scores = scores[:max(0, t_ai - 1)]
        false_alarms += int((neg_scores >= threshold).sum())
        total_negative_frames += len(neg_scores)

    if total_negative_frames == 0:
        return 0.0
    return false_alarms / total_negative_frames


def compute_mtta_lambda(video_predictions, fps=10, lambda_far=0.1, num_thresholds=500):
    """
    Returns the best mTTA among thresholds satisfying FAR <= lambda_far.

    TTA for one video:
    - first alarm after anomaly appears and before collision
    - TTA = (t_co - alarm_frame) / fps
    - if no valid alarm, TTA = 0

    This follows the revised TTA idea in TOP.
    """
    all_scores = np.concatenate([np.asarray(v["scores"]) for v in video_predictions])
    thresholds = np.unique(np.quantile(all_scores, np.linspace(0.0, 1.0, num_thresholds)))

    valid_mttas = []

    for thr in thresholds:
        far = compute_far_for_threshold(video_predictions, thr)
        if far > lambda_far:
            continue

        tta_list = []
        for vp in video_predictions:
            scores = np.asarray(vp["scores"])
            t_ai = int(vp["t_ai"])
            t_co = int(vp["t_co"])

            start = max(0, t_ai - 1)
            end = min(len(scores), t_co)

            valid_alarm_idx = None
            for i in range(start, end):
                if scores[i] >= thr:
                    valid_alarm_idx = i + 1  # convert back to 1-based frame index
                    break

            if valid_alarm_idx is None:
                tta_list.append(0.0)
            else:
                tta_sec = max(0.0, (t_co - valid_alarm_idx) / fps)
                tta_list.append(tta_sec)

        valid_mttas.append(float(np.mean(tta_list)))

    if not valid_mttas:
        return 0.0
    return max(valid_mttas)



def binary_classification_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["ap"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["roc_auc"] = 0.0
        metrics["ap"] = 0.0

    return metrics


def _evaluate_threshold_tta_with_all_negative_frames(
    video_predictions,
    thr,
    anticipation_horizon_sec=2.0,
    include_post_collision_in_fpr=False,
):
    """
    Evaluate one threshold.

    FPR:
      - computed on all negative frames
      - for accident videos: frames with TTC > anticipation_horizon_sec
      - for non-accident videos: all frames

    TTA:
      - computed from the first alarm in [t_ai, t_co]
      - pre-anomaly alarms do NOT invalidate TTA
      - they are already penalized through FPR
    """
    fp = 0
    neg = 0
    tta_values = []

    for vp in video_predictions:
        scores = np.asarray(vp["scores"], dtype=np.float32)
        t_ai = int(vp["t_ai"])
        t_co = int(vp["t_co"])
        fps = float(vp.get("fps", 10.0))

        n_frames = len(scores)
        horizon_frames = int(round(anticipation_horizon_sec * fps))

        # -------------------------
        # FPR on all negative frames
        # -------------------------
        if t_co < 0:
            # non-accident video: all frames are negative
            neg_mask = np.ones(n_frames, dtype=bool)
        else:
            frame_ids_1b = np.arange(1, n_frames + 1)
            ttc_frames = t_co - frame_ids_1b

            # negative if still outside anticipation horizon
            neg_mask = ttc_frames > horizon_frames

            if not include_post_collision_in_fpr:
                neg_mask &= frame_ids_1b <= t_co

        neg_scores = scores[neg_mask]
        fp += int((neg_scores >= thr).sum())
        neg += int(len(neg_scores))

        # -------------------------
        # TTA for accident videos
        # -------------------------
        if t_co < 0:
            continue

        # search only after anomaly start until accident
        start = max(0, t_ai - 1)
        end = min(n_frames, t_co)   # Python exclusive
        valid_scores = scores[start:end]

        hit = np.where(valid_scores >= thr)[0]

        if len(hit) == 0:
            tta_values.append(0.0)
        else:
            first_alarm_frame_1b = start + int(hit[0]) + 1
            tta_sec = max(0.0, (t_co - first_alarm_frame_1b) / fps)
            tta_values.append(tta_sec)

    fpr = fp / neg if neg > 0 else 0.0
    mean_tta = float(np.mean(tta_values)) if len(tta_values) > 0 else 0.0
    return fpr, mean_tta


def compute_mtta_top_style(
    video_predictions,
    fpr_targets=(0.01, 0.10, 1.00),
    anticipation_horizon_sec=2.0,
    include_post_collision_in_fpr=False,
    num_thresholds=1000,
):
    """
    Compute mTTA at requested FPR targets.

    Returns:
      {
        "mTTA@0.01": ...,
        "mTTA@0.05": ...,
        "mTTA@0.10": ...,
        "mTTA@1.00": ...,
        "curve": {
            "fpr": [...],
            "mtta": [...],
            "thresholds": [...]
        }
      }
    """
    all_scores = []
    for vp in video_predictions:
        all_scores.extend(np.asarray(vp["scores"], dtype=np.float32).tolist())

    if len(all_scores) == 0:
        raise ValueError("video_predictions is empty or has no scores.")

    smin = float(np.min(all_scores))
    smax = float(np.max(all_scores))

    thresholds = np.linspace(smax + 1e-8, smin - 1e-8, num_thresholds)

    fpr_curve = []
    mtta_curve = []

    for thr in thresholds:
        fpr, mtta = _evaluate_threshold_tta_with_all_negative_frames(
            video_predictions=video_predictions,
            thr=thr,
            anticipation_horizon_sec=anticipation_horizon_sec,
            include_post_collision_in_fpr=include_post_collision_in_fpr,
        )
        fpr_curve.append(fpr)
        mtta_curve.append(mtta)

    fpr_curve = np.asarray(fpr_curve, dtype=np.float32)
    mtta_curve = np.asarray(mtta_curve, dtype=np.float32)

    results = {
        "curve": {
            "fpr": fpr_curve.tolist(),
            "mtta": mtta_curve.tolist(),
            "thresholds": thresholds.tolist(),
        }
    }

    for target in fpr_targets:
        valid = np.where(fpr_curve <= target)[0]
        if len(valid) == 0:
            results[f"mTTA@{target:.2f}"] = 0.0
        else:
            results[f"mTTA@{target:.2f}"] = float(np.max(mtta_curve[valid]))

    return results


# ---------------------------------------------------------------
# Shared metric primitives, previously duplicated across
# evaluation/run_{mmau,nexar,dad}.py. Verified numerically identical
# across all copies before consolidation.
# ---------------------------------------------------------------

def sklearn_auc(y_true, y_scores, fpr_max=0.1):
    score = roc_auc_score(y_true, y_scores, max_fpr=fpr_max)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return score, fpr, tpr


def partial_auc_raw(y_true, y_scores, max_fpr=0.1):
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    stop_idx = np.searchsorted(fpr, max_fpr, side='right')

    fpr_sliced = fpr[:stop_idx].copy()
    tpr_sliced = tpr[:stop_idx].copy()

    if len(fpr_sliced) < len(fpr) and fpr_sliced[-1] < max_fpr:
        x1, x2 = fpr[stop_idx - 1], fpr[stop_idx]
        y1, y2 = tpr[stop_idx - 1], tpr[stop_idx]
        tpr_interp = y1 + (y2 - y1) * (max_fpr - x1) / (x2 - x1)

        fpr_sliced = np.append(fpr_sliced, max_fpr)
        tpr_sliced = np.append(tpr_sliced, tpr_interp)

    return auc(fpr_sliced, tpr_sliced)


def partial_auc_normalized(y_true, y_scores, max_fpr=0.1):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    stop_idx = np.searchsorted(fpr, max_fpr, side='right')

    fpr_sliced = fpr[:stop_idx].copy()
    tpr_sliced = tpr[:stop_idx].copy()

    if len(fpr_sliced) < len(fpr) and len(fpr_sliced) > 0 and fpr_sliced[-1] < max_fpr:
        x1, x2 = fpr[stop_idx - 1], fpr[stop_idx]
        y1, y2 = tpr[stop_idx - 1], tpr[stop_idx]
        tpr_interp = y1 + (y2 - y1) * (max_fpr - x1) / (x2 - x1)
        fpr_sliced = np.append(fpr_sliced, max_fpr)
        tpr_sliced = np.append(tpr_sliced, tpr_interp)

    if len(fpr_sliced) == 0:
        return np.nan
    return auc(fpr_sliced, tpr_sliced) / max_fpr


def interval_auc_normalized(pred: np.ndarray, start_idx: int, end_idx: int) -> float:
    if end_idx < start_idx:
        return float("nan")
    seg = pred[start_idx:end_idx + 1]
    if len(seg) == 0:
        return float("nan")
    if len(seg) == 1:
        return float(seg[0])
    return float(np.mean(seg))


def top_auc(y_true, y_scores, fpr_max=0.1):
    # 1. 按预测得分降序排序，并记录真实标签
    sorted_indices = np.argsort(y_scores)[::-1]  # 从高到低排序
    y_true_sorted = y_true[sorted_indices]

    # 2. 计算正负样本数量
    P = np.sum(y_true == 1)  # 正样本数
    N = np.sum(y_true == 0)  # 负样本数

    # 3. 初始化TPR和FPR
    TPR = [0]  # 真正例率（初始为0）
    FPR = [0]  # 假正例率（初始为0）
    TP, FP = 0, 0  # 累积真正例和假正例数

    # 4. 遍历排序后的样本，动态更新TPR和FPR
    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            TP += 1  # 真正例+1
        else:
            FP += 1  # 假正例+1
        TPR.append(TP / P)  # 计算当前TPR
        FPR.append(FP / N)  # 计算当前FPR

    # 5. 梯形法计算AUC（积分ROC曲线下面积）
    auc = 0
    for i in range(1, len(FPR)):
        if FPR[i] > fpr_max:
            break
        dx = FPR[i] - FPR[i - 1]  # x轴宽度
        dy = TPR[i] + TPR[i - 1]  # y轴平均高度
        auc += dx * dy / 2  # 梯形面积累加

    return auc / fpr_max, FPR, TPR


def _bin_segment(values, n_bins):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return np.zeros(n_bins, dtype=np.float32)
    if len(values) == 1:
        return np.full(n_bins, float(values[0]), dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, len(values), endpoint=True)
    x_new = np.linspace(0.0, 1.0, n_bins, endpoint=True)
    return np.interp(x_new, x_old, values).astype(np.float32)


def compute_interval_metrics(preds, labels, abnormal_start_inds, accident_inds):
    """
    Compute two custom metrics on positive sequences only:

    1. pre_anomaly_auc:
       normalized AUC from t0 to just before t_ai
       perfect model -> low

    2. anomaly_to_collision_auc:
       normalized AUC from t_ai to t_co
       perfect model -> high
    """
    pre_vals = []
    post_vals = []

    for pred, label, ai_idx, co_idx in zip(preds, labels, abnormal_start_inds, accident_inds):
        if not label:
            continue

        # pre-anomaly: [0, ai_idx-1]
        if ai_idx > 0:
            pre_auc = interval_auc_normalized(pred, 0, ai_idx) # -1
            if not np.isnan(pre_auc):
                pre_vals.append(pre_auc)

        # anomaly to collision: [ai_idx, co_idx]
        if co_idx >= ai_idx:
            post_auc = interval_auc_normalized(pred, ai_idx+1, co_idx)
            if not np.isnan(post_auc):
                post_vals.append(post_auc)

    results = {
        "pre_anomaly_auc_mean": float(np.mean(pre_vals)) if len(pre_vals) > 0 else float("nan"),
        "pre_anomaly_auc_std": float(np.std(pre_vals)) if len(pre_vals) > 0 else float("nan"),
        "anomaly_to_collision_auc_mean": float(np.mean(post_vals)) if len(post_vals) > 0 else float("nan"),
        "anomaly_to_collision_auc_std": float(np.std(post_vals)) if len(post_vals) > 0 else float("nan"),
        "num_positive_videos_for_pre": len(pre_vals),
        "num_positive_videos_for_post": len(post_vals),
    }

    # optional combined score, high is better
    if len(pre_vals) > 0 and len(post_vals) > 0:
        results["separation_score"] = float((1.0 - np.mean(pre_vals)) + np.mean(post_vals))
    else:
        results["separation_score"] = float("nan")

    return results


def pad_repeat_first(arr: np.ndarray, target_len: int) -> np.ndarray:
    if len(arr) >= target_len:
        return arr
    if len(arr) == 0:
        raise ValueError("Cannot pad an empty array.")
    pad_value = arr[0]
    pad = np.full(target_len - len(arr), pad_value, dtype=arr.dtype)
    return np.concatenate([pad, arr], axis=0)


def pad_repeat_last(arr: np.ndarray, target_len: int) -> np.ndarray:
    if len(arr) >= target_len:
        return arr
    if len(arr) == 0:
        raise ValueError("Cannot pad an empty array.")
    pad_value = arr[-1]
    pad = np.full(target_len - len(arr), pad_value, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)
