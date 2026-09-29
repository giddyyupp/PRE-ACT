from collections import defaultdict

from evaluation.metrics import (
    auc_lambda,
    mean_auc_lambda,
    binary_classification_metrics,
)


def collect_horizon_scores(eval_outputs):
    """
    eval_outputs: iterable of dicts like:
      {
        "horizon_sec": 0.5 or 1.0 or 1.5 or None,
        "label": 0/1,
        "score": float,
      }

    Returns:
      {
        0.5: (y_true, y_score),
        1.0: (y_true, y_score),
        1.5: (y_true, y_score),
      }
    """
    groups = defaultdict(lambda: {"y_true": [], "y_score": []})

    for x in eval_outputs:
        h = x["horizon_sec"]
        if h is None:
            # negatives can be shared into all horizon groups if you want balanced horizon-specific AUC
            continue
        groups[h]["y_true"].append(int(x["label"]))
        groups[h]["y_score"].append(float(x["score"]))

    return {
        h: (v["y_true"], v["y_score"])
        for h, v in groups.items()
    }


def collect_horizon_scores_with_shared_negatives(eval_outputs, horizons=(0.5, 1.0, 1.5), score_key='score'):
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
            y_score.append(float(p[score_key]))

        for n in negatives:
            y_true.append(0)
            y_score.append(float(n[score_key]))

        result[h] = (y_true, y_score)

    return result


def calculate_metrics_from_clip_outputs(clip_outputs, score_key='score'):

    metrics = {}
    y_true = [x["label"] for x in clip_outputs]
    y_score = [x[score_key] for x in clip_outputs]
    metrics = binary_classification_metrics(y_true, y_score, threshold=0.5)

    horizon_data = collect_horizon_scores_with_shared_negatives(
        clip_outputs,
        horizons=(0.5, 1.0, 1.5),
        score_key=score_key
    )
    for h, (yt, ys) in horizon_data.items():
        metrics[f'AUC_0.1@{h:.1f}s'] = auc_lambda(yt, ys, lambda_far=0.1)
        metrics[f'AUC@{h:.1f}s'] = auc_lambda(yt, ys, lambda_far=1.0)
        print(f"AUC_0.1 @ {h:.1f}s =", metrics[f'AUC_0.1@{h:.1f}s'])
        print(f"AUC @ {h:.1f}s =", metrics[f'AUC@{h:.1f}s'])

    metrics['mAUC_0.1'] = mean_auc_lambda(horizon_data, lambda_far=0.1)
    metrics['mAUC'] = mean_auc_lambda(horizon_data, lambda_far=1.0)
    # print("mTTA_0.1 =", compute_mtta_lambda(video_predictions, fps=10, lambda_far=0.1))
    print("mAUC_0.1 =", metrics['mAUC_0.1'])
    print("mAUC =", metrics['mAUC'])  
    print(metrics)
    return metrics

