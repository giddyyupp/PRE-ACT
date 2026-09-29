import numpy as np
import random
import math
from loguru import logger as logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, average_precision_score, roc_curve, auc
)

def raw_partial_auc(y_true, y_scores, max_fpr):
    """Calculates the exact unnormalized partial AUC."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    stop_idx = np.searchsorted(fpr, max_fpr, side='right')
    fpr_sliced = fpr[:stop_idx].copy()
    tpr_sliced = tpr[:stop_idx].copy()
    
    if len(fpr_sliced) < len(fpr) and fpr_sliced[-1] < max_fpr:
        x1, x2 = fpr[stop_idx - 1], fpr[stop_idx]
        y1, y2 = tpr[stop_idx - 1], tpr[stop_idx]
        tpr_interp = y1 + (y2 - y1) * (max_fpr - x1) / (x2 - x1)
        fpr_sliced = np.append(fpr_sliced, max_fpr)
        tpr_sliced = np.append(tpr_sliced, tpr_interp)
        
    return auc(fpr_sliced, tpr_sliced) / max_fpr

def evaluate_predictions(global_probs_np, global_labels_np, global_clip_names, anno_dict):
    """
    Runs the full evaluation suite.
    """
    # =========================================================
    # 1. EVALUATION ON ALL CLIPS
    # =========================================================
    logging.info("="*50)
    logging.info(" 1. EVALUATION: ALL CLIPS")
    logging.info("="*50)
    
    logging.info(f"Total evaluated samples: {len(global_labels_np)}")
    
    y_pred = (global_probs_np >= 0.5).astype(int)

    logging.info(f"Accuracy:  {accuracy_score(global_labels_np, y_pred):.4f}")
    logging.info(f"Precision: {precision_score(global_labels_np, y_pred, zero_division=0):.4f}")
    logging.info(f"Recall:    {recall_score(global_labels_np, y_pred, zero_division=0):.4f}")
    logging.info(f"F1 Score:  {f1_score(global_labels_np, y_pred, zero_division=0):.4f}")
    logging.info(f"ROC AUC:   {roc_auc_score(global_labels_np, global_probs_np):.4f}")
    logging.info(f"AP:        {average_precision_score(global_labels_np, global_probs_np):.4f}")
    logging.info(f"Confusion Matrix:\n{confusion_matrix(global_labels_np, y_pred)}")

    fpr_thresholds = [0.01, 0.1, 1.0]
    for th in fpr_thresholds:
        if th == 1.0:
            auc_th = roc_auc_score(global_labels_np, global_probs_np)
            logging.info(f"AUC @ FPR 1.0 : {auc_th:.4f}")
        else:
            raw_pauc = raw_partial_auc(global_labels_np, global_probs_np, max_fpr=th)
            try:
                sk_pauc = roc_auc_score(global_labels_np, global_probs_np, max_fpr=th)
            except ValueError:
                sk_pauc = float('nan')
            logging.info(f"AUC @ FPR {th:<4}: Raw={raw_pauc:.4f} | Sklearn={sk_pauc:.4f}")

    # =========================================================
    # 2. EVALUATION ON STRUCTURED/ANNOTATED SET
    # =========================================================
    logging.info("\n" + "="*50)
    logging.info(" 2. EVALUATION: ANNOTATED CLIPS (AUC/TTA)")
    logging.info("="*50)
    
    raw_video_groups = {}
    for i, clip_name in enumerate(global_clip_names):
        if i >= len(global_probs_np):
            break 
            
        vid_id = clip_name.split("_")[0] 
        clip_idx = int(clip_name.split("_")[1])
        score = float(global_probs_np[i])
        label = int(global_labels_np[i])
        
        clip_info = {'clip_name': clip_name, 'vid_id': vid_id, 'idx': clip_idx, 'score': score, 'label': label}
        
        if vid_id not in raw_video_groups:
            raw_video_groups[vid_id] = []
        raw_video_groups[vid_id].append(clip_info)

    video_groups = {}
    all_filtered_clips = []
    
    valid_videos = set()
    
    for vid_id, clips in raw_video_groups.items():
        if vid_id not in anno_dict:
            logging.warning(f"Video {vid_id} missing from JSON annotations. Skipping.")
            continue
            
        vid_info = anno_dict[vid_id]
        fps = vid_info["fps"]
        accident_frame = vid_info["accident_frame"]
        
        if fps == 20: step = 2
        elif fps == 30: step = 3
        else: step = 1
            
        current_fps = 10
        step_x = step * (10 // current_fps)
        
        clips = sorted(clips, key=lambda x: x['idx'])
        
        valid_videos.add(vid_id)
        
        # Calculate precise TTA for all clips
        for c in clips:
            state_x_end = c['idx'] + (4 * step_x)
            c['tta'] = (accident_frame - state_x_end) / fps

        for c in clips:
            if vid_id not in video_groups:
                video_groups[vid_id] = []
            video_groups[vid_id].append(c)
            all_filtered_clips.append(c)

    if len(all_filtered_clips) > 0:
        y_true_f = np.array([c['label'] for c in all_filtered_clips])
        y_scores_f = np.array([c['score'] for c in all_filtered_clips])
        y_pred_f = (y_scores_f >= 0.5).astype(int)

        logging.info(f"Total Clips Evaluated in this stage: {len(all_filtered_clips)}")
        logging.info(f"Accuracy:  {accuracy_score(y_true_f, y_pred_f):.4f}")
        logging.info(f"ROC AUC:   {roc_auc_score(y_true_f, y_scores_f):.4f}")
        logging.info(f"AP:        {average_precision_score(y_true_f, y_scores_f):.4f}")

        # =========================================================
        # PRIOR SCRIPT METHOD: AUC AT SPECIFIC TIME HORIZONS (Window Max)
        # =========================================================
        logging.info("\n--- AUC AT SPECIFIC TIME HORIZONS (Max-Pooling Window) ---")
        
        score_n_window = []
        
        for vid in valid_videos:
            if vid not in video_groups: continue
            clips = video_groups[vid]
            fps = anno_dict[vid]["fps"]
            
            pos_clips = [c for c in clips if c['label'] == 1]
            if not pos_clips: continue
            first_pos_idx = min(c['idx'] for c in pos_clips)
            
            window_frames = 0.5 * fps
            
            neg_window_scores = [
                c['score'] for c in clips 
                if c['label'] == 0 and 0 < (first_pos_idx - c['idx']) <= window_frames
            ]
            
            if neg_window_scores:
                score_n_window.append(max(neg_window_scores))
            else:
                all_prev_negs = [c['score'] for c in clips if c['label'] == 0 and c['idx'] < first_pos_idx]
                if all_prev_negs:
                    score_n_window.append(max(all_prev_negs))
                else:
                    score_n_window.append(0.0)
            
        target_ttas_window = [0.0, 0.5, 1.0, 1.5] 
        
        for t_target in target_ttas_window:
            score_pos_window = []
            
            for vid in valid_videos:
                if vid not in video_groups: continue
                clips = video_groups[vid]
                
                window_clips = [c for c in clips if c['label'] == 1 and t_target <= c['tta'] < (t_target + 0.5)]
                if window_clips:
                    score_pos_window.append(max(c['score'] for c in window_clips))
            
            if not score_pos_window or not score_n_window:
                logging.info(f"Time {t_target}s: Not enough samples for window comparison.")
                continue
                
            y_true_t = np.array([0] * len(score_n_window) + [1] * len(score_pos_window))
            y_scores_t = np.array(score_n_window + score_pos_window)
            
            logging.info(f"Target Time Window {t_target}s - {t_target+0.5}s ({len(score_pos_window)} pos vs {len(score_n_window)} baseline negs):")
            
            for th in fpr_thresholds:
                try:
                    if th == 1.0:
                        auc_t_th = roc_auc_score(y_true_t, y_scores_t)
                        logging.info(f"  AUC @ FPR 1.0 : {auc_t_th:.4f}")
                    else:
                        raw_pauc = raw_partial_auc(y_true_t, y_scores_t, max_fpr=th)
                        sk_pauc = roc_auc_score(y_true_t, y_scores_t, max_fpr=th)
                        logging.info(f"  AUC @ FPR {th:<4}: Raw={raw_pauc:.4f} | Sklearn={sk_pauc:.4f}")
                except ValueError:
                    logging.info(f"  AUC @ FPR {th:<4}: N/A (Not enough variability)")

        # =========================================================
        # NEW METHOD: AUC AT EXACT TIME HORIZONS (Exact Clips)
        # =========================================================
        logging.info("\n--- AUC AT EXACT TIME HORIZONS (Closest Clip Match) ---")
        
        score_n_exact = []
        
        # 1. Gather negative samples: Random negative clip before 2.0s TTA (TTA > 2.0)
        for vid in valid_videos:
            if vid not in video_groups: continue
            clips = video_groups[vid]
            
            valid_negs = [c for c in clips if c['label'] == 0 and c['tta'] > 2.0]
            if valid_negs:
                chosen_neg = random.choice(valid_negs)
                score_n_exact.append(chosen_neg['score'])

        exact_targets = [0.5, 1.0, 1.5, 2.0]
        
        for t_target in exact_targets:
            score_pos_exact = []
            
            for vid in valid_videos:
                if vid not in video_groups: continue
                clips = video_groups[vid]
                
                pos_clips = [c for c in clips if c['label'] == 1]
                if not pos_clips: continue
                
                # Find the clip closest to the exact target TTA
                closest_clip = min(pos_clips, key=lambda c: abs(c['tta'] - t_target))
                
                # Use a small tolerance (e.g., 0.15s) in case clips don't perfectly land on 0.5 boundaries
                if abs(closest_clip['tta'] - t_target) <= 0.15:
                    score_pos_exact.append(closest_clip['score'])
            
            if not score_pos_exact or not score_n_exact:
                logging.info(f"Exact Time {t_target}s: Not enough samples for exact comparison.")
                continue
                
            y_true_e = np.array([0] * len(score_n_exact) + [1] * len(score_pos_exact))
            y_scores_e = np.array(score_n_exact + score_pos_exact)
            
            logging.info(f"Exact Target Time {t_target}s ({len(score_pos_exact)} pos vs {len(score_n_exact)} baseline negs):")
            
            for th in fpr_thresholds:
                try:
                    if th == 1.0:
                        auc_e_th = roc_auc_score(y_true_e, y_scores_e)
                        logging.info(f"  AUC @ FPR 1.0 : {auc_e_th:.4f}")
                    else:
                        raw_pauc_e = raw_partial_auc(y_true_e, y_scores_e, max_fpr=th)
                        sk_pauc_e = roc_auc_score(y_true_e, y_scores_e, max_fpr=th)
                        logging.info(f"  AUC @ FPR {th:<4}: Raw={raw_pauc_e:.4f} | Sklearn={sk_pauc_e:.4f}")
                except ValueError:
                    logging.info(f"  AUC @ FPR {th:<4}: N/A (Not enough variability)")

        # =========================================================
        # PRIOR SCRIPT METHOD: mTTA (Mean of Means across Thresholds)
        # =========================================================
        logging.info("\n--- MEAN TIME-TO-ACCIDENT (mTTA via Negative Thresholding) ---")
        
        all_neg_scores = np.array([c['score'] for c in all_filtered_clips if c['label'] == 0])
        
        if len(all_neg_scores) == 0 or len(valid_videos) == 0:
            logging.warning("Cannot calculate TTA: Missing positive or negative samples.")
        else:
            thresholds = np.sort(all_neg_scores)[::-1]
            tta_means_per_threshold = []
            far_means_per_threshold = []
            
            for th in thresholds:
                far = np.mean(all_neg_scores >= th)
                far_means_per_threshold.append(far)
                
                ttas_th = []
                for vid in valid_videos:
                    if vid not in video_groups: continue
                    clips = sorted(video_groups[vid], key=lambda x: x['idx'])
                    tta_val = 0.0
                    for c in clips:
                        if c['label'] == 1 and c['score'] >= th:
                            tta_val = c['tta']
                            break
                    ttas_th.append(tta_val)
                    
                tta_means_per_threshold.append(np.mean(ttas_th) if ttas_th else 0.0)
                
            tta_means_per_threshold = np.array(tta_means_per_threshold)
            far_means_per_threshold = np.array(far_means_per_threshold)
            
            for th in fpr_thresholds:
                valid_mask = far_means_per_threshold <= th
                if np.any(valid_mask):
                    m_tta = float(tta_means_per_threshold[valid_mask].mean())
                else:
                    m_tta = 0.0
                logging.info(f"mTTA @ FPR <= {th:<4}: {m_tta:.4f}s")
                
    else:
        logging.warning("No clips remained after grouping by video.")


def fuse_scores_adaptive(score, risk_score, k=8.0, b=0.10):
    risk_prob = 1.0 / (1.0 + np.exp(-k * (risk_score - b)))

    # use risk itself as weight
    lam = risk_prob  # dynamic!

    fused = (1.0 - lam) * score + lam * risk_prob
    return fused



def fuse_scores_residual(score, risk_score, k=8.0, b=0.10, lam=0.2):
    risk_prob = 1.0 / (1.0 + np.exp(-k * (risk_score - b)))

    # only use deviation from neutral
    delta = risk_prob - 0.5

    return score + lam * delta


def fuse_scores_boost(score, risk_score, k=8.0, b=0.10, lam=0.5):
    risk_prob = 1.0 / (1.0 + np.exp(-k * (risk_score - b)))

    # ONLY boost, never penalize
    boost = np.maximum(0, risk_prob - 0.5)

    return score + lam * boost


def fuse_scores_max(score, risk_score, k=8.0, b=0.10):
    risk_prob = 1.0 / (1.0 + np.exp(-k * (risk_score - b)))
    return np.maximum(score, risk_prob)


def _fmt_metric(metrics, key, digits=4):
    value = metrics.get(key, None)
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def print_validation_metrics(metrics, epoch, num_epochs, global_step, score_key):
    """Compact, readable validation summary for SLURM/stdout logs."""
    bar = "=" * 112
    print("\\n" + bar)
    print(
        f"[VAL] epoch {epoch + 1}/{num_epochs} complete | "
        f"global_step {global_step} | score={score_key}"
    )
    print("-" * 112)

    print(
        "pAUC@0.10 : "
        f"0.0s={_fmt_metric(metrics, 'AUC@0.0s')}  "
        f"0.5s={_fmt_metric(metrics, 'AUC@0.5s')}  "
        f"1.0s={_fmt_metric(metrics, 'AUC@1.0s')}  "
        f"1.5s={_fmt_metric(metrics, 'AUC@1.5s')}  |  "
        f"mAUC@0.1={_fmt_metric(metrics, 'mAUC@0.1')}"
    )

    print(
        "pAUC@0.01 : "
        f"0.0s={_fmt_metric(metrics, 'AUC_0.01@0.0s')}  "
        f"0.5s={_fmt_metric(metrics, 'AUC_0.01@0.5s')}  "
        f"1.0s={_fmt_metric(metrics, 'AUC_0.01@1.0s')}  "
        f"1.5s={_fmt_metric(metrics, 'AUC_0.01@1.5s')}  |  "
        f"mAUC@0.01={_fmt_metric(metrics, 'mAUC@0.01')}"
    )

    print(
        "Full AUC   : "
        f"0.0s={_fmt_metric(metrics, 'AUC_full@0.0s')}  "
        f"0.5s={_fmt_metric(metrics, 'AUC_full@0.5s')}  "
        f"1.0s={_fmt_metric(metrics, 'AUC_full@1.0s')}  "
        f"1.5s={_fmt_metric(metrics, 'AUC_full@1.5s')}  |  "
        f"mAUC={_fmt_metric(metrics, 'mAUC')}"
    )

    print(
        "AP         : "
        f"0.0s={_fmt_metric(metrics, 'AP@0.0s')}  "
        f"0.5s={_fmt_metric(metrics, 'AP@0.5s')}  "
        f"1.0s={_fmt_metric(metrics, 'AP@1.0s')}  "
        f"1.5s={_fmt_metric(metrics, 'AP@1.5s')}  |  "
        f"mAP={_fmt_metric(metrics, 'mAP')}"
    )

    print(
        "TTA        : "
        f"@0.01={_fmt_metric(metrics, 'tta@0.01')}s  "
        f"@0.05={_fmt_metric(metrics, 'tta@0.05')}s  "
        f"@0.10={_fmt_metric(metrics, 'tta@0.1')}s  "
        f"mTTA@0.1={_fmt_metric(metrics, 'mtta@0.1')}s  "
        f"@1={_fmt_metric(metrics, 'tta@1')}s"
    )

    if (
        "pre_anomaly_auc_mean" in metrics
        or "anomaly_to_collision_auc_mean" in metrics
        or "separation_score" in metrics
    ):
        print(
            "Intervals  : "
            f"pre={_fmt_metric(metrics, 'pre_anomaly_auc_mean')}"
            f"±{_fmt_metric(metrics, 'pre_anomaly_auc_std')}  "
            f"post={_fmt_metric(metrics, 'anomaly_to_collision_auc_mean')}"
            f"±{_fmt_metric(metrics, 'anomaly_to_collision_auc_std')}  |  "
            f"separation={_fmt_metric(metrics, 'separation_score')}"
        )

    print(
        "Other      : "
        f"threshold@0.10={_fmt_metric(metrics, 'threshold@0.10')}  "
        f"num_samples={metrics.get('num_samples', 'n/a')}"
    )
    print(bar, flush=True)
