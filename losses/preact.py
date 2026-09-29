import torch
import torch.nn.functional as F

from engine.risk_targets import build_ttc_ordinal_targets


def compute_losses(model, batch, lambda_bce=1.0, lambda_prog=1.0, lambda_pref=0.2, margin=0.05, pref_loss_type="logsigmoid"):
    out = model(batch["frames"])
    risk_logit = out.get("risk_logit")

    if risk_logit is not None:
        risk_prob = torch.sigmoid(risk_logit)
    else:
        risk_prob = torch.zeros(batch["frames"].shape[0], device=batch["frames"].device)
    
    prog_score = out.get("progress_logit")
    pref_score = out.get("preference_logit")

    # models without a dedicated preference head fall back to the progress head
    if pref_score is None:
        pref_score = prog_score

    has_prog_head = prog_score is not None
    has_pref_head = pref_score is not None

    if prog_score is None:
        prog_score = risk_prob.new_tensor(0.0)

    if pref_score is None:
        pref_score = risk_prob.new_tensor(0.0)

    # if 'preference_logit' in out:
    #     pref_score = out["preference_logit"]
    # else:
    #     pref_score = risk_prob.new_tensor(0.0)
    
    if risk_logit is not None and lambda_bce > 0:
        loss_bce = F.binary_cross_entropy_with_logits(
                risk_logit,
                batch["binary_target"],
            )
    else:
        loss_bce = risk_prob.new_tensor(0.0)    

    valid_progress = batch["valid_progress"] > 0
    if has_prog_head and valid_progress.any() and lambda_prog > 0:
        loss_prog = F.smooth_l1_loss(
            prog_score[valid_progress],
            batch["risk_target"][valid_progress],
        )
    else:
        loss_prog = risk_prob.new_tensor(0.0)

    # preference loss only on items with valid future clips
    has_future = batch["has_future"].to(risk_prob.device)
    pref_valid = (batch["pref_valid"] > 0).to(risk_prob.device)
    pref_valid_risk = (batch["future_risk_target"] > 0).to(risk_prob.device)  # NOTE: only consider items where the future clip has a risk event, to avoid noisy labels
    valid_pref = has_future & pref_valid & pref_valid_risk

    if has_pref_head and valid_pref.any() and batch["future_frames"] is not None and lambda_pref > 0:
        future_out = model(batch["future_frames"])

        pref_score_future = future_out.get("preference_logit")
        if pref_score_future is None:
            pref_score_future = future_out["progress_logit"]

        if pref_loss_type == "logsigmoid":
            loss_pref = -torch.log(torch.sigmoid(pref_score_future[valid_pref] - pref_score[valid_pref]) + 1e-8).mean()
        elif pref_loss_type == "margin_ranking":
            loss_pref = F.margin_ranking_loss(
                pref_score_future[valid_pref],
                pref_score[valid_pref],
                torch.ones_like(pref_score_future[valid_pref]),
                margin=margin,
            )
        else:
            raise ValueError(f"Invalid pref_loss_type: {pref_loss_type}")

        # loss_pref = -F.logsigmoid(prog_score_future[valid_pref] - prog_score[valid_pref]).mean()
    else:
        loss_pref = risk_prob.new_tensor(0.0)

    loss = lambda_bce * loss_bce + lambda_prog * loss_prog + lambda_pref * loss_pref

    return {
        "loss": loss,
        "loss_bce": loss_bce.detach(),
        "loss_prog": loss_prog.detach(),
        "loss_pref": loss_pref.detach(),
        "risk_prob": risk_prob.detach(),
        "prog_score": prog_score.detach(), 
        "pref_score": pref_score.detach(), 
    }


def compute_losses_ordinal_ttc(
    model,
    batch,
    lambda_bce=1.0,
    lambda_ttc=0.5,
):
    out = model(batch["frames"])

    risk_logit = out["risk_logit"] # [B]
    risk_prob = torch.sigmoid(risk_logit)
    ttc_ord_logits = out["progress_logit"] # [B, 4]
  
    # -------------------------
    # BCE detection loss
    # -------------------------
    loss_bce = F.binary_cross_entropy_with_logits(
        risk_logit,
        batch["binary_target"],
    )

    # -------------------------
    # TTC ordinal loss
    # -------------------------
    ttc_frames = batch["ttc_frames"]           # [B]
    fps = batch["fps"]                         # [B]
    ttc_sec = torch.where(
        ttc_frames >= 0,
        ttc_frames / torch.clamp(fps, min=1.0),
        torch.full_like(ttc_frames, 999.0),    # non-accident / invalid -> >2s bin
    )

    ttc_targets = build_ttc_ordinal_targets(ttc_sec)   # [B, 4]

    loss_ttc = F.binary_cross_entropy_with_logits(
        ttc_ord_logits,
        ttc_targets.to(risk_prob.device),
    )

    loss = lambda_bce * loss_bce + lambda_ttc * loss_ttc

    loss_pref = risk_prob.new_tensor(0.0)

    return {
        "loss": loss,
        "loss_bce": loss_bce.detach(),
        "loss_prog": loss_ttc.detach(),
        "loss_pref": loss_pref.detach(),
        "risk_prob": risk_prob.detach(),
        "prog_score": torch.sigmoid(ttc_ord_logits).detach(),
    }
