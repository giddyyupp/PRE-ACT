import math
from functools import partial
from typing import Callable

import torch


def build_ttc_ordinal_targets(ttc_sec: torch.Tensor) -> torch.Tensor:
    """
    ttc_sec: [B]
    returns ordinal targets [B, 4]
    thresholds:
      <= 2.0, <= 1.5, <= 1.0, <= 0.5
    """
    thresholds = torch.tensor([2.0, 1.5, 1.0, 0.5], device=ttc_sec.device)
    return (ttc_sec.unsqueeze(1) <= thresholds.unsqueeze(0)).float()


def exp_under_linear(ttc_frames_10fps: float, alpha: float = 3.0) -> float:
    if ttc_frames_10fps <= 0:
        return 1.0
    if ttc_frames_10fps >= 20:
        return 0.0

    u = 1.0 - ttc_frames_10fps / 20.0  # progress: 0 far, 1 crash

    return (math.exp(alpha * u) - 1.0) / (math.exp(alpha) - 1.0)


def exp_above_linear(ttc_frames_10fps: float, alpha: float = 3.0) -> float:
    if ttc_frames_10fps <= 0:
        return 1.0
    if ttc_frames_10fps >= 20:
        return 0.0

    u = 1.0 - ttc_frames_10fps / 20.0
    return (1.0 - math.exp(-alpha * u)) / (1.0 - math.exp(-alpha))


def piecewise_risk_target_custom_strong(ttc_frames_10fps: float) -> float:
    """
    Risk target as a function of time-to-collision in frames.

    ttc_frames:
        20  -> far from accident
        0   -> collision now

    Piecewise design:
        20..10 : 0.0 -> 0.4
        10..5  : 0.4 -> 0.9
        5..0   : 0.9 -> 1.0
    """
    if ttc_frames_10fps <= 0:
        return 1.0
    if ttc_frames_10fps >= 20:
        return 0.0

    # Stage 1: 20 -> 10 frames, 0.0 -> 0.4
    if ttc_frames_10fps > 10:
        progress = (20 - ttc_frames_10fps) / 10.0   # 0 at 20, 1 at 10
        return 0.0 + 0.4 * progress

    # Stage 2: 10 -> 5 frames, 0.4 -> 0.9
    if ttc_frames_10fps > 5:
        progress = (10 - ttc_frames_10fps) / 5.0    # 0 at 10, 1 at 5
        return 0.4 + 0.5 * progress

    # Stage 3: 5 -> 0 frames, 0.9 -> 1.0
    progress = (5 - ttc_frames_10fps) / 5.0         # 0 at 5, 1 at 0
    return 0.9 + 0.1 * progress


def piecewise_risk_target_custom_mild(ttc_frames_10fps: float) -> float:
    """
    Same target, but input is TTC in 10-FPS-equivalent frames.

    Key points:
        20 frames -> 0.0
        10 frames -> 0.2
        5 frames  -> 0.6
        0 frames  -> 1.0
    """
    if ttc_frames_10fps <= 0.0:
        return 1.0
    if ttc_frames_10fps >= 20.0:
        return 0.0

    if ttc_frames_10fps > 10.0:
        progress = (20.0 - ttc_frames_10fps) / 10.0
        return 0.0 + 0.2 * progress

    if ttc_frames_10fps > 5.0:
        progress = (10.0 - ttc_frames_10fps) / 5.0
        return 0.2 + 0.4 * progress

    progress = (5.0 - ttc_frames_10fps) / 5.0
    return 0.6 + 0.4 * progress


def piecewise_risk_target_custom_milder(ttc_frames_10fps: float) -> float:
    """
    Milder piecewise risk target in 10-FPS-equivalent frames.

    Key points:
        20 frames -> 0.0
        10 frames -> 0.12
        5 frames  -> 0.35
        0 frames  -> 1.0
    """
    if ttc_frames_10fps <= 0.0:
        return 1.0
    if ttc_frames_10fps >= 20.0:
        return 0.0

    # Stage 1: 20 -> 10 frames, 0.0 -> 0.12
    if ttc_frames_10fps > 10.0:
        progress = (20.0 - ttc_frames_10fps) / 10.0
        return 0.12 * progress

    # Stage 2: 10 -> 5 frames, 0.12 -> 0.35
    if ttc_frames_10fps > 5.0:
        progress = (10.0 - ttc_frames_10fps) / 5.0
        return 0.12 + (0.35 - 0.12) * progress

    # Stage 3: 5 -> 0 frames, 0.35 -> 1.0
    progress = (5.0 - ttc_frames_10fps) / 5.0
    return 0.35 + (1.0 - 0.35) * progress


def piecewise_risk_target_custom_linear(ttc_frames_10fps: float) -> float:
    """
    Linear risk target in 10-FPS-equivalent frames.

    Key points:
        20 frames -> 0.0
        0 frames  -> 1.0

    So:
        risk = 1 - ttc/20, clipped to [0, 1]
    """
    if ttc_frames_10fps <= 0.0:
        return 1.0
    if ttc_frames_10fps >= 20.0:
        return 0.0

    return 1.0 - (ttc_frames_10fps / 20.0)


def get_risk_target_fn(cfg) -> Callable[[float], float]:
    """
    Returns a function that maps TTC in 10-FPS-equivalent frames -> risk target.
    """
    mode = getattr(cfg, "custom_risk_mode", None)

    if mode == "mild":
        print('Using MILD risk function!')
        return piecewise_risk_target_custom_mild
    if mode == "milder":
        print('Using MILDER risk function!')
        return piecewise_risk_target_custom_milder
    if mode == "strong":
        print('Using STRONG risk function!')
        return piecewise_risk_target_custom_strong
    if mode == "linear":
        print('Using LINEAR risk function!')
        return piecewise_risk_target_custom_linear
    if mode == "exp_above":
        print('Using EXPONENTIAL ABOVE LINEAR risk function!')
        return partial(exp_above_linear, alpha=cfg.progress_alpha)
    if mode == "exp_under":
        print('Using EXPONENTIAL UNDER LINEAR risk function!')
        return partial(exp_under_linear, alpha=cfg.progress_alpha)

    print(f'Using EXPONENTIAL risk function with alpha = {cfg.progress_alpha}!')

    # default exponential below linear fallback
    return lambda ttc_frames_10fps: float(
        math.exp(-cfg.progress_alpha * ttc_frames_10fps)
    )


def full_video_progress_risk_fn(
    current_frame_idx_1based: int,
    t_co: int,
    first_valid_frame_1based: int,
    mode: str = "linear",
    alpha: float = 3.0,
) -> float:
    """
    Risk from the beginning of the usable video until accident.

    first_valid_frame_1based: first clip endpoint, usually first_valid_end_idx + 1
    t_co: collision frame
    current_frame_idx_1based: current clip endpoint

    Returns:
        0 at beginning
        1 at collision
    """
    if t_co <= first_valid_frame_1based:
        return 1.0

    if current_frame_idx_1based >= t_co:
        return 1.0

    if current_frame_idx_1based <= first_valid_frame_1based:
        return 0.0

    u = (current_frame_idx_1based - first_valid_frame_1based) / (
        t_co - first_valid_frame_1based
    )
    u = max(0.0, min(1.0, u))

    if mode == "linear":
        return u

    if mode == "exp_above":
        return (1.0 - math.exp(-alpha * u)) / (1.0 - math.exp(-alpha))

    if mode == "exp_under":
        return (math.exp(alpha * u) - 1.0) / (math.exp(alpha) - 1.0)

    raise ValueError(f"Unknown full-video risk mode: {mode}")

