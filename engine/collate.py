import torch


def expand_video_by_index_repeat(video: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    video: [T, C, H, W]

    Expands or shrinks a clip to target_len using uniform index sampling.
    For T < target_len, this repeats frames approximately uniformly.

    Example:
      T=5, target_len=16
      -> indices [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4]
    """
    if video.ndim != 4:
        raise ValueError(f"Expected [T, C, H, W], got {video.shape}")

    T = video.shape[0]

    if T == target_len:
        return video

    idx = torch.floor(torch.arange(target_len, device=video.device) * T / target_len).long()
    idx = torch.clamp(idx, max=T - 1)
    return video[idx]


def anticipation_eval_collate_fn_pad(batch, backbone_type: str = "dinov2"):
    if backbone_type == "videomae":
        target_len = 16
    elif backbone_type == "vjepa2":
        target_len = 16
    elif backbone_type == "xclip":
        target_len = 8
    elif backbone_type == "cosmos":
        target_len = 8
    elif backbone_type == "x3d":
        target_len = 16
    else:
        target_len = None

    frames = []
    meta = []

    for item in batch:
        imgs = item["images"]
        x = torch.stack(imgs, dim=0) if isinstance(imgs, list) else imgs
        if target_len is not None:
            x = expand_video_by_index_repeat(x, target_len=target_len)
        frames.append(x)
        meta.append(item)

    frames = torch.stack(frames, dim=0)
    return {
        "frames": frames,
        "meta": meta,
    }


def anticipation_collate_fn_pad(batch, backbone_type: str = "dinov2"):
    """
    batch items expected to contain:
      - frames: [T, C, H, W]
      - future_frames: [T, C, H, W] or None
    """

    if backbone_type == "videomae":
        target_len = 16
    elif backbone_type == "vjepa2":
        target_len = 16
    elif backbone_type == "xclip":
        target_len = 8
    elif backbone_type == "cosmos":
        target_len = 8
    elif backbone_type == "x3d":
        target_len = 16
    else:
        target_len = None  # keep original length, e.g. DINOv2 with 5 frames

    frame_list = []
    future_frame_list = []
    has_future = []

    for x in batch:
        frames = x["frames"]
        if target_len is not None:
            frames = expand_video_by_index_repeat(frames, target_len=target_len)
        frame_list.append(frames)

        if x["future_frames"] is not None:
            future_frames = x["future_frames"]
            if target_len is not None:
                future_frames = expand_video_by_index_repeat(future_frames, target_len=target_len)
            has_future.append(True)
        else:
            future_frames = frames.clone()
            has_future.append(False)

        future_frame_list.append(future_frames)

    frames = torch.stack(frame_list, dim=0)              # [B, T, C, H, W]
    future_frames = torch.stack(future_frame_list, dim=0)

    return {
        "frames": frames,
        "future_frames": future_frames,
        "has_future": torch.tensor(has_future, dtype=torch.bool),
        "binary_target": torch.stack([x["binary_target"] for x in batch], dim=0),
        "risk_target": torch.stack([x["risk_target"] for x in batch], dim=0),
        "future_risk_target": torch.stack([x["future_risk_target"] for x in batch], dim=0),
        "valid_progress": torch.stack([x["valid_progress"] for x in batch], dim=0),
        "pref_valid": torch.stack([x["pref_valid"] for x in batch], dim=0),
        "video_hashcode": [x["video_hashcode"] for x in batch],
        "current_frame_idx_1based": [x["current_frame_idx_1based"] for x in batch],
        "t_ai": [x["t_ai"] for x in batch],
        "t_co": [x["t_co"] for x in batch],
        "ttc_frames": torch.stack([x["ttc_frames"] for x in batch], dim=0),
        "fps": torch.stack([x["fps"] for x in batch], dim=0),
    }


def pad_or_trim_frames(frame_paths, target_len=8):
    """
    If fewer than target_len frames, repeat the last frame.
    If more than target_len frames, keep the last target_len frames.
    """
    if len(frame_paths) == target_len:
        return frame_paths

    if len(frame_paths) > target_len:
        return frame_paths[-target_len:]

    if len(frame_paths) == 0:
        raise ValueError("frame_paths is empty")

    pad_count = target_len - len(frame_paths)
    return frame_paths + [frame_paths[-1]] * pad_count


def pad_or_trim_video_tensor(video: torch.Tensor, target_len: int = 8) -> torch.Tensor:
    """
    video: [T, C, H, W]
    If T < target_len, repeat the last frame.
    If T > target_len, keep the last target_len frames.
    """
    if video.ndim != 4:
        raise ValueError(f"Expected video tensor of shape [T, C, H, W], got {video.shape}")

    T = video.shape[0]

    if T == target_len:
        return video

    if T > target_len:
        return video[-target_len:]

    pad_count = target_len - T
    last = video[-1:].repeat(pad_count, 1, 1, 1)
    return torch.cat([video, last], dim=0)


def anticipation_collate_fn_videomae(batch, target_len: int = 16):
    frame_list = []
    future_frame_list = []
    has_future = []

    for x in batch:
        frames = pad_or_trim_video_tensor(x["frames"], target_len=target_len)
        frame_list.append(frames)

        if x["future_frames"] is not None:
            future_frames = pad_or_trim_video_tensor(x["future_frames"], target_len=target_len)
            has_future.append(True)
        else:
            future_frames = frames.clone()
            has_future.append(False)

        future_frame_list.append(future_frames)

    frames = torch.stack(frame_list, dim=0)              # [B, 16, C, H, W]
    future_frames = torch.stack(future_frame_list, dim=0)

    return {
        "frames": frames,
        "future_frames": future_frames,
        "has_future": torch.tensor(has_future, dtype=torch.bool),
        "binary_target": torch.stack([x["binary_target"] for x in batch], dim=0),
        "risk_target": torch.stack([x["risk_target"] for x in batch], dim=0),
        "valid_progress": torch.stack([x["valid_progress"] for x in batch], dim=0),
        "pref_valid": torch.stack([x["pref_valid"] for x in batch], dim=0),
        "video_hashcode": [x["video_hashcode"] for x in batch],
        "current_frame_idx_1based": [x["current_frame_idx_1based"] for x in batch],
        "t_ai": [x["t_ai"] for x in batch],
        "t_co": [x["t_co"] for x in batch],
    }


def anticipation_collate_fn_xclip(batch, target_len: int = 8):
    # TODO: unused remove later
    frame_list = []
    future_frame_list = []
    has_future = []

    for x in batch:
        frames = pad_or_trim_video_tensor(x["frames"], target_len=target_len)
        frame_list.append(frames)

        if x["future_frames"] is not None:
            future_frames = pad_or_trim_video_tensor(x["future_frames"], target_len=target_len)
            has_future.append(True)
        else:
            # placeholder: use current clip, mask out later with pref_valid / has_future
            future_frames = frames.clone()
            has_future.append(False)

        future_frame_list.append(future_frames)

    frames = torch.stack(frame_list, dim=0)              # [B, T, C, H, W]
    future_frames = torch.stack(future_frame_list, dim=0)

    return {
        "frames": frames,
        "future_frames": future_frames,
        "has_future": torch.tensor(has_future, dtype=torch.bool),
        "binary_target": torch.stack([x["binary_target"] for x in batch], dim=0),
        "risk_target": torch.stack([x["risk_target"] for x in batch], dim=0),
        "valid_progress": torch.stack([x["valid_progress"] for x in batch], dim=0),
        "pref_valid": torch.stack([x["pref_valid"] for x in batch], dim=0),
        "video_hashcode": [x["video_hashcode"] for x in batch],
        "current_frame_idx_1based": [x["current_frame_idx_1based"] for x in batch],
        "t_ai": [x["t_ai"] for x in batch],
        "t_co": [x["t_co"] for x in batch],
    }


def anticipation_collate_fn(batch):
    frames = torch.stack([x["frames"] for x in batch], dim=0)  # [B, T, C, H, W]

    # Only stack future frames if ALL items have them
    has_future = [x["future_frames"] is not None for x in batch]

    if all(has_future):
        future_frames = torch.stack([x["future_frames"] for x in batch], dim=0)
    else:
        future_frames = None

    return {
        "frames": frames,
        "future_frames": future_frames,
        "has_future": torch.tensor(has_future, dtype=torch.bool),
        "binary_target": torch.stack([x["binary_target"] for x in batch], dim=0),
        "risk_target": torch.stack([x["risk_target"] for x in batch], dim=0),
        "valid_progress": torch.stack([x["valid_progress"] for x in batch], dim=0),
        "pref_valid": torch.stack([x["pref_valid"] for x in batch], dim=0),
        "video_hashcode": [x["video_hashcode"] for x in batch],
        "current_frame_idx_1based": [x["current_frame_idx_1based"] for x in batch],
        "t_ai": [x["t_ai"] for x in batch],
        "t_co": [x["t_co"] for x in batch],
    }

