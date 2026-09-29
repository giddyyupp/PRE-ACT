import torch


def remap_videomae_qkv_bias_keys(state_dict):
    """
    Convert old VideoMAE q_bias/v_bias checkpoint format to
    query.bias/key.bias/value.bias format expected by newer implementations.
    """
    new_state_dict = {}
    used_old_keys = set()

    for k, v in state_dict.items():
        if k.endswith(".attention.attention.q_bias"):
            prefix = k[:-len("q_bias")]
            new_state_dict[prefix + "query.bias"] = v
            new_state_dict[prefix + "key.bias"] = torch.zeros_like(v)
            used_old_keys.add(k)

        elif k.endswith(".attention.attention.v_bias"):
            prefix = k[:-len("v_bias")]
            new_state_dict[prefix + "value.bias"] = v
            used_old_keys.add(k)

        else:
            new_state_dict[k] = v

    return new_state_dict


def load_weights(checkpoint_path: str, model: torch.nn.Module):
    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    try:  # transformer v4 trained models.
        model.load_state_dict(ckpt["model"], strict=True)
    except RuntimeError as e:
        state_dict = ckpt["model"]
        state_dict = remap_videomae_qkv_bias_keys(state_dict)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("Missing:", missing)
        print("Unexpected:", unexpected)

