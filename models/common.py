import re
import torch.nn as nn


def _resolve_module(root: nn.Module, path_options):
    for path in path_options:
        obj = root
        ok = True
        for name in path:
            if hasattr(obj, name):
                obj = getattr(obj, name)
            else:
                ok = False
                break
        if ok:
            return obj, path
    return None, None


def _unfreeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def _freeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def remap_downsample(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        # downsample.conv/bn  ->  downsample.0.conv/bn
        k = re.sub(r'\.downsample\.(conv|bn)\.', r'.downsample.0.\1.', k)
        new_sd[k] = v
    return new_sd
