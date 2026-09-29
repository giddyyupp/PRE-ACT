import torch


def build_optimizer(model, args):
    backbone_params = []
    head_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone"):
            backbone_params.append(p)
        else:
            head_params.append(p)

    param_groups = []
    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": args.backbone_lr,
            "weight_decay": args.weight_decay,
        })
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": args.head_lr,
            "weight_decay": args.weight_decay,
        })

    optimizer = torch.optim.AdamW(param_groups)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10, # this should have been 10 or even 5.
        gamma=0.1,
    )

    return optimizer, scheduler

