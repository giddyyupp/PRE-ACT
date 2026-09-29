import random

from torchvision import transforms
from torchvision.transforms import functional as TF


class ClipTransform:
    """Frame transform applied to a whole clip at once.

    augment=True reproduces:
        RandomResizedCrop(area_range=(0.8, 1.0), aspect_ratio_range=(4/3, 16/9))
        -> Resize((size, size), keep_ratio=False) -> Flip(flip_ratio=0.5)
    The crop box and flip are drawn once per call and reused for every frame, as
    mmaction does. Drawing them per frame would break temporal alignment.

    augment=False is the plain Resize -> ToTensor -> Normalize used for eval.

    Takes a list of PIL images and returns a list of [C, H, W] tensors. A single
    PIL image is also accepted and returns a single tensor, so this stays a drop-in
    for loaders that still transform frame by frame.

    clip_level marks this as list-capable; dataloaders check it to decide whether
    to hand over the whole clip or keep calling per frame.
    """

    clip_level = True

    def __init__(
        self,
        size,
        mean,
        std,
        scale=(0.8, 1.0),
        ratio=(4.0 / 3.0, 16.0 / 9.0),
        flip_p=0.5,
    ):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, (list, tuple)):
            assert len(size) == 2, "size must be int or (h, w)"
            self.size = size

        self.scale = scale
        self.ratio = ratio
        self.flip_p = flip_p
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, images, augment=False):
        single = not isinstance(images, (list, tuple))
        if single:
            images = [images]

        crop = (
            transforms.RandomResizedCrop.get_params(images[0], self.scale, self.ratio)
            if augment
            else None
        )
        flip = augment and random.random() < self.flip_p

        out = []
        for img in images:
            img = (
                TF.resized_crop(img, *crop, self.size)
                if crop is not None
                else TF.resize(img, self.size)
            )
            if flip:
                img = TF.hflip(img)
            out.append(self.normalize(self.to_tensor(img)))
        return out[0] if single else out

