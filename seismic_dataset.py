import numpy as np
import torch
from torch.utils.data import Dataset

REGIONS = (
    (0, 368, 0, 462),
    (581, 951, 0, 462),
    (0, 951, 0, 179),
    (0, 951, 282, 462),
)

PATCH_H = 368
PATCH_W = 176
MASK_H = 213
MASK_W = 103

OUT_PAINT_PATCH_H = 368
OUT_PAINT_PATCH_W = 256
OUT_PAINT_MASK_BAND = 24

AMP_SCALE_RANGE = (0.9, 1.1)
NOISE_SIGMA = 0.05
NOISE_BANK_SIZE = 64


def make_inpainting_sample(rng, center, augment):
    y_min, y_max, z_min, z_max = REGIONS[rng.integers(len(REGIONS))]
    y = int(rng.integers(y_min, y_max - PATCH_H + 1))
    z = int(rng.integers(z_min, z_max - PATCH_W + 1))
    mask_y = int(rng.integers(16, PATCH_H - MASK_H - 15))
    mask_z = int(rng.integers(16, PATCH_W - MASK_W - 15))
    flip_h = augment and bool(rng.integers(2))
    flip_v = augment and bool(rng.integers(2))
    amp_scale = float(rng.uniform(*AMP_SCALE_RANGE)) if augment else 1.0
    return center, y, z, mask_y, mask_z, flip_h, flip_v, amp_scale


def make_outpainting_sample(rng, center, augment):
    y = int(rng.integers(OUT_PAINT_MASK_BAND, 457 - OUT_PAINT_PATCH_H + 1))
    z = int(rng.integers(0, 800 - OUT_PAINT_PATCH_W + 1))
    flip_h = augment and bool(rng.integers(2))
    flip_v = augment and bool(rng.integers(2))
    amp_scale = float(rng.uniform(*AMP_SCALE_RANGE)) if augment else 1.0
    return center, y, z, flip_h, flip_v, amp_scale


def _make_samples(make_sample, centers, samples_per_center, augment, seed):
    rng = np.random.default_rng(seed)
    return [
        make_sample(rng, int(center), augment)
        for center in centers
        for _ in range(samples_per_center)
    ]


def _make_noise_bank(rng, patch_h, patch_w):
    bank = []
    for _ in range(NOISE_BANK_SIZE):
        sigma = float(rng.uniform(0.0, NOISE_SIGMA))
        noise = rng.standard_normal((5, patch_h, patch_w)).astype(np.float32)
        bank.append(noise * sigma)
    return np.stack(bank)


def _prepare(
    cube,
    mean,
    std,
    patch_h,
    patch_w,
    center,
    y,
    z,
    mask,
    flip_h,
    flip_v,
    amp_scale,
    noise
):
    patch = cube[
        center - 2: center + 3,
        y: y + patch_h,
        z: z + patch_w
    ].copy()
    target = patch[2].copy()

    patch = (patch - mean) / std
    target = (target - mean) / std
    patch *= amp_scale
    target *= amp_scale
    if noise is not None:
        patch += noise

    patch *= (1 - mask)[None]

    if flip_h:
        patch = patch[:, :, ::-1]
        target = target[:, ::-1]
        mask = mask[:, ::-1]

    if flip_v:
        patch = patch[:, ::-1, :]
        target = target[::-1, :]
        mask = mask[::-1, :]

    inputs = np.concatenate((patch, mask[None]), axis=0)

    return (
        torch.from_numpy(inputs).float(),
        torch.from_numpy(target[None].copy()).float(),
        torch.from_numpy(mask[None].copy()).float()
    )


class InpaintingDataset(Dataset):
    def __init__(
        self,
        cube,
        centers,
        samples_per_center,
        mean,
        std,
        augment,
        seed,
        dynamic
    ):
        self.cube = cube
        self.centers = centers
        self.samples_per_center = samples_per_center
        self.mean = mean
        self.std = std
        self.augment = augment
        self.dynamic = dynamic
        self.rng = np.random.default_rng(seed)
        self.noise_bank = (
            _make_noise_bank(self.rng, PATCH_H, PATCH_W)
            if augment and NOISE_SIGMA > 0 else None
        )
        self.samples = None if dynamic else _make_samples(
            make_inpainting_sample,
            centers,
            samples_per_center,
            augment,
            seed
        )

    def __len__(self):
        return len(self.centers) * self.samples_per_center

    def get_sample(self, index):
        if self.samples is not None:
            return self.samples[index]
        center = int(self.centers[index // self.samples_per_center])
        return make_inpainting_sample(self.rng, center, self.augment)

    def __getitem__(self, index):
        center, y, z, mask_y, mask_z, flip_h, flip_v, amp_scale = self.get_sample(index)
        mask = np.zeros((PATCH_H, PATCH_W), dtype=np.float32)
        mask[mask_y: mask_y + MASK_H, mask_z: mask_z + MASK_W] = 1
        noise = (
            self.noise_bank[index % len(self.noise_bank)]
            if self.noise_bank is not None else None
        )
        return _prepare(
            self.cube,
            self.mean,
            self.std,
            PATCH_H,
            PATCH_W,
            center,
            y,
            z,
            mask,
            flip_h,
            flip_v,
            amp_scale,
            noise
        )


class OutpaintingDataset(Dataset):
    def __init__(
        self,
        cube,
        centers,
        samples_per_center,
        mean,
        std,
        augment,
        seed,
        dynamic
    ):
        self.cube = cube
        self.centers = centers
        self.samples_per_center = samples_per_center
        self.mean = mean
        self.std = std
        self.augment = augment
        self.dynamic = dynamic
        self.rng = np.random.default_rng(seed)
        self.noise_bank = (
            _make_noise_bank(self.rng, OUT_PAINT_PATCH_H, OUT_PAINT_PATCH_W)
            if augment and NOISE_SIGMA > 0 else None
        )
        self.samples = None if dynamic else _make_samples(
            make_outpainting_sample,
            centers,
            samples_per_center,
            augment,
            seed
        )

    def __len__(self):
        return len(self.centers) * self.samples_per_center

    def get_sample(self, index):
        if self.samples is not None:
            return self.samples[index]
        center = int(self.centers[index // self.samples_per_center])
        return make_outpainting_sample(self.rng, center, self.augment)

    def __getitem__(self, index):
        center, y, z, flip_h, flip_v, amp_scale = self.get_sample(index)
        mask = np.zeros((OUT_PAINT_PATCH_H, OUT_PAINT_PATCH_W), dtype=np.float32)
        mask[:OUT_PAINT_MASK_BAND, :] = 1
        mask[-OUT_PAINT_MASK_BAND:, :] = 1
        noise = (
            self.noise_bank[index % len(self.noise_bank)]
            if self.noise_bank is not None else None
        )
        return _prepare(
            self.cube,
            self.mean,
            self.std,
            OUT_PAINT_PATCH_H,
            OUT_PAINT_PATCH_W,
            center,
            y,
            z,
            mask,
            flip_h,
            flip_v,
            amp_scale,
            noise
        )
