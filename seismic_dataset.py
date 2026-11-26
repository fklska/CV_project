import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

REGIONS = (
    (0, 368, 0, 462),
    (581, 951, 0, 462),
    (0, 951, 0, 179),
    (0, 951, 282, 462),
)

PATCH_H = 304
PATCH_W = 144
MASK_H = 213
MASK_W = 103
MASK_MARGIN = 16
CONTEXT_OFFSETS = (
    (-MASK_H, 0),
    (-(MASK_H // 2), 0),
    (0, -MASK_W),
    (0, -(MASK_W // 2)),
    (0, MASK_W // 2),
    (0, MASK_W),
    (MASK_H // 2, 0),
    (MASK_H, 0)
)
HORIZONTAL_FLIP_ORDER = [0, 1, 5, 4, 3, 2, 6, 7, 8]
VERTICAL_FLIP_ORDER = [7, 6, 2, 3, 4, 5, 1, 0, 8]
INPAINTING_HOLE = (368, 581, 179, 282)

HELD_OUT_BOX = (77, 290, 179, 282)

OUT_PAINT_PATCH_H = 368
OUT_PAINT_PATCH_W = 256
OUT_PAINT_MASK_BAND = 24

AMP_SCALE_RANGE = (0.9, 1.1)
NOISE_SIGMA = 0.05
NOISE_BANK_SIZE = 64


def overlaps_held_out(y, z):
    hy0, hy1, hz0, hz1 = HELD_OUT_BOX
    return y < hy1 and y + PATCH_H > hy0 and z < hz1 and z + PATCH_W > hz0


def context_overlaps_held_out(y, z):
    return any(
        overlaps_held_out(y + dy, z + dz)
        for dy, dz in CONTEXT_OFFSETS
    )


def _region_has_free_origin(region):
    y_min, y_max, z_min, z_max = region
    y_hi = y_max - PATCH_H
    z_hi = z_max - PATCH_W
    if y_hi < y_min or z_hi < z_min:
        return False
    for y in range(y_min, y_hi + 1):
        for z in range(z_min, z_hi + 1):
            if not overlaps_held_out(y, z) and not context_overlaps_held_out(y, z):
                return True
    return False


def _held_out_origins():
    hy0, hy1, hz0, hz1 = HELD_OUT_BOX
    origins = []
    for y_min, y_max, z_min, z_max in REGIONS:
        y_lo = max(y_min, hy0 - (PATCH_H - MASK_H - MASK_MARGIN))
        y_hi = min(y_max - PATCH_H, hy0 - MASK_MARGIN)
        z_lo = max(z_min, hz0 - (PATCH_W - MASK_W - MASK_MARGIN))
        z_hi = min(z_max - PATCH_W, hz0 - MASK_MARGIN)
        if y_lo <= y_hi and z_lo <= z_hi:
            origins.append((y_lo, y_hi, z_lo, z_hi))
    return tuple(origins)


assert HELD_OUT_BOX[1] - HELD_OUT_BOX[0] == MASK_H
assert HELD_OUT_BOX[3] - HELD_OUT_BOX[2] == MASK_W
PATCH_REGIONS = tuple(region for region in REGIONS if _region_has_free_origin(region))
HELD_OUT_ORIGINS = _held_out_origins()
assert PATCH_REGIONS and HELD_OUT_ORIGINS


def make_inpainting_sample(rng, center, augment):
    y_min, y_max, z_min, z_max = PATCH_REGIONS[rng.integers(len(PATCH_REGIONS))]
    for _ in range(1000):
        y = int(rng.integers(y_min, y_max - PATCH_H + 1))
        z = int(rng.integers(z_min, z_max - PATCH_W + 1))
        if not overlaps_held_out(y, z) and not context_overlaps_held_out(y, z):
            break
    else:
        raise RuntimeError("Не удалось выбрать патч вне held-out блока")
    mask_y = int(rng.integers(MASK_MARGIN, PATCH_H - MASK_H - MASK_MARGIN + 1))
    mask_z = int(rng.integers(MASK_MARGIN, PATCH_W - MASK_W - MASK_MARGIN + 1))
    flip_h = augment and bool(rng.integers(2))
    flip_v = augment and bool(rng.integers(2))
    amp_scale = float(rng.uniform(*AMP_SCALE_RANGE)) if augment else 1.0
    return center, y, z, mask_y, mask_z, flip_h, flip_v, amp_scale


def make_held_out_sample(rng, center, augment):
    y_lo, y_hi, z_lo, z_hi = HELD_OUT_ORIGINS[rng.integers(len(HELD_OUT_ORIGINS))]
    y = int(rng.integers(y_lo, y_hi + 1))
    z = int(rng.integers(z_lo, z_hi + 1))
    mask_y = HELD_OUT_BOX[0] - y
    mask_z = HELD_OUT_BOX[2] - z
    flip_h = augment and bool(rng.integers(2))
    flip_v = augment and bool(rng.integers(2))
    amp_scale = float(rng.uniform(*AMP_SCALE_RANGE)) if augment else 1.0
    return center, y, z, mask_y, mask_z, flip_h, flip_v, amp_scale


def seed_worker(worker_id):
    info = get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed)


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


def _make_noise_bank(rng, channels, patch_h, patch_w):
    bank = []
    for _ in range(NOISE_BANK_SIZE):
        sigma = float(rng.uniform(0.0, NOISE_SIGMA))
        noise = rng.standard_normal((channels, patch_h, patch_w)).astype(np.float32)
        bank.append(noise * sigma)
    return np.stack(bank)


def directional_patch(plane, y, z, mask, real_mask):
    target = plane[y: y + PATCH_H, z: z + PATCH_W].copy()
    padding = ((MASK_H, MASK_H), (MASK_W, MASK_W))
    padded = np.pad(plane, padding, mode="edge")
    padded_mask = np.pad(mask, padding)
    padded_real = np.pad(real_mask, padding)
    patches = []
    masks = []
    for dy, dz in CONTEXT_OFFSETS:
        patch_y = y + dy + MASK_H
        patch_z = z + dz + MASK_W
        mask_y = dy + MASK_H
        mask_z = dz + MASK_W
        context_mask = np.maximum(
            padded_mask[mask_y: mask_y + PATCH_H, mask_z: mask_z + PATCH_W],
            padded_real[patch_y: patch_y + PATCH_H, patch_z: patch_z + PATCH_W]
        )
        patches.append(padded[patch_y: patch_y + PATCH_H, patch_z: patch_z + PATCH_W].copy())
        masks.append(context_mask)
    patches.append(target)
    masks.append(mask)
    return np.stack(patches), np.stack(masks), target


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
    noise,
    directional=False,
    real_mask=None
):
    if directional:
        patch, masks, target = directional_patch(cube[center], y, z, mask, real_mask)
    else:
        patch = cube[
            center - 2: center + 3,
            y: y + patch_h,
            z: z + patch_w
        ].copy()
        target = patch[2].copy()
        masks = mask[None]

    patch = (patch - mean) / std
    target = (target - mean) / std
    patch *= amp_scale
    target *= amp_scale
    if noise is not None:
        patch += noise

    patch *= (1 - masks)

    if flip_h:
        if directional:
            patch = patch[HORIZONTAL_FLIP_ORDER]
        patch = patch[:, :, ::-1]
        target = target[:, ::-1]
        mask = mask[:, ::-1]

    if flip_v:
        if directional:
            patch = patch[VERTICAL_FLIP_ORDER]
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
        dynamic,
        held_out=False
    ):
        self.cube = cube
        self.centers = centers
        self.samples_per_center = samples_per_center
        self.mean = mean
        self.std = std
        self.augment = augment
        self.dynamic = dynamic
        self.sample_fn = make_held_out_sample if held_out else make_inpainting_sample
        self.rng = np.random.default_rng(seed)
        self.real_mask = np.zeros(cube.shape[1:], dtype=np.float32)
        y0, y1, z0, z1 = INPAINTING_HOLE
        self.real_mask[y0:y1, z0:z1] = 1
        self.noise_bank = (
            _make_noise_bank(self.rng, len(CONTEXT_OFFSETS) + 1, PATCH_H, PATCH_W)
            if augment and NOISE_SIGMA > 0 else None
        )
        self.samples = None if dynamic else _make_samples(
            self.sample_fn,
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
        return self.sample_fn(self.rng, center, self.augment)

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
            noise,
            True,
            self.real_mask
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
            _make_noise_bank(self.rng, 5, OUT_PAINT_PATCH_H, OUT_PAINT_PATCH_W)
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
