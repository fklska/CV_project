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
VOLUME_X = 580
SEQUENCE_STEPS = VOLUME_X
REAL_HOLE_Y = (368, 581)
REAL_HOLE_Z = (179, 282)

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


def overlaps_real_hole(y, z):
    return (
        y < REAL_HOLE_Y[1] and y + PATCH_H > REAL_HOLE_Y[0]
        and z < REAL_HOLE_Z[1] and z + PATCH_W > REAL_HOLE_Z[0]
    )


def _region_has_free_origin(region):
    y_min, y_max, z_min, z_max = region
    y_hi = y_max - PATCH_H
    z_hi = z_max - PATCH_W
    if y_hi < y_min or z_hi < z_min:
        return False
    hy0, hy1, hz0, hz1 = HELD_OUT_BOX
    covered = (
        y_min > hy0 - PATCH_H and y_hi < hy1
        and z_min > hz0 - PATCH_W and z_hi < hz1
    )
    return not covered


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


def sample_sequence_depth(rng, center):
    max_depth = min(SEQUENCE_STEPS - 1, center)
    return int(rng.integers(max_depth + 1))


def make_inpainting_sample(rng, center, augment):
    y_min, y_max, z_min, z_max = PATCH_REGIONS[rng.integers(len(PATCH_REGIONS))]
    for _ in range(1000):
        y = int(rng.integers(y_min, y_max - PATCH_H + 1))
        z = int(rng.integers(z_min, z_max - PATCH_W + 1))
        if not overlaps_held_out(y, z) and not overlaps_real_hole(y, z):
            break
    else:
        raise RuntimeError("Не удалось выбрать патч вне held-out и реального пропуска")
    mask_y = int(rng.integers(MASK_MARGIN, PATCH_H - MASK_H - MASK_MARGIN + 1))
    mask_z = int(rng.integers(MASK_MARGIN, PATCH_W - MASK_W - MASK_MARGIN + 1))
    depth = sample_sequence_depth(rng, center)
    flip_h = augment and bool(rng.integers(2))
    flip_v = augment and bool(rng.integers(2))
    amp_scale = float(rng.uniform(*AMP_SCALE_RANGE)) if augment else 1.0
    return center, y, z, mask_y, mask_z, depth, flip_h, flip_v, amp_scale


def make_held_out_sample(rng, center, augment):
    y_lo, y_hi, z_lo, z_hi = HELD_OUT_ORIGINS[rng.integers(len(HELD_OUT_ORIGINS))]
    y = int(rng.integers(y_lo, y_hi + 1))
    z = int(rng.integers(z_lo, z_hi + 1))
    mask_y = HELD_OUT_BOX[0] - y
    mask_z = HELD_OUT_BOX[2] - z
    depth = sample_sequence_depth(rng, center)
    flip_h = augment and bool(rng.integers(2))
    flip_v = augment and bool(rng.integers(2))
    amp_scale = float(rng.uniform(*AMP_SCALE_RANGE)) if augment else 1.0
    return center, y, z, mask_y, mask_z, depth, flip_h, flip_v, amp_scale


def extract_sequential_frames(cube, center, y, z):
    indices = np.clip(np.arange(center - 2, center + 3), 0, cube.shape[0] - 1)
    return np.stack([
        cube[index, y:y + PATCH_H, z:z + PATCH_W]
        for index in indices
    ])


def spatial_mask(mask_y, mask_z):
    mask = np.zeros((PATCH_H, PATCH_W), dtype=np.float32)
    mask[mask_y:mask_y + MASK_H, mask_z:mask_z + MASK_W] = 1
    return mask


def prediction_sources(start, steps):
    sources = np.full((steps, 2), -1, dtype=np.int64)
    for step in range(steps):
        center = start + step
        for channel, offset in enumerate((2, 1)):
            frame = max(center - offset, 0)
            source = frame - start
            if 0 <= source < step:
                sources[step, channel] = source
    return sources


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


def _make_noise_bank(rng, patch_h, patch_w):
    bank = []
    for _ in range(NOISE_BANK_SIZE):
        sigma = float(rng.uniform(0.0, NOISE_SIGMA))
        noise = rng.standard_normal((5, patch_h, patch_w)).astype(np.float32)
        bank.append(noise * sigma)
    return np.stack(bank)


def _prepare_sequential(
    cube,
    mean,
    std,
    center,
    y,
    z,
    mask_y,
    mask_z,
    flip_h,
    flip_v,
    amp_scale,
    noise
):
    patch = extract_sequential_frames(cube, center, y, z)
    target = patch[2].copy()
    target_mask = spatial_mask(mask_y, mask_z)
    frame_masks = np.repeat(target_mask[None], 5, axis=0)
    if center > 0:
        frame_masks[:2] = 0

    patch = (patch - mean) / std
    target = (target - mean) / std
    patch *= amp_scale
    target *= amp_scale
    if noise is not None:
        patch += noise
    patch *= 1 - frame_masks

    if flip_h:
        patch = patch[:, :, ::-1]
        target = target[:, ::-1]
        target_mask = target_mask[:, ::-1]

    if flip_v:
        patch = patch[:, ::-1, :]
        target = target[::-1, :]
        target_mask = target_mask[::-1, :]

    inputs = np.concatenate((patch, target_mask[None]), axis=0)
    return (
        torch.from_numpy(inputs).float(),
        torch.from_numpy(target[None].copy()).float(),
        torch.from_numpy(target_mask[None].copy()).float()
    )


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
        dynamic,
        held_out=False,
        rollout_steps=0
    ):
        self.cube = cube
        self.centers = centers
        self.samples_per_center = samples_per_center
        self.mean = mean
        self.std = std
        self.augment = augment
        self.dynamic = dynamic
        self.rollout_steps = rollout_steps
        self.sample_fn = make_held_out_sample if held_out else make_inpainting_sample
        self.rng = np.random.default_rng(seed)
        self.noise_bank = (
            _make_noise_bank(self.rng, PATCH_H, PATCH_W)
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
        center, y, z, mask_y, mask_z, depth, flip_h, flip_v, amp_scale = self.get_sample(index)
        noise = (
            self.noise_bank[index % len(self.noise_bank)]
            if self.noise_bank is not None else None
        )
        steps = self.rollout_steps or 1
        start = center
        if self.rollout_steps:
            start_offset = min(steps - 1, depth, center)
            start = min(center - start_offset, VOLUME_X - steps)

        inputs = []
        targets = []
        mask = None
        for offset in range(steps):
            step_inputs, step_target, mask = _prepare_sequential(
                self.cube,
                self.mean,
                self.std,
                start + offset,
                y,
                z,
                mask_y,
                mask_z,
                flip_h,
                flip_v,
                amp_scale,
                noise
            )
            inputs.append(step_inputs)
            targets.append(step_target)

        if not self.rollout_steps:
            return inputs[0], targets[0], mask

        sources = prediction_sources(start, steps)
        return (
            torch.stack(inputs),
            torch.stack(targets),
            mask,
            torch.from_numpy(sources)
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
