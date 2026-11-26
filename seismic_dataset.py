import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

PATCH_H = 144
PATCH_W = 304
MASK_H = 103
MASK_W = 213
MASK_MARGIN = 16
CONTEXT_X = (PATCH_W - MASK_W) // 2
VOLUME_W = 580
STRIP_STARTS = tuple(range(0, VOLUME_W, MASK_W))
REAL_HOLE_Z = (179, 282)
REAL_HOLE_Y = (368, 581)
SEQUENCE_STEPS = REAL_HOLE_Y[1] - REAL_HOLE_Y[0]

HELD_OUT_BOX = (76, 179, 0, VOLUME_W)
PATCH_REGIONS = ((282, 462, 0, VOLUME_W),)

OUT_PAINT_PATCH_H = 368
OUT_PAINT_PATCH_W = 256
OUT_PAINT_MASK_BAND = 24

AMP_SCALE_RANGE = (0.9, 1.1)
NOISE_SIGMA = 0.05
NOISE_BANK_SIZE = 64


def overlaps_held_out(z, x):
    hz0, hz1, hx0, hx1 = HELD_OUT_BOX
    return z < hz1 and z + PATCH_H > hz0 and x < hx1 and x + PATCH_W > hx0


def overlaps_real_hole(z):
    return z < REAL_HOLE_Z[1] and z + PATCH_H > REAL_HOLE_Z[0]


def sample_sequence_depth(rng, center):
    max_depth = min(SEQUENCE_STEPS - 1, center - 1)
    return int(rng.integers(max_depth + 1))


def make_inpainting_sample(rng, center, augment):
    z_min, z_max, _, _ = PATCH_REGIONS[rng.integers(len(PATCH_REGIONS))]
    z = int(rng.integers(z_min, z_max - PATCH_H + 1))
    strip_index = int(rng.integers(len(STRIP_STARTS)))
    mask_z = int(rng.integers(MASK_MARGIN, PATCH_H - MASK_H - MASK_MARGIN + 1))
    depth = sample_sequence_depth(rng, center)
    flip_h = augment and bool(rng.integers(2))
    amp_scale = float(rng.uniform(*AMP_SCALE_RANGE)) if augment else 1.0
    return center, z, strip_index, mask_z, depth, flip_h, amp_scale


def make_held_out_sample(rng, center, augment):
    strip_index = int(rng.integers(len(STRIP_STARTS)))
    mask_z = int(rng.integers(MASK_MARGIN, PATCH_H - MASK_H - MASK_MARGIN + 1))
    z = HELD_OUT_BOX[0] - mask_z
    depth = sample_sequence_depth(rng, center)
    flip_h = augment and bool(rng.integers(2))
    amp_scale = float(rng.uniform(*AMP_SCALE_RANGE)) if augment else 1.0
    return center, z, strip_index, mask_z, depth, flip_h, amp_scale


def extract_sequential_patch(frames, z, patch_x):
    frames = frames[:, z:z + PATCH_H, :]
    pad_left = max(0, -patch_x)
    pad_right = max(0, patch_x + PATCH_W - frames.shape[2])
    if pad_left or pad_right:
        frames = np.pad(
            frames,
            ((0, 0), (0, 0), (pad_left, pad_right)),
            mode="edge"
        )
    start = patch_x + pad_left
    return frames[:, :, start:start + PATCH_W].copy()


def extract_sequential_frames(cube, center, z, patch_x):
    frames = cube[center - 2:center + 3]
    return extract_sequential_patch(frames, z, patch_x)


def extract_sequential_frame(cube, center, z, patch_x):
    frame = cube[center:center + 1]
    return extract_sequential_patch(frame, z, patch_x)[0]


def sequential_masks(strip_index, mask_z):
    x_start = STRIP_STARTS[strip_index]
    width = min(MASK_W, VOLUME_W - x_start)
    target_mask = np.zeros((PATCH_H, PATCH_W), dtype=np.float32)
    current_mask = np.zeros_like(target_mask)
    future_mask = np.zeros_like(target_mask)
    target_mask[
        mask_z:mask_z + MASK_H,
        CONTEXT_X:CONTEXT_X + width
    ] = 1
    current_x = 0 if strip_index == 0 else CONTEXT_X
    current_mask[mask_z:mask_z + MASK_H, current_x:] = 1
    future_mask[mask_z:mask_z + MASK_H, :] = 1
    return target_mask, current_mask, future_mask


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
    z,
    strip_index,
    mask_z,
    depth,
    flip_h,
    amp_scale,
    noise
):
    patch_x = STRIP_STARTS[strip_index] - CONTEXT_X
    patch = extract_sequential_frames(cube, center, z, patch_x)
    anchor = extract_sequential_frame(cube, center - depth - 1, z, patch_x)
    target = patch[2].copy()
    target_mask, current_mask, future_mask = sequential_masks(strip_index, mask_z)
    empty_mask = np.zeros_like(current_mask)
    frame_masks = np.stack((
        empty_mask,
        empty_mask,
        current_mask,
        future_mask,
        future_mask
    ))

    patch = (patch - mean) / std
    anchor = (anchor - mean) / std
    target = (target - mean) / std
    patch *= amp_scale
    anchor *= amp_scale
    target *= amp_scale
    if noise is not None:
        patch += noise
        anchor += noise[2]
    patch *= 1 - frame_masks

    if flip_h:
        patch = patch[:, :, ::-1]
        anchor = anchor[:, ::-1]
        target = target[:, ::-1]
        target_mask = target_mask[:, ::-1]
        current_mask = current_mask[:, ::-1]

    inputs = np.concatenate((patch, anchor[None], current_mask[None]), axis=0)
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
        center, z, strip_index, mask_z, depth, flip_h, amp_scale = self.get_sample(index)
        noise = (
            self.noise_bank[index % len(self.noise_bank)]
            if self.noise_bank is not None else None
        )
        return _prepare_sequential(
            self.cube,
            self.mean,
            self.std,
            center,
            z,
            strip_index,
            mask_z,
            depth,
            flip_h,
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
