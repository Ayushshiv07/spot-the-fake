"""
features.py — Handcrafted feature extraction for photo authenticity detection.

Optimised for speed and memory efficiency using PIL lazy loading.
Only 512x512 patches are converted to NumPy — never the full megapixel image.

Feature vector (length 13):
  0:  hf_energy_ratio       -- HF/total FFT energy ratio (avg over patches)
  1:  hf_peak_to_mean       -- FFT peak-to-mean in HF band; moire = sharp spikes
  2:  directional_asymmetry -- H/V vs diagonal FFT energy ratio
  3:  laplacian_variance     -- sharpness (Laplacian response variance)
  4:  saturation_entropy     -- entropy of HSV saturation histogram
  5:  brightness_mean        -- mean of HSV value channel
  6:  brightness_std         -- std of HSV value channel
  7:  glare_ratio            -- fraction of near-white (>250/255) pixels
  8:  jpeg_blockiness        -- 8x8 DCT block-edge artefact score (double-JPEG)
  9:  noise_level            -- sensor noise in smooth flat regions
  10: chroma_blur_ratio      -- colour fringing / chromatic aberration proxy
  11: saturation_std         -- spread of saturation; screens are more uniform
  12: fft_radial_falloff     -- how fast FFT magnitude drops from DC to HF (natural images fall off smoothly, screens have bumps)
"""

import numpy as np
from PIL import Image
from scipy.ndimage import laplace, uniform_filter
from scipy.stats import entropy as scipy_entropy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "hf_energy_ratio",
    "hf_peak_to_mean",
    "directional_asymmetry",
    "laplacian_variance",
    "saturation_entropy",
    "brightness_mean",
    "brightness_std",
    "glare_ratio",
    "jpeg_blockiness",
    "noise_level",
    "chroma_blur_ratio",
    "saturation_std",
    "fft_radial_falloff",
]

PATCH_SIZE = 512   # crop size for FFT & spatial features
N_PATCHES  = 5     # number of random crops to average FFT features over
RNG_SEED   = 42    # reproducible patch positions


# ---------------------------------------------------------------------------
# PIL lazy-loading patch extractor
# ---------------------------------------------------------------------------

def _get_pil_image_and_size(image_path: str):
    img = Image.open(image_path)
    return img, img.width, img.height


def _extract_pil_patches(img: Image.Image, w: int, h: int, n: int, size: int, mode: str) -> list:
    rng = np.random.default_rng(RNG_SEED)
    if w <= size or h <= size:
        full_arr = np.array(img.convert(mode))
        pad_h = max(size - h, 0)
        pad_w = max(size - w, 0)
        if full_arr.ndim == 2:
            p = np.pad(full_arr, ((0, pad_h), (0, pad_w)), mode="reflect")
        else:
            p = np.pad(full_arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        return [p[:size, :size]] * n
    patches = []
    for _ in range(n):
        x0 = int(rng.integers(0, w - size))
        y0 = int(rng.integers(0, h - size))
        patches.append(np.array(img.crop((x0, y0, x0 + size, y0 + size)).convert(mode)))
    return patches


def _extract_center_pil_patch(img: Image.Image, w: int, h: int, size: int, mode: str) -> np.ndarray:
    if w <= size or h <= size:
        full_arr = np.array(img.convert(mode))
        pad_h = max(size - h, 0)
        pad_w = max(size - w, 0)
        if full_arr.ndim == 2:
            p = np.pad(full_arr, ((0, pad_h), (0, pad_w)), mode="reflect")
        else:
            p = np.pad(full_arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        return p[:size, :size]
    x0 = (w - size) // 2
    y0 = (h - size) // 2
    return np.array(img.crop((x0, y0, x0 + size, y0 + size)).convert(mode))


# ---------------------------------------------------------------------------
# HSV conversion
# ---------------------------------------------------------------------------

def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0] / 255.0
    g = rgb[:, :, 1] / 255.0
    b = rgb[:, :, 2] / 255.0
    cmax  = np.maximum(np.maximum(r, g), b)
    cmin  = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    h = np.zeros_like(r)
    mask = delta != 0
    mr = mask & (cmax == r)
    mg = mask & (cmax == g)
    mb = mask & (cmax == b)
    h[mr] = (60 * ((g[mr] - b[mr]) / delta[mr])) % 360
    h[mg] = (60 * ((b[mg] - r[mg]) / delta[mg]) + 120) % 360
    h[mb] = (60 * ((r[mb] - g[mb]) / delta[mb]) + 240) % 360
    h /= 360.0
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(cmax == 0, 0.0, delta / cmax)
    return np.stack([h, s, cmax], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# FFT patch features
# ---------------------------------------------------------------------------

def _fft_features_patch(patch: np.ndarray) -> np.ndarray:
    """Compute 4 FFT features on one grayscale patch. Returns shape (4,)."""
    fft = np.fft.rfft2(patch)
    mag = np.log1p(np.abs(fft))

    h, w = mag.shape
    cy = h // 2
    Y, X = np.ogrid[:h, :w]
    dist   = np.sqrt((Y - cy) ** 2 + X ** 2)
    radius = min(cy, w)

    hf_mask = dist > 0.35 * radius

    # Feature 0: HF energy ratio
    hf_energy    = mag[hf_mask].sum()
    total_energy = mag.sum()
    hf_ratio     = hf_energy / (total_energy + 1e-8)

    # Feature 1: peak-to-mean in HF band
    hf_vals      = mag[hf_mask]
    peak_to_mean = hf_vals.max() / (hf_vals.mean() + 1e-8)

    # Feature 2: directional asymmetry (H/V vs diagonals)
    angle     = np.degrees(np.arctan2(np.abs(Y - cy), X + 1e-6)) % 180
    hv_mask   = hf_mask & ((angle <= 10) | (angle >= 170) | ((angle >= 80) & (angle <= 100)))
    diag_mask = hf_mask & (((angle >= 35) & (angle <= 55)) | ((angle >= 125) & (angle <= 145)))
    hv_energy   = mag[hv_mask].sum()
    diag_energy = mag[diag_mask].sum()
    dir_asym    = hv_energy / (diag_energy + 1e-8)

    # Feature 3: radial falloff — how fast spectrum energy drops from DC to HF.
    # Natural images: smooth ~1/f fall-off. Screens: bumpy due to pixel grid peaks.
    # Measure: ratio of energy in mid-freq ring (15-35% radius) vs HF ring (>35%).
    mid_mask = (dist > 0.15 * radius) & (dist <= 0.35 * radius)
    mid_energy = mag[mid_mask].sum()
    radial_falloff = mid_energy / (hf_energy + 1e-8)

    return np.array([hf_ratio, peak_to_mean, dir_asym, radial_falloff], dtype=np.float32)


def _fft_features_averaged(gray: np.ndarray) -> np.ndarray:
    """Average 4 FFT features over N_PATCHES random crops. Returns shape (4,)."""
    patches = _extract_pil_patches_from_array(gray, N_PATCHES, PATCH_SIZE)
    feats   = np.stack([_fft_features_patch(p) for p in patches], axis=0)
    return feats.mean(axis=0)


def _extract_pil_patches_from_array(arr: np.ndarray, n: int, size: int) -> list:
    """Extract n random size×size crops from a numpy array directly."""
    h, w = arr.shape[:2]
    rng = np.random.default_rng(RNG_SEED)
    if h <= size or w <= size:
        pad_h = max(size - h, 0)
        pad_w = max(size - w, 0)
        p = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="reflect")
        return [p[:size, :size]] * n
    patches = []
    for _ in range(n):
        y0 = int(rng.integers(0, h - size))
        x0 = int(rng.integers(0, w - size))
        patches.append(arr[y0:y0 + size, x0:x0 + size])
    return patches


# ---------------------------------------------------------------------------
# Individual spatial / colour features (on center patch)
# ---------------------------------------------------------------------------

def _feat_laplacian_variance(gray_patch: np.ndarray) -> float:
    return float(laplace(gray_patch.astype(np.float32)).var())


def _feat_saturation_entropy(rgb_patch: np.ndarray) -> float:
    hsv  = _rgb_to_hsv(rgb_patch)
    hist, _ = np.histogram(hsv[:, :, 1], bins=32, range=(0.0, 1.0))
    hist = hist.astype(np.float64) + 1e-8
    hist /= hist.sum()
    return float(scipy_entropy(hist))


def _feat_saturation_std(rgb_patch: np.ndarray) -> float:
    """Std of saturation channel — screens show unnaturally uniform saturation."""
    hsv = _rgb_to_hsv(rgb_patch)
    return float(hsv[:, :, 1].std())


def _feat_brightness_mean(rgb_patch: np.ndarray) -> float:
    return float(_rgb_to_hsv(rgb_patch)[:, :, 2].mean())


def _feat_brightness_std(rgb_patch: np.ndarray) -> float:
    return float(_rgb_to_hsv(rgb_patch)[:, :, 2].std())


def _feat_glare_ratio(gray_patch: np.ndarray) -> float:
    return float(np.sum(gray_patch > 250) / (gray_patch.size + 1e-8))


def _feat_jpeg_blockiness(gray_patch: np.ndarray) -> float:
    crop = gray_patch.astype(np.float32)
    h, w = crop.shape
    block_diffs, nonblock_diffs = [], []
    for off in range(8, h - 1, 8):
        block_diffs.append(np.abs(crop[off, :] - crop[off - 1, :]).mean())
    for off in range(4, h - 1, 8):
        nonblock_diffs.append(np.abs(crop[off, :] - crop[off - 1, :]).mean())
    for off in range(8, w - 1, 8):
        block_diffs.append(np.abs(crop[:, off] - crop[:, off - 1]).mean())
    for off in range(4, w - 1, 8):
        nonblock_diffs.append(np.abs(crop[:, off] - crop[:, off - 1]).mean())
    block_mean    = float(np.mean(block_diffs))    if block_diffs    else 0.0
    nonblock_mean = float(np.mean(nonblock_diffs)) if nonblock_diffs else 1.0
    return block_mean / (nonblock_mean + 1e-8)


def _feat_noise_level(gray_patch: np.ndarray) -> float:
    crop = gray_patch.astype(np.float32)
    h, w = crop.shape
    block_vars = []
    for y in range(0, h - 8, 8):
        for x in range(0, w - 8, 8):
            block_vars.append(crop[y:y + 8, x:x + 8].var())
    if not block_vars:
        return 0.0
    block_vars = np.array(block_vars)
    flat_threshold = np.percentile(block_vars, 20)
    flat_vars = block_vars[block_vars <= flat_threshold + 1e-8]
    return float(flat_vars.mean()) if len(flat_vars) > 0 else 0.0


def _feat_chroma_blur_ratio(rgb_patch: np.ndarray) -> float:
    """
    Chromatic aberration proxy — real camera lenses create slight colour fringing
    at high-contrast edges (different wavelengths focus at slightly different depths).
    Screens display perfect pixel-aligned RGB with no chromatic aberration.

    Method: compute Laplacian sharpness separately on R, G, B channels.
    In real photos, channels have slightly different sharpness.
    In screen recaptures, all channels are nearly equally sharp (pixel-perfect rendering).
    Measure: max(channel_variances) / min(channel_variances).
    Higher ratio = more inter-channel sharpness difference = more likely REAL.
    """
    r = rgb_patch[:, :, 0].astype(np.float32)
    g = rgb_patch[:, :, 1].astype(np.float32)
    b = rgb_patch[:, :, 2].astype(np.float32)
    lap_r = float(laplace(r).var())
    lap_g = float(laplace(g).var())
    lap_b = float(laplace(b).var())
    ch_vars = sorted([lap_r, lap_g, lap_b])
    return float(ch_vars[2] / (ch_vars[0] + 1e-8))


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_features(image_path: str) -> np.ndarray:
    """
    Extract the 13-dimensional feature vector for a single image.

    Returns
    -------
    np.ndarray of shape (13,) with dtype float32.
    """
    img, w, h = _get_pil_image_and_size(image_path)

    # Load center patches for spatial/color features
    gray_center = _extract_center_pil_patch(img, w, h, PATCH_SIZE, "L")
    rgb_center  = _extract_center_pil_patch(img, w, h, PATCH_SIZE, "RGB")

    # FFT features averaged over N random patches
    fft_feats = _fft_features_averaged(gray_center)   # shape (4,) — avoids re-reading file

    # HSV computed once
    hsv_center = _rgb_to_hsv(rgb_center)

    return np.array([
        fft_feats[0],                              # hf_energy_ratio
        fft_feats[1],                              # hf_peak_to_mean
        fft_feats[2],                              # directional_asymmetry
        _feat_laplacian_variance(gray_center),     # laplacian_variance
        float(scipy_entropy(                       # saturation_entropy
            (np.histogram(hsv_center[:,:,1], bins=32, range=(0,1))[0].astype(np.float64) + 1e-8)
            / (np.histogram(hsv_center[:,:,1], bins=32, range=(0,1))[0].astype(np.float64).sum() + 1e-8)
        )),
        float(hsv_center[:, :, 2].mean()),         # brightness_mean
        float(hsv_center[:, :, 2].std()),          # brightness_std
        _feat_glare_ratio(gray_center),            # glare_ratio
        _feat_jpeg_blockiness(gray_center),        # jpeg_blockiness
        _feat_noise_level(gray_center),            # noise_level
        _feat_chroma_blur_ratio(rgb_center),       # chroma_blur_ratio  (NEW)
        float(hsv_center[:, :, 1].std()),          # saturation_std     (NEW)
        fft_feats[3],                              # fft_radial_falloff (NEW)
    ], dtype=np.float32)
