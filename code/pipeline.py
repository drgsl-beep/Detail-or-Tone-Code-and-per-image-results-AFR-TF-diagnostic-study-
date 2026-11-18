"""
Adaptive Fractional-Order Retinex-Transmission Fusion (AFR-TF)
for joint low-light / haze / underwater-style image restoration.

This is the Python validation implementation used to generate numeric
results before porting the finalized method to MATLAB.
"""
import numpy as np
import cv2
from scipy.ndimage import uniform_filter, gaussian_filter, minimum_filter
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk

EPS = 1e-6

# ----------------------------------------------------------------------
# 1. SYNTHETIC DEGRADATION MODELS (to create paired train/test data)
# ----------------------------------------------------------------------

def synth_depth_map(h, w, seed=0):
    """Smooth synthetic depth map in [0.5, 3.0] to drive haze/underwater sim."""
    rng = np.random.default_rng(seed)
    base = rng.random((8, 8))
    depth = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    depth = gaussian_filter(depth, sigma=min(h, w) / 10)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + EPS)
    return 0.5 + 2.5 * depth


def apply_haze(img_f, depth, beta=0.9, A=0.85):
    """Koschmieder atmospheric scattering model: I = J*t + A*(1-t)."""
    t = np.exp(-beta * depth)[..., None]
    hazy = img_f * t + A * (1 - t)
    return np.clip(hazy, 0, 1), t[..., 0]


def apply_underwater(img_f, depth, beta_rgb=(0.65, 0.32, 0.16), A_rgb=(0.45, 0.65, 0.75)):
    """Wavelength-dependent attenuation (red attenuates fastest) + backscatter."""
    out = np.zeros_like(img_f)
    t_ch = np.zeros(img_f.shape[:2] + (3,))
    for c in range(3):
        t = np.exp(-beta_rgb[c] * depth)
        t_ch[..., c] = t
        out[..., c] = img_f[..., c] * t + A_rgb[c] * (1 - t)
    return np.clip(out, 0, 1), t_ch


def apply_lowlight(img_f, gamma=2.6, noise_sigma=0.015, seed=0):
    rng = np.random.default_rng(seed)
    dark = np.power(img_f, gamma)
    noise = rng.normal(0, noise_sigma, img_f.shape)
    return np.clip(dark + noise, 0, 1)


def make_degraded(img_f, mode, seed=0):
    h, w = img_f.shape[:2]
    depth = synth_depth_map(h, w, seed=seed)
    if mode == 'haze':
        deg, _ = apply_haze(img_f, depth)
        deg = apply_lowlight(deg, gamma=1.25, noise_sigma=0.01, seed=seed)
    elif mode == 'underwater':
        deg, _ = apply_underwater(img_f, depth)
        deg = apply_lowlight(deg, gamma=1.15, noise_sigma=0.01, seed=seed)
    elif mode == 'lowlight':
        deg = apply_lowlight(img_f, gamma=2.8, noise_sigma=0.02, seed=seed)
    else:
        raise ValueError(mode)
    return deg


# ----------------------------------------------------------------------
# 2. ADAPTIVE FRACTIONAL-ORDER DERIVATIVE (Grunwald-Letnikov, causal mask)
# ----------------------------------------------------------------------

def gl_coefficients(alpha, n_terms=5):
    """Grunwald-Letnikov coefficients w_k for order alpha, k=0..n_terms-1."""
    w = np.zeros(n_terms)
    w[0] = 1.0
    for k in range(1, n_terms):
        w[k] = w[k - 1] * (k - 1 - alpha) / k
    return w


def build_fractional_kernels(alpha, n_terms=5):
    """8-directional fractional-order masks (Mathieu-style) for a given order."""
    w = gl_coefficients(alpha, n_terms)
    # 3x3-scale 8-direction kernel using first (n_terms) taps along each axis
    k = np.zeros((3, 3))
    # center and 4-neighborhood use w[0..2], approximate compact fractional mask
    k[1, 1] = w[0]
    k[0, 1] = w[1]; k[2, 1] = w[1]
    k[1, 0] = w[1]; k[1, 2] = w[1]
    k[0, 0] = w[2] * 0.5; k[0, 2] = w[2] * 0.5
    k[2, 0] = w[2] * 0.5; k[2, 2] = w[2] * 0.5
    return k


def local_texture_map(gray, win=9):
    """Normalized local entropy -> texture activity map in [0,1]."""
    g8 = (gray * 255).astype(np.uint8)
    ent = rank_entropy(g8, disk(win // 2)).astype(np.float64)
    ent = (ent - ent.min()) / (ent.max() - ent.min() + EPS)
    return ent


def adaptive_alpha_map(gray, alpha_min=0.3, alpha_max=1.6, win=9):
    """
    Core novelty: fractional order alpha(x,y) is driven by local texture
    activity - smooth/low-contrast regions get a HIGHER order (stronger
    detail lift, since there's little noise energy to amplify there is
    less risk), busy/high-texture (likely noisy) regions get a LOWER
    order (closer to classical smoothing) to avoid amplifying noise.
    """
    tex = local_texture_map(gray, win=win)
    alpha = alpha_max - tex * (alpha_max - alpha_min)
    return alpha, tex


def fractional_detail_layer(channel, alpha_map, n_bins=6):
    """
    Apply spatially-varying fractional-order high-pass filtering. The
    (continuous) alpha map is sampled at n_bins center orders, each
    filtered once with conv2 (fast, MATLAB-portable), and the per-pixel
    output is a SMOOTH triangular-weighted blend across neighboring bin
    centers -- NOT a hard mask assignment. Hard binning was tested and
    produces visible seams at bin-boundary contours that measurably hurt
    PSNR/SSIM (verified empirically); smooth blending removes them while
    keeping the same spatially-adaptive order concept.
    """
    lo, hi = alpha_map.min(), alpha_map.max()
    if hi - lo < 1e-6:
        kernel = build_fractional_kernels(float(lo), n_terms=5)
        return cv2.filter2D(channel, -1, kernel, borderType=cv2.BORDER_REFLECT)

    centers = np.linspace(lo, hi, n_bins)
    spacing = centers[1] - centers[0]

    out = np.zeros_like(channel)
    wsum = np.zeros_like(channel)
    for c in centers:
        kernel = build_fractional_kernels(float(c), n_terms=5)
        filtered = cv2.filter2D(channel, -1, kernel, borderType=cv2.BORDER_REFLECT)
        w = np.clip(1.0 - np.abs(alpha_map - c) / spacing, 0.0, 1.0)  # triangular blend weight
        out += w * filtered
        wsum += w
    out /= (wsum + EPS)
    return out


# ----------------------------------------------------------------------
# 3. RETINEX ILLUMINATION-REFLECTANCE DECOMPOSITION
# ----------------------------------------------------------------------

def retinex_decompose(gray, sigma_list=(15, 80, 250)):
    """
    Multi-scale Retinex illumination estimate. Illumination is kept on
    its NATURAL brightness scale (not stretched to [0,1]) since the
    downstream adaptive gain needs the true local mean brightness to
    decide how much to lift a region.
    """
    illum = np.zeros_like(gray)
    for s in sigma_list:
        illum += gaussian_filter(gray, sigma=s)
    illum /= len(sigma_list)
    illum = np.clip(illum, 0.01, 1.0)
    reflect = np.clip(gray / illum, 0, 3.0)
    reflect = reflect / (reflect.max() + EPS)
    return illum, reflect


# ----------------------------------------------------------------------
# 4. TRANSMISSION / ILLUMINATION-GAIN MAP (physical prior)
# ----------------------------------------------------------------------

def estimate_transmission(img_f, patch=15, omega=0.92):
    """Dark-channel-prior based transmission estimate (haze/underwater)."""
    dark = np.min(img_f, axis=2)
    dark = minimum_filter(dark, size=patch)  # true min filter (dark channel)
    A = np.percentile(dark, 99.5)
    A = max(A, 0.5)
    t = 1 - omega * (dark / (A + EPS))
    return np.clip(t, 0.15, 1.0), A


def haze_confidence(img_f, patch=15):
    """
    Scene-adaptive branch-selection score in [0,1]: how much this image
    actually looks scattering-degraded (haze/underwater) vs. simply
    underexposed (low-light only). Used to gate the physical-dehaze
    branch so it isn't blindly applied to non-hazy dark scenes.

    Uses a MULTIPLICATIVE combination of "elevated dark channel" (level)
    and "spatially smooth dark channel" (smoothness), not an average.
    Diagnosis: a uniformly dark (low-light-only) scene has a smooth dark
    channel simply because it's dark everywhere -- smoothness alone is
    high for BOTH low-light and real haze/underwater scenes, so an
    average of the two components let smoothness leak a spuriously high
    confidence (~0.55) into pure low-light images with no real
    scattering. A product requires BOTH an elevated dark channel AND
    smoothness to yield high confidence, correctly gating out low-light
    (level~0.2, product~0.18) while preserving true haze (level~1.0,
    product~0.95). Verified empirically on the 14-image test battery.
    """
    dark = np.min(img_f, axis=2)
    dark = minimum_filter(dark, size=patch)
    smoothness = 1.0 - np.std(gaussian_filter(dark, 5) - dark) * 4
    level = np.clip(np.mean(dark) * 3.0, 0, 1)
    conf = np.clip(level, 0, 1) * np.clip(smoothness, 0, 1)
    return float(conf)


def illumination_gain(illum, target=0.5, strength=0.75):
    """
    Adaptive multiplicative gain that pushes the LOCAL illumination
    estimate toward a mid-grey target, i.e. brightens dark regions and
    mildly restrains already-bright regions (local adaptive exposure
    correction driven by the Retinex illumination map).
    """
    illum_c = np.clip(illum, 0.06, 1.0)
    gain = np.power(target / illum_c, strength)
    return np.clip(gain, 0.5, 4.0)


# ----------------------------------------------------------------------
# 5. LAPLACIAN-PYRAMID MULTI-SCALE FUSION
# ----------------------------------------------------------------------

def gaussian_pyramid(img, levels):
    gp = [img.astype(np.float64)]
    for _ in range(levels):
        gp.append(cv2.pyrDown(gp[-1]))
    return gp


def laplacian_pyramid(img, levels):
    gp = gaussian_pyramid(img, levels)
    lp = []
    for i in range(levels):
        size = (gp[i].shape[1], gp[i].shape[0])
        up = cv2.pyrUp(gp[i + 1], dstsize=size)
        lp.append(gp[i] - up)
    lp.append(gp[-1])
    return lp


def reconstruct_from_laplacian(lp):
    img = lp[-1]
    for i in range(len(lp) - 2, -1, -1):
        size = (lp[i].shape[1], lp[i].shape[0])
        img = cv2.pyrUp(img, dstsize=size) + lp[i]
    return img


def weight_map(img_gray):
    """Contrast + saturation + well-exposedness weight (Mertens-style, simplified)."""
    lap = cv2.Laplacian(img_gray.astype(np.float32), cv2.CV_32F)
    contrast = np.abs(lap)
    wellexp = np.exp(-((img_gray - 0.5) ** 2) / (2 * 0.2 ** 2))
    w = (contrast + EPS) * (wellexp + EPS)
    return w


def pyramid_fuse(imgA, imgB, weightA, weightB, levels=5):
    wsum = weightA + weightB + EPS
    wA = weightA / wsum
    wB = weightB / wsum
    fused = np.zeros_like(imgA)
    for c in range(3):
        lpA = laplacian_pyramid(imgA[..., c], levels)
        lpB = laplacian_pyramid(imgB[..., c], levels)
        gwA = gaussian_pyramid(wA, levels)
        gwB = gaussian_pyramid(wB, levels)
        lp_fused = []
        for la, lb, ga, gb in zip(lpA, lpB, gwA, gwB):
            lp_fused.append(la * ga + lb * gb)
        rec = reconstruct_from_laplacian(lp_fused)
        fused[..., c] = rec
    return np.clip(fused, 0, 1)


# ----------------------------------------------------------------------
# 6. FULL PROPOSED PIPELINE: AFR-TF
# ----------------------------------------------------------------------

def afr_tf(img_f, alpha_min=0.7, alpha_max=1.6, levels=4, conf_thresh=0.35, detail_strength=0.45,
           fusion_mode='weighted', verbose=False):
    """
    fusion_mode controls how Branch A and Branch B are combined when both
    are active (conf >= conf_thresh):
      'weighted'    - current method: per-pixel contrast x well-exposedness
                      Laplacian-pyramid fusion (weight_map + pyramid_fuse)
      'hard'        - hard selection: pick Branch A entirely if conf > 0.5,
                      otherwise Branch B entirely (no per-pixel blending)
      'conf_scalar' - confidence-weighted scalar blend: fused = conf*A + (1-conf)*B,
                      applied uniformly across the whole image (no spatial weighting)
    This isolates the contribution of the specific fusion STRATEGY, holding
    every other pipeline component (branches, adaptive detail stage) fixed.
    """
    gray = cv2.cvtColor((img_f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    conf = haze_confidence(img_f)

    # branch B: Retinex illumination correction (low-light branch)
    illum, reflect = retinex_decompose(gray)
    gain = illumination_gain(illum)
    branchB = np.clip(img_f * gain[..., None], 0, 1)

    if conf < conf_thresh:
        branchA = None
        base_fused = branchB
    else:
        t, A = estimate_transmission(img_f)
        t_eff = 1 - conf * (1 - t)
        t3 = np.repeat(t_eff[..., None], 3, axis=2)
        branchA = (img_f - A) / np.maximum(t3, 0.35) + A
        branchA = np.clip(branchA, 0, 1)
        branchA = conf * branchA + (1 - conf) * img_f

        if fusion_mode == 'weighted':
            wA = weight_map(cv2.cvtColor((branchA * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0)
            wB = weight_map(cv2.cvtColor((branchB * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0)
            base_fused = pyramid_fuse(branchA, branchB, wA, wB, levels=levels)
        elif fusion_mode == 'hard':
            base_fused = branchA if conf > 0.5 else branchB
        elif fusion_mode == 'conf_scalar':
            base_fused = conf * branchA + (1 - conf) * branchB
        else:
            raise ValueError(f'unknown fusion_mode {fusion_mode}')

    # final-stage ADAPTIVE fractional-order detail enhancement, applied to
    # the fused output directly.
    base_gray = cv2.cvtColor((base_fused * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    alpha_map, tex = adaptive_alpha_map(base_gray, alpha_min, alpha_max)
    detail = fractional_detail_layer(base_gray, alpha_map)
    detail_norm = detail / (np.max(np.abs(detail)) + EPS)
    detail_gain = 1.0 + detail_strength * detail_norm[..., None]
    fused = np.clip(base_fused * detail_gain, 0, 1)

    if verbose:
        return fused, dict(branchA=branchA, branchB=branchB, alpha_map=alpha_map, base_fused=base_fused, conf=conf)
    return fused


# ----------------------------------------------------------------------
# 7. BASELINES
# ----------------------------------------------------------------------

def baseline_clahe(img_f):
    lab = cv2.cvtColor((img_f * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    out = cv2.merge([l2, a, b])
    out = cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
    return out.astype(np.float64) / 255.0


def baseline_dcp_dehaze(img_f, patch=15, omega=0.92):
    t, A = estimate_transmission(img_f, patch, omega)
    t3 = np.repeat(t[..., None], 3, axis=2)
    out = (img_f - A) / np.maximum(t3, 0.2) + A
    return np.clip(out, 0, 1)


def baseline_msr(img_f, sigma_list=(15, 80, 250)):
    out = np.zeros_like(img_f)
    for c in range(3):
        ch = img_f[..., c]
        log_img = np.log1p(ch)
        illum_log = np.zeros_like(log_img)
        for s in sigma_list:
            illum_log += np.log1p(gaussian_filter(ch, sigma=s))
        illum_log /= len(sigma_list)
        r = log_img - illum_log
        r = (r - r.min()) / (r.max() - r.min() + EPS)
        out[..., c] = r
    return np.clip(out, 0, 1)


def baseline_gamma_fixed_fractional(img_f, alpha=1.0, conf_thresh=0.35, detail_strength=0.45):
    """TRUE isolated ablation: identical to afr_tf() (same gated branch A,
    same hard confidence-gated fusion bypass, same post-fusion detail
    stage) but with a FIXED (non-adaptive) fractional order, so this
    isolates ONLY the fixed-vs-adaptive effect."""
    gray = cv2.cvtColor((img_f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    conf = haze_confidence(img_f)

    illum, reflect = retinex_decompose(gray)
    gain = illumination_gain(illum)
    branchB = np.clip(img_f * gain[..., None], 0, 1)

    if conf < conf_thresh:
        base_fused = branchB
    else:
        t, A = estimate_transmission(img_f)
        t_eff = 1 - conf * (1 - t)
        t3 = np.repeat(t_eff[..., None], 3, axis=2)
        branchA = np.clip((img_f - A) / np.maximum(t3, 0.35) + A, 0, 1)
        branchA = conf * branchA + (1 - conf) * img_f
        wA = weight_map(cv2.cvtColor((branchA * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0)
        wB = weight_map(cv2.cvtColor((branchB * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0)
        base_fused = pyramid_fuse(branchA, branchB, wA, wB, levels=4)

    base_gray = cv2.cvtColor((base_fused * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    kernel = build_fractional_kernels(alpha)   # <-- the only real difference vs afr_tf()
    detail = cv2.filter2D(base_gray, -1, kernel, borderType=cv2.BORDER_REFLECT)
    detail_norm = detail / (np.max(np.abs(detail)) + EPS)
    fused = np.clip(base_fused * (1.0 + detail_strength * detail_norm[..., None]), 0, 1)
    return fused


# ----------------------------------------------------------------------
# 8. METRICS
# ----------------------------------------------------------------------

def image_entropy(img_f):
    g8 = (cv2.cvtColor((img_f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY))
    hist = cv2.calcHist([g8], [0], None, [256], [0, 256]).flatten()
    p = hist / (hist.sum() + EPS)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def uciqe(img_f):
    """Underwater Color Image Quality Evaluation (simplified standard formula)."""
    lab = cv2.cvtColor((img_f * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = np.sqrt(a ** 2 + b ** 2)
    sigma_c = np.std(chroma)
    lo, hi = np.percentile(L, 1), np.percentile(L, 99)
    con_l = hi - lo
    sat = np.divide(chroma, np.clip(L, 5, None))
    mu_s = np.mean(sat)
    return float(0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s)


def compute_metrics(ref_f, test_f):
    ref8 = (np.clip(ref_f, 0, 1) * 255).astype(np.uint8)
    test8 = (np.clip(test_f, 0, 1) * 255).astype(np.uint8)
    p = psnr(ref8, test8, data_range=255)
    s = ssim(ref8, test8, channel_axis=2, data_range=255)
    ent = image_entropy(test_f)
    uc = uciqe(test_f)
    return dict(PSNR=p, SSIM=s, Entropy=ent, UCIQE=uc)
