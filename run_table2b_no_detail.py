"""
run_table2b_no_detail.py — Task 1.1 (Table 2b)

Runs five conditions on the confirmed 14-image synthetic battery, for all
three degradation types (haze, underwater, low-light), using pipeline.py's
own branch/fusion/detail-stage code so every condition shares the exact
upstream computation:

    1. adaptive        — proposed method, alpha(x,y) from Eq. 12
    2. fixed_alpha_1.0  — integer-order ablation (matches Table 2)
    3. best_fixed       — whichever fixed alpha in {0.7..1.6} scores highest
                           mean PSNR for that degradation type (computed
                           internally from a quick sweep before the main run)
    4. mean_matched      — alpha_mean = mean(alpha(x,y)) per image, applied
                           as a single global fixed order
    5. no_detail         — detail stage entirely disabled (O = F)

Degradation parameters match Section 3.1 of the manuscript exactly:
  haze:       beta=0.9, A=0.85, + secondary low-light pass gamma=1.25, noise sigma=0.01
  underwater: beta=(0.65,0.32,0.16), A=(0.45,0.65,0.75), + secondary low-light
              pass gamma=1.15, noise sigma=0.01
  low-light:  gamma=2.8, noise sigma=0.02

Results produced with this script are in ../results (see README.md).
Fill in BATTERY below (or pass --battery), then run it and paste the
printed LaTeX table into Section 4.2 (Table 2b).

HOW TO RUN
----------
    python run_table2b_no_detail.py --battery battery.txt --out results/
    (battery.txt: one image path per line. If omitted, the default
    14-image scikit-image battery of the paper is used.)
"""
import argparse
import os
import numpy as np
import pandas as pd
import cv2
from scipy import stats
from statsmodels.stats.multitest import multipletests

from pipeline import (
    make_degraded, haze_confidence, retinex_decompose, illumination_gain,
    estimate_transmission, weight_map, pyramid_fuse,
    adaptive_alpha_map, fractional_detail_layer, build_fractional_kernels,
    compute_metrics,
)

EPS = 1e-6

# =====================================================================
# Battery image list (default: the 14 scikit-image images of the paper)
# Supply --battery battery.txt, or leave empty to use the default battery
# (one path per line). If omitted, the default scikit-image battery is used.
# =====================================================================
BATTERY = []  # e.g. ["data/astronaut.png", "data/rocket.png", ...]

DEGRADATIONS = ["haze", "underwater", "lowlight"]

# Degradation parameters exactly as documented in Section 3.1 (confirmed
# from pipeline.py's own defaults for make_degraded()).
DEGRADATION_PARAMS = {
    "haze": dict(beta=0.9, A=0.85, gamma=1.25, noise_sigma=0.01),
    "underwater": dict(beta=(0.65, 0.32, 0.16), A=(0.45, 0.65, 0.75), gamma=1.15, noise_sigma=0.01),
    "lowlight": dict(gamma=2.8, noise_sigma=0.02),
}

# Degradation seeds (the paper uses --seed-mode distinct)
# pipeline.py's make_degraded() accepts an explicit seed; the seed(s)
# actually used to produce Tables 1-6 are not recoverable from pipeline.py
# alone. Fill in the confirmed seed below (default: 0).
SEED = 0

FIXED_ALPHAS = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
ALPHA_MIN, ALPHA_MAX = 0.7, 1.6


def _load_skimage_by_name(name):
    from skimage import data, img_as_float
    loader = getattr(data, name, None)
    if loader is None:
        raise ValueError(f"skimage.data has no image called '{name}'")
    img = img_as_float(loader())
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)   # single-channel replicated to 3 channels
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img.astype(np.float64)


SEED_MODE = "single"      # "single": every case uses SEED; "distinct": SEED + 10*image_index + degradation_index
_IDX = {}                 # id(reference image) -> index in the battery, filled in main()

def case_seed(ref, deg):
    if SEED_MODE == "single":
        return SEED
    return SEED + 10 * _IDX[id(ref)] + DEGRADATIONS.index(deg)


def load_battery(paths):
    """paths: list of entries. Each entry is either a file path, or
    'skimage:<name>' to load a built-in scikit-image sample image
    (e.g. 'skimage:astronaut'). If empty, falls back to the default 14-image battery."""
    if not paths:
        print("No --battery supplied: using the default 14-image scikit-image battery of the paper.")
        default_battery = ["astronaut", "chelsea", "coffee", "rocket", "immunohistochemistry",
                       "retina", "brick", "grass", "text", "page", "moon",
                       "hubble_deep_field", "horse", "camera"]
        return {n: _load_skimage_by_name(n) for n in default_battery}
    out = {}
    for p in paths:
        if p.lower().startswith("skimage:"):
            name = p.split(":", 1)[1].strip()
            out[name] = _load_skimage_by_name(name)
            continue
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read '{p}'")
        out[os.path.basename(p)] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    return out

def _branch_fusion(img_f, alpha_min=ALPHA_MIN, alpha_max=ALPHA_MAX, levels=4, conf_thresh=0.35):
    gray = cv2.cvtColor((img_f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    conf = haze_confidence(img_f)
    illum, reflect = retinex_decompose(gray)
    gain = illumination_gain(illum)
    branchB = np.clip(img_f * gain[..., None], 0, 1)
    if conf < conf_thresh:
        return branchB
    t, A = estimate_transmission(img_f)
    t_eff = 1 - conf * (1 - t)
    t3 = np.repeat(t_eff[..., None], 3, axis=2)
    branchA = np.clip((img_f - A) / np.maximum(t3, 0.35) + A, 0, 1)
    branchA = conf * branchA + (1 - conf) * img_f
    wA = weight_map(cv2.cvtColor((branchA * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0)
    wB = weight_map(cv2.cvtColor((branchB * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0)
    return pyramid_fuse(branchA, branchB, wA, wB, levels=levels)


def cond_adaptive(img_f, detail_strength=0.45):
    base = _branch_fusion(img_f)
    gray = cv2.cvtColor((base * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    amap, _ = adaptive_alpha_map(gray, ALPHA_MIN, ALPHA_MAX)
    detail = fractional_detail_layer(gray, amap)
    detail /= (np.max(np.abs(detail)) + EPS)
    return np.clip(base * (1.0 + detail_strength * detail[..., None]), 0, 1), amap


def cond_fixed(img_f, alpha, detail_strength=0.45):
    base = _branch_fusion(img_f)
    gray = cv2.cvtColor((base * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    k = build_fractional_kernels(alpha)
    detail = cv2.filter2D(gray, -1, k, borderType=cv2.BORDER_REFLECT)
    detail /= (np.max(np.abs(detail)) + EPS)
    return np.clip(base * (1.0 + detail_strength * detail[..., None]), 0, 1)


def cond_mean_matched(img_f, detail_strength=0.45):
    base = _branch_fusion(img_f)
    gray = cv2.cvtColor((base * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    amap, _ = adaptive_alpha_map(gray, ALPHA_MIN, ALPHA_MAX)
    return cond_fixed(img_f, float(amap.mean()), detail_strength)


def cond_no_detail(img_f):
    return _branch_fusion(img_f)


def find_best_fixed_alpha(battery, deg_mode):
    """Quick internal sweep (mean PSNR only) to pick the best fixed alpha
    for this degradation type, used as condition 3 ('best_fixed')."""
    scores = {a: [] for a in FIXED_ALPHAS}
    for ref in battery.values():
        degraded = make_degraded(ref, deg_mode, seed=case_seed(ref, deg_mode))
        for a in FIXED_ALPHAS:
            out = cond_fixed(degraded, a)
            scores[a].append(compute_metrics(ref, out)['PSNR'])
    means = {a: np.mean(v) for a, v in scores.items()}
    best_a = max(means, key=means.get)
    return best_a, means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default=None, help="text file, one image path per line")
    ap.add_argument("--out", default=".", help="output directory")
    ap.add_argument("--seed", type=int, default=None, help="degradation seed (default: SEED in file)")
    ap.add_argument("--seed-mode", choices=["single", "distinct"], default="single")
    args = ap.parse_args()
    global SEED, SEED_MODE
    if args.seed is not None:
        SEED = args.seed
    SEED_MODE = args.seed_mode

    paths = []
    if args.battery:
        with open(args.battery) as f:
            paths = [l.strip() for l in f if l.strip()]
    elif BATTERY:
        paths = BATTERY

    battery = load_battery(paths)
    _IDX.clear(); _IDX.update({id(r): i for i, r in enumerate(battery.values())})
    os.makedirs(args.out, exist_ok=True)

    rows = []
    best_alpha_by_deg = {}
    for deg in DEGRADATIONS:
        best_a, _ = find_best_fixed_alpha(battery, deg)
        best_alpha_by_deg[deg] = best_a
        print(f"best fixed alpha for {deg}: {best_a}")

        for img_name, ref in battery.items():
            degraded = make_degraded(ref, deg, seed=case_seed(ref, deg))

            out_adapt, _ = cond_adaptive(degraded)
            m = compute_metrics(ref, out_adapt)
            rows.append(dict(image=img_name, degradation=deg, condition="adaptive", PSNR=m['PSNR'], SSIM=m['SSIM']))

            out_f1 = cond_fixed(degraded, 1.0)
            m = compute_metrics(ref, out_f1)
            rows.append(dict(image=img_name, degradation=deg, condition="fixed_alpha_1.0", PSNR=m['PSNR'], SSIM=m['SSIM']))

            out_best = cond_fixed(degraded, best_a)
            m = compute_metrics(ref, out_best)
            rows.append(dict(image=img_name, degradation=deg, condition="best_fixed", PSNR=m['PSNR'], SSIM=m['SSIM']))

            out_mm = cond_mean_matched(degraded)
            m = compute_metrics(ref, out_mm)
            rows.append(dict(image=img_name, degradation=deg, condition="mean_matched", PSNR=m['PSNR'], SSIM=m['SSIM']))

            out_nd = cond_no_detail(degraded)
            m = compute_metrics(ref, out_nd)
            rows.append(dict(image=img_name, degradation=deg, condition="no_detail", PSNR=m['PSNR'], SSIM=m['SSIM']))

    per_image = pd.DataFrame(rows)
    per_image.to_csv(os.path.join(args.out, "table2b_per_image.csv"), index=False)

    # --- summary with Holm correction, per degradation ---
    summary_rows = []
    for deg in DEGRADATIONS:
        sub = per_image[per_image.degradation == deg]
        piv_p = sub.pivot(index="image", columns="condition", values="PSNR")
        piv_s = sub.pivot(index="image", columns="condition", values="SSIM")
        conds = [c for c in piv_p.columns if c != "adaptive"]
        p_ts = []
        rows2 = []
        for c in conds:
            d = (piv_p["adaptive"] - piv_p[c]).dropna()
            t, p_t = stats.ttest_rel(piv_p["adaptive"], piv_p[c])
            try:
                w, p_w = stats.wilcoxon(piv_p["adaptive"], piv_p[c])
            except Exception:
                p_w = np.nan
            wins = (piv_p["adaptive"] > piv_p[c]).mean() * 100
            p_ts.append(p_t)
            rows2.append(dict(degradation=deg, condition=c,
                               mean_PSNR=piv_p[c].mean(), std_PSNR=piv_p[c].std(),
                               mean_SSIM=piv_s[c].mean(), std_SSIM=piv_s[c].std(),
                               delta_vs_adaptive=d.mean(), p_t=p_t, p_wilcoxon=p_w, wins=wins))
        # also add the adaptive row itself (no comparison, delta=0)
        rows2.append(dict(degradation=deg, condition="adaptive",
                           mean_PSNR=piv_p["adaptive"].mean(), std_PSNR=piv_p["adaptive"].std(),
                           mean_SSIM=piv_s["adaptive"].mean(), std_SSIM=piv_s["adaptive"].std(),
                           delta_vs_adaptive=0.0, p_t=np.nan, p_wilcoxon=np.nan, wins=np.nan))
        summary_rows.extend(rows2)

    summary = pd.DataFrame(summary_rows)
    summary["p_holm"] = np.nan
    summary = summary[["degradation", "condition", "mean_PSNR", "std_PSNR",
                       "mean_SSIM", "std_SSIM", "delta_vs_adaptive",
                       "p_t", "p_wilcoxon", "p_holm", "wins"]]
    # Holm correction ACROSS THE THREE DEGRADATION TYPES, separately for each condition
    # (as specified in the revision plan; each family = 3 paired t-tests).
    summary["p_holm"] = np.nan
    for cond in summary.condition.unique():
        if cond == "adaptive":
            continue
        idx = summary.index[summary.condition == cond]
        summary.loc[idx, "p_holm"] = multipletests(summary.loc[idx, "p_t"].values, method="holm")[1]
    summary.to_csv(os.path.join(args.out, "table2b_summary.csv"), index=False)
    print("\nWrote table2b_per_image.csv and table2b_summary.csv\n")
    print(summary.round(4).to_string(index=False))

    # --- LaTeX table ---
    print("\n% --- Table 2b: paste into Section 4.2 ---")
    print(r"\begin{table}[t]\centering\caption{No-detail and best/mean-matched fixed-order "
          r"comparison on the synthetic battery.}\label{tab:table2b}")
    print(r"\begin{tabular}{llrrrr}\toprule")
    print(r"Degradation & Condition & PSNR (dB) & SSIM & $\Delta$ vs.\ adaptive & Holm $p$ \\\midrule")
    for _, r in summary.iterrows():
        delta = f"{r['delta_vs_adaptive']:+.2f}" if r['condition'] != 'adaptive' else '--'
        ph = f"{r['p_holm']:.3g}" if pd.notna(r['p_holm']) else '--'
        print(f"{r['degradation']} & {r['condition']} & {r['mean_PSNR']:.2f} $\\pm$ {r['std_PSNR']:.2f} "
              f"& {r['mean_SSIM']:.3f} & {delta} & {ph} \\\\")
    print(r"\bottomrule\end{tabular}\end{table}")


if __name__ == "__main__":
    main()
