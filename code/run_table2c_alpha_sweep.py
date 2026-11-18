"""
run_table2c_alpha_sweep.py — Task 1.2 (Table 2c, Fig. 2b)

Sweeps fixed alpha over {0.7, 0.8, ..., 1.6} for each of the three
degradation types on the confirmed 14-image synthetic battery, and
computes the mean-matched fixed-order result per image. Built on
pipeline.py so it shares branch/fusion code exactly with
run_table2b_no_detail.py.

Results produced with this script are in ../results (see README.md).
Fill in BATTERY below (or pass --battery), then run it and insert
table2c_sweep.csv's numbers into Table 2c and fig2b_alpha_sweep.pdf as
Fig. 2b, both in Section 4.2.

HOW TO RUN
----------
    python run_table2c_alpha_sweep.py --battery battery.txt --out results/
"""
import argparse
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from pipeline import (
    make_degraded, haze_confidence, retinex_decompose, illumination_gain,
    estimate_transmission, weight_map, pyramid_fuse,
    adaptive_alpha_map, fractional_detail_layer, build_fractional_kernels,
    compute_metrics,
)

EPS = 1e-6

# Battery image list (default: the 14 scikit-image images of the paper) (see run_table2b_no_detail.py)
BATTERY = []

DEGRADATIONS = ["haze", "underwater", "lowlight"]
FIXED_ALPHAS = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
ALPHA_MIN, ALPHA_MAX = 0.7, 1.6
SEED = 0  # Degradation seeds (the paper uses --seed-mode distinct)


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

def _branch_fusion(img_f, levels=4, conf_thresh=0.35):
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
    return np.clip(base * (1.0 + detail_strength * detail[..., None]), 0, 1)


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
    return cond_fixed(img_f, float(amap.mean()), detail_strength), float(amap.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default=None)
    ap.add_argument("--out", default=".")
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
    adaptive_level = {}
    mean_matched_level = {}
    for deg in DEGRADATIONS:
        psnr_adapt, psnr_mm = [], []
        for ref in battery.values():
            degraded = make_degraded(ref, deg, seed=case_seed(ref, deg))
            out_a = cond_adaptive(degraded)
            psnr_adapt.append(compute_metrics(ref, out_a)['PSNR'])
            out_mm, _ = cond_mean_matched(degraded)
            psnr_mm.append(compute_metrics(ref, out_mm)['PSNR'])
        adaptive_level[deg] = np.mean(psnr_adapt)
        mean_matched_level[deg] = np.mean(psnr_mm)

        for a in FIXED_ALPHAS:
            psnrs, ssims = [], []
            for ref in battery.values():
                degraded = make_degraded(ref, deg, seed=case_seed(ref, deg))
                out = cond_fixed(degraded, a)
                m = compute_metrics(ref, out)
                psnrs.append(m['PSNR'])
                ssims.append(m['SSIM'])
            rows.append(dict(degradation=deg, alpha=a, mean_PSNR=np.mean(psnrs),
                              std_PSNR=np.std(psnrs, ddof=1), mean_SSIM=np.mean(ssims)))
        print(f"{deg}: adaptive={adaptive_level[deg]:.3f} dB, mean_matched={mean_matched_level[deg]:.3f} dB")

    sweep = pd.DataFrame(rows)
    sweep.to_csv(os.path.join(args.out, "table2c_sweep.csv"), index=False)
    print("\nWrote table2c_sweep.csv")

    best_rows = []
    for deg in DEGRADATIONS:
        sub = sweep[sweep.degradation == deg]
        best = sub.loc[sub.mean_PSNR.idxmax()]
        beats_best = adaptive_level[deg] > best.mean_PSNR
        print(f"{deg}: best fixed alpha = {best.alpha} ({best.mean_PSNR:.3f} dB); "
              f"adaptive {'BEATS' if beats_best else 'DOES NOT BEAT'} best fixed "
              f"({adaptive_level[deg]:.3f} dB)")
        best_rows.append(dict(degradation=deg, best_alpha=best.alpha, best_PSNR=best.mean_PSNR,
                               adaptive_PSNR=adaptive_level[deg], adaptive_beats_best=beats_best))
    pd.DataFrame(best_rows).to_csv(os.path.join(args.out, "table2c_best_summary.csv"), index=False)

    # --- Fig. 2b ---
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"haze": "tab:orange", "underwater": "tab:blue", "lowlight": "tab:green"}
    for deg in DEGRADATIONS:
        sub = sweep[sweep.degradation == deg]
        ax.plot(sub.alpha, sub.mean_PSNR, marker="o", label=f"{deg} (fixed sweep)", color=colors[deg])
        ax.axhline(adaptive_level[deg], linestyle="--", color=colors[deg], alpha=0.6,
                   label=f"{deg} (adaptive)")
        ax.axhline(mean_matched_level[deg], linestyle=":", color=colors[deg], alpha=0.6,
                   label=f"{deg} (mean-matched)")
    ax.set_xlabel("fixed \u03b1")
    ax.set_ylabel("mean PSNR (dB)")
    ax.set_title("Fig. 2b. PSNR vs. fixed order \u03b1, synthetic battery")
    ax.legend(fontsize=7, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig2b_alpha_sweep.pdf"), bbox_inches="tight")
    print(f"Wrote {os.path.join(args.out, 'fig2b_alpha_sweep.pdf')}")


if __name__ == "__main__":
    main()
