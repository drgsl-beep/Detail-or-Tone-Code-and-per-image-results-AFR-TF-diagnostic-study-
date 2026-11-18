"""
run_canonical_tables.py — Task 3.1: regenerate Tables 1 and 3 from ONE run.

Why: the AFR-TF haze mean appears as 14.55 dB (Table 1), 14.64 dB (Table 3) and
14.40 dB (Table 4) because those tables came from different runs. This script
produces Tables 1 and 3 from a single configuration and seed set, calling
pipeline.py's own functions directly (afr_tf, baseline_*), so every AFR-TF mean
is traceable to this run.

  Table 1: Degraded input, CLAHE, MSR, DCP-Dehaze, Fixed-order (alpha=1), AFR-TF,
           PSNR and SSIM, mean +/- std, for haze / underwater / low-light.
  Table 3: fusion strategy ablation: afr_tf(fusion_mode = 'weighted' | 'hard' | 'conf_scalar'),
           with paired t-test, Wilcoxon and Holm correction (within the table) vs 'weighted'.

NOT covered (needs files that are not in pipeline.py):
  Table 4 (AOD-Net / Zero-DCE / FUnIE-GAN comparison) needs the trained checkpoints and
  your PyTorch inference code. Run that comparison on the SAME battery and seed, then paste
  those numbers next to this script's AFR-TF column. Do not mix runs.

Run with the official 14-image battery (see README.md).
The official battery is the default; use --seed-mode distinct for the seeds of the paper.

USAGE
-----
    python run_canonical_tables.py --battery battery.txt --seed 0 --out results
    # --seed-mode distinct  uses a different seed for every image/degradation
    #                       (seed = seed + 10*image_index + degradation_index).
    #                       The manuscript says "a distinct random seed"; the exact original
    #                       rule is unknown, so this is an assumption, not a reproduction.
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from pipeline import (make_degraded, afr_tf, baseline_clahe, baseline_msr,
                      baseline_dcp_dehaze, baseline_gamma_fixed_fractional, compute_metrics)
from run_table2b_no_detail import load_battery, DEGRADATIONS

BATTERY = []  # (or pass --battery)


def _out(x):
    return x[0] if isinstance(x, tuple) else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-mode", choices=["single", "distinct"], default="single")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    paths = [l.strip() for l in open(args.battery) if l.strip() and not l.startswith("#")] if args.battery else BATTERY
    battery = load_battery(paths)
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for di, deg in enumerate(DEGRADATIONS):
        for ii, (name, ref) in enumerate(battery.items()):
            seed = args.seed if args.seed_mode == "single" else args.seed + 10 * ii + di
            d = make_degraded(ref, deg, seed=seed)
            afr = _out(afr_tf(d, fusion_mode="weighted"))
            outputs = {
                ("T1", "Degraded input"): d,
                ("T1", "CLAHE"): baseline_clahe(d),
                ("T1", "MSR"): baseline_msr(d),
                ("T1", "DCP-Dehaze"): baseline_dcp_dehaze(d),
                ("T1", "Fixed-order (alpha=1)"): baseline_gamma_fixed_fractional(d, alpha=1.0),
                ("T1", "Proposed AFR-TF"): afr,
                ("T3", "weighted"): afr,
                ("T3", "hard"): _out(afr_tf(d, fusion_mode="hard")),
                ("T3", "conf_scalar"): _out(afr_tf(d, fusion_mode="conf_scalar")),
            }
            for (tbl, method), img in outputs.items():
                m = compute_metrics(ref, img)
                rows.append(dict(table=tbl, degradation=deg, method=method, image=name,
                                 PSNR=float(m["PSNR"]), SSIM=float(m["SSIM"]), seed=seed))
        print("done", deg, flush=True)

    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(args.out, "canonical_per_image.csv"), index=False)

    summ = []
    for (tbl, deg, method), g in per.groupby(["table", "degradation", "method"], sort=False):
        summ.append(dict(table=tbl, degradation=deg, method=method,
                         mean_PSNR=g.PSNR.mean(), std_PSNR=g.PSNR.std(ddof=1),
                         mean_SSIM=g.SSIM.mean(), std_SSIM=g.SSIM.std(ddof=1)))
    summ = pd.DataFrame(summ)

    stat_rows = []
    for deg in DEGRADATIONS:
        sub = per[(per.table == "T3") & (per.degradation == deg)].pivot(index="image", columns="method", values="PSNR")
        for other in ["hard", "conf_scalar"]:
            diff = sub["weighted"] - sub[other]
            try:
                pw = stats.wilcoxon(sub["weighted"], sub[other]).pvalue
            except ValueError:
                pw = np.nan
            pt = stats.ttest_rel(sub["weighted"], sub[other]).pvalue
            stat_rows.append(dict(table="T3", degradation=deg, method=other,
                                  delta_weighted_minus_method=diff.mean(), p_t=pt, p_wilcoxon=pw))
    st = pd.DataFrame(stat_rows)
    st["p_holm"] = multipletests(st.p_t.fillna(1.0).values, method="holm")[1]
    summ = summ.merge(st, on=["table", "degradation", "method"], how="left")
    summ.to_csv(os.path.join(args.out, "canonical_tables.csv"), index=False)

    print("\n=== Table 1 (PSNR mean +/- std | SSIM mean) ===")
    for deg in DEGRADATIONS:
        print(f"-- {deg}")
        for _, r in summ[(summ.table == "T1") & (summ.degradation == deg)].iterrows():
            print(f"   {r.method:24s} {r.mean_PSNR:6.2f} +/- {r.std_PSNR:4.2f} | {r.mean_SSIM:.3f}")
    print("\n=== Table 3 (fusion strategy; Holm within table) ===")
    for deg in DEGRADATIONS:
        print(f"-- {deg}")
        for _, r in summ[(summ.table == "T3") & (summ.degradation == deg)].iterrows():
            extra = "" if r.method == "weighted" else (
                f"  delta(weighted-this)={r.delta_weighted_minus_method:+.2f} dB  p_t={r.p_t:.3g}  holm={r.p_holm:.3g}")
            print(f"   {r.method:12s} {r.mean_PSNR:6.2f} +/- {r.std_PSNR:4.2f}{extra}")
    print(f"\nInsert in Section 3.1: \"Tables 1 and 3 were regenerated from a single canonical run "
          f"(seed mode: {args.seed_mode}, base seed {args.seed}, n = {len(battery)} images); all AFR-TF means "
          f"in this paper are drawn from that run.\"")
    print("Wrote canonical_tables.csv and canonical_per_image.csv in", args.out)


if __name__ == "__main__":
    main()
