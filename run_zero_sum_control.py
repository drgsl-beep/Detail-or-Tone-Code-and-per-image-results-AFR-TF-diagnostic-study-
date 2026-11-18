"""
run_zero_sum_control.py — does the 'detail stage' work as DETAIL enhancement, or as a tone curve?

pipeline.py's fractional kernel (build_fractional_kernels) does NOT sum to zero
(taps sum to about -2.0 at alpha=0.7, -3.0 at alpha=1.0, -4.4 at alpha=1.6), so it is not
a true high-pass filter: its output D is >95% correlated with image brightness, and
O = F * (1 + 0.45 * D/max|D|) behaves mostly like a brightness-dependent darkening.

This script compares, per degradation, on the same images and seed:
    adaptive_original   the pipeline exactly as it is
    adaptive_zero_sum   same pipeline, kernel centre tap adjusted so the taps sum to 0
    no_detail           detail stage removed (O = F)
Expected if the hypothesis is right: adaptive_zero_sum ~ no_detail, and both differ from
adaptive_original in a direction that depends on image brightness.

USAGE (same folder as pipeline.py and run_table2b_no_detail.py):
    python run_zero_sum_control.py --battery battery.txt --out results
"""
import argparse, os
import numpy as np, pandas as pd
from scipy import stats
import pipeline

_orig = pipeline.build_fractional_kernels
def _zero_sum(alpha, *a, **kw):
    k = _orig(alpha, *a, **kw).copy()
    k[1, 1] -= k.sum()
    return k

import run_table2b_no_detail as t          # uses the ORIGINAL kernel when imported unpatched
from pipeline import make_degraded, compute_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default=None)
    ap.add_argument("--out", default=".")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-mode", choices=["single", "distinct"], default="single")
    args = ap.parse_args()
    t.SEED, t.SEED_MODE = args.seed, args.seed_mode
    paths = [l.strip() for l in open(args.battery) if l.strip()] if args.battery else []
    battery = t.load_battery(paths)
    t._IDX.clear(); t._IDX.update({id(r): i for i, r in enumerate(battery.values())})
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for deg in t.DEGRADATIONS:
        for name, ref in battery.items():
            d = make_degraded(ref, deg, seed=t.case_seed(ref, deg))
            # original kernel
            pipeline.build_fractional_kernels = _orig
            t.build_fractional_kernels = _orig
            out, _ = t.cond_adaptive(d)
            rows.append(dict(degradation=deg, image=name, condition="adaptive_original", PSNR=compute_metrics(ref, out)['PSNR']))
            # zero-sum kernel
            pipeline.build_fractional_kernels = _zero_sum
            t.build_fractional_kernels = _zero_sum
            out, _ = t.cond_adaptive(d)
            rows.append(dict(degradation=deg, image=name, condition="adaptive_zero_sum", PSNR=compute_metrics(ref, out)['PSNR']))
            # no detail
            out = t.cond_no_detail(d)
            rows.append(dict(degradation=deg, image=name, condition="no_detail", PSNR=compute_metrics(ref, out)['PSNR']))
        print("done", deg, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "zero_sum_control_per_image.csv"), index=False)
    summ = []
    for deg in t.DEGRADATIONS:
        piv = df[df.degradation == deg].pivot(index="image", columns="condition", values="PSNR")
        for a, b in [("adaptive_zero_sum", "no_detail"), ("adaptive_zero_sum", "adaptive_original"), ("adaptive_original", "no_detail")]:
            diff = piv[a] - piv[b]
            summ.append(dict(degradation=deg, comparison=f"{a} - {b}", mean_diff_dB=diff.mean(),
                              p_t=stats.ttest_rel(piv[a], piv[b]).pvalue, n=len(diff)))
        print(deg, {c: round(piv[c].mean(), 2) for c in piv.columns})
    s = pd.DataFrame(summ)
    s.to_csv(os.path.join(args.out, "zero_sum_control_summary.csv"), index=False)
    print(s.round(4).to_string(index=False))

if __name__ == "__main__":
    main()
