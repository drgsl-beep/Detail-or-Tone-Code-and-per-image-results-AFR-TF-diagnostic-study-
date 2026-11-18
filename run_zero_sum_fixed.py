"""
run_zero_sum_fixed.py  --  Table 8 rows "Adaptive - fixed alpha=1, zero-sum kernel" (and the
as-implemented counterpart, as a cross-check against Table 2).

Same folder as pipeline.py and run_table2b_no_detail.py. Same battery, same distinct seeds as
the final run_zero_sum_control.py.

  python run_zero_sum_fixed.py --seed-mode distinct --out results_fixed
Writes zero_sum_fixed_per_image.csv and zero_sum_fixed_summary.csv (Holm over 3 degradations).
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

import run_table2b_no_detail as t
from pipeline import make_degraded, compute_metrics

NAMES = ["astronaut", "chelsea", "coffee", "rocket", "immunohistochemistry", "retina", "brick",
         "grass", "text", "page", "moon", "hubble_deep_field", "horse", "camera"]

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); a = np.zeros(len(p)); m = 0.0
    for k, i in enumerate(o):
        m = max(m, min(1.0, p[i] * (len(p) - k))); a[i] = m
    return a

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-mode", choices=["single", "distinct"], default="distinct")
    args = ap.parse_args()
    t.SEED, t.SEED_MODE = args.seed, args.seed_mode
    battery = t.load_battery(["skimage:" + n for n in NAMES])
    t._IDX.clear(); t._IDX.update({id(r): i for i, r in enumerate(battery.values())})
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for deg in t.DEGRADATIONS:
        for name, ref in battery.items():
            d = make_degraded(ref, deg, seed=t.case_seed(ref, deg))
            for tag, kern in (("original", _orig), ("zero_sum", _zero_sum)):
                pipeline.build_fractional_kernels = kern; t.build_fractional_kernels = kern
                out, _ = t.cond_adaptive(d)
                rows.append(dict(degradation=deg, image=name, condition=f"adaptive_{tag}", PSNR=compute_metrics(ref, out)["PSNR"]))
                out = t.cond_fixed(d, 1.0)
                rows.append(dict(degradation=deg, image=name, condition=f"fixed1_{tag}", PSNR=compute_metrics(ref, out)["PSNR"]))
        print("done", deg, flush=True)
    pipeline.build_fractional_kernels = _orig; t.build_fractional_kernels = _orig
    df = pd.DataFrame(rows); df.to_csv(os.path.join(args.out, "zero_sum_fixed_per_image.csv"), index=False)
    summ = []
    for tag in ("original", "zero_sum"):
        res = []
        for deg in t.DEGRADATIONS:
            p = df[df.degradation == deg].pivot(index="image", columns="condition", values="PSNR")
            x = (p[f"adaptive_{tag}"] - p[f"fixed1_{tag}"]).values; n = len(x)
            se = x.std(ddof=1) / np.sqrt(n); h = stats.t.ppf(.975, n - 1) * se
            res.append(dict(kernel=tag, degradation=deg, diff_dB=x.mean(), ci_lo=x.mean() - h, ci_hi=x.mean() + h,
                            p_t=stats.ttest_1samp(x, 0).pvalue, wins=int((x > 0).sum()), n=n))
        r = pd.DataFrame(res); r["holm_t"] = holm(r.p_t); summ.append(r)
    s = pd.concat(summ); s.to_csv(os.path.join(args.out, "zero_sum_fixed_summary.csv"), index=False)
    print(s.round(4).to_string(index=False))

if __name__ == "__main__":
    main()
