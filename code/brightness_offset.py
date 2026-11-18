"""
brightness_offset.py -- Section 5.4 / Fig. 6: association between the brightness offset of the
degraded input (mean(degraded) - mean(reference)) and the effect of the detail stage as implemented
(PSNR of adaptive_original minus PSNR of no_detail), over the 42 synthetic cases.

Needs results/synthetic/zero_sum_control_per_image.csv (from run_zero_sum_control.py with
--seed-mode distinct). Run from the folder that holds pipeline.py and run_table2b_no_detail.py:
    python brightness_offset.py --csv ../results/synthetic/zero_sum_control_per_image.csv
"""
import argparse, numpy as np, pandas as pd
from scipy import stats
import pipeline as P, run_table2b_no_detail as t

NAMES = ["astronaut","chelsea","coffee","rocket","immunohistochemistry","retina","brick",
         "grass","text","page","moon","hubble_deep_field","horse","camera"]

ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); a = ap.parse_args()
t.SEED, t.SEED_MODE = 0, "distinct"
bat = t.load_battery(["skimage:" + n for n in NAMES])
t._IDX.clear(); t._IDX.update({id(r): i for i, r in enumerate(bat.values())})
z = pd.read_csv(a.csv).pivot_table(index=["degradation", "image"], columns="condition", values="PSNR")
rows = []
for deg in t.DEGRADATIONS:
    for n, ref in bat.items():
        d = P.make_degraded(ref, deg, seed=t.case_seed(ref, deg))
        rows.append((deg, n, d.mean() - ref.mean(), z.loc[(deg, n), "adaptive_original"] - z.loc[(deg, n), "no_detail"]))
df = pd.DataFrame(rows, columns=["degradation", "image", "offset", "effect"])
r = stats.spearmanr(df.offset, df.effect)
print(f"ALL n={len(df)}  rho={r.statistic:.3f}  p={r.pvalue:.4f}")
for d, g in df.groupby("degradation"):
    r = stats.spearmanr(g.offset, g.effect)
    print(f"{d:11s} rho={r.statistic:.3f}  p={r.pvalue:.4g}  mean offset={g.offset.mean():+.3f}")
