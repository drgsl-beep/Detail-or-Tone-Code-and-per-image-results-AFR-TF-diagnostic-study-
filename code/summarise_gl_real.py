"""summarise_gl_real.py <tag> : adaptive minus fixed alpha=1 (and minus no stage) on the held-out file <tag>_test_results.csv,
for each kernel, with 95% CI, wins and a Holm correction over the kernel comparisons shown (3 adaptive-minus-fixed rows)."""
import sys, numpy as np, pandas as pd
from scipy import stats
tag = sys.argv[1]
d = pd.read_csv(f'{tag}_test_results.csv')
P = d.pivot(index='image', columns='variant', values='PSNR')
rows = []
for lab, a, b in [('implemented', 'current (unchanged)', 'fixed alpha=1 (implemented)'),
                  ('GL8-K3', 'GL8-K3 adaptive', 'GL8-K3 fixed alpha=1'),
                  ('GL8-K3 zero-sum', 'GL8-K3 zero-sum adaptive', 'GL8-K3 zero-sum fixed alpha=1')]:
    x = (P[a] - P[b]).values; n = len(x); se = x.std(ddof=1) / np.sqrt(n); h = stats.t.ppf(.975, n - 1) * se
    rows.append(dict(kernel=lab, comparison='adaptive - fixed', n=n, diff_dB=x.mean(), lo=x.mean() - h, hi=x.mean() + h,
                     p=stats.ttest_1samp(x, 0).pvalue, wins=int((x > 0).sum())))
r = pd.DataFrame(rows); p = r.p.values; o = np.argsort(p); a = np.zeros(len(p)); m = 0
for k, i in enumerate(o): m = max(m, min(1, p[i] * (len(p) - k))); a[i] = m
r['p_holm'] = a
print(r.round(4).to_string(index=False))
print('\nMean PSNR by variant:'); print(P.mean().round(3).to_string())
