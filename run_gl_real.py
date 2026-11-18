"""
run_gl_real.py -- real-data test of the generality claim (Section 5.6): adaptive versus fixed order with
the kernel as implemented and with standard 8-direction Grunwald-Letnikov (GL) stencils (K = 3 taps),
with and without the zero-sum correction. Wrapper around diagnose_and_fix_branchA.py (same pairing, MD5
split, 480 px resizing, metrics), like run_zero_sum_real.py. Seven variants on the same base image:

    current (unchanged)              adaptive order, kernel as implemented
    fixed alpha=1 (implemented)      fixed order 1, kernel as implemented
    no detail stage                  O = F
    GL8-K3 adaptive                  adaptive order, standard 8-direction GL stencil, K = 3
    GL8-K3 fixed alpha=1             fixed order 1, same stencil
    GL8-K3 zero-sum adaptive         adaptive order, same stencil with the centre tap zero-summed
    GL8-K3 zero-sum fixed alpha=1    fixed order 1, zero-summed stencil

USAGE (same folder as pipeline.py and diagnose_and_fix_branchA.py; add --no_lpips to save time):
    python run_gl_real.py --raw_dir <UIEB raw> --ref_dir <UIEB reference> --tag gl_uieb --no_lpips
    python run_gl_real.py --raw_dir <RESIDE hazy> --ref_dir <RESIDE clear> --tag gl_reside --no_lpips
    python run_gl_real.py --raw_dir <LOL-v2 Test\\Low> --ref_dir <LOL-v2 Test\\Normal> --tag gl_lol --no_lpips
(LOL-v2: the ORIGINAL Test\\Low and Test\\Normal folders.) Writes <tag>_tune_results.csv and
<tag>_test_results.csv. Then run summarise_gl_real.py <tag> for adaptive-minus-fixed statistics.
"""
import numpy as np
import pipeline as P
import diagnose_and_fix_branchA as D

_orig = P.build_fractional_kernels

def _gl8(alpha, K=3, zero=False, **kw):
    w = [1.0]
    for k in range(1, K):
        w.append(w[-1] * (k - 1 - float(alpha)) / k)
    r = K - 1
    m = np.zeros((2 * r + 1, 2 * r + 1))
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
        for k in range(K):
            m[r + k * dy, r + k * dx] += w[k]
    if zero:
        m[r, r] -= m.sum()
    return m

def _with(kernel, fn):
    def run(deg):
        P.build_fractional_kernels = kernel
        try:
            return fn(deg)
        finally:
            P.build_fractional_kernels = _orig
    return run

adaptive = lambda deg: P.afr_tf(deg)
fixed1 = lambda deg: P.baseline_gamma_fixed_fractional(deg, alpha=1.0)
gl = lambda a, **kw: _gl8(a)
gl0 = lambda a, **kw: _gl8(a, zero=True)

D.VARIANTS = {
    'current (unchanged)': adaptive,
    'fixed alpha=1 (implemented)': fixed1,
    'no detail stage': D.variant_no_detail_stage,
    'GL8-K3 adaptive': _with(gl, adaptive),
    'GL8-K3 fixed alpha=1': _with(gl, fixed1),
    'GL8-K3 zero-sum adaptive': _with(gl0, adaptive),
    'GL8-K3 zero-sum fixed alpha=1': _with(gl0, fixed1),
}

if __name__ == '__main__':
    D.main()
