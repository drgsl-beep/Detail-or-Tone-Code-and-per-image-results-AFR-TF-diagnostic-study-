"""
run_zero_sum_real.py — the zero-sum-kernel control on REAL data, using exactly the same
split, image loading, resizing and metrics as diagnose_and_fix_branchA.py.

It reruns your diagnostic with three variants:
    current (unchanged)          the pipeline exactly as it is
    no detail stage              afr_tf(detail_strength=0)
    zero-sum high-pass detail    same pipeline, but the fractional kernel's centre tap is
                                 adjusted so that the taps sum to zero (a true high-pass)

Use the same folders and caps as for the other real-data tables (for LOL-v2 the ORIGINAL Test\\Low and
Test\\Normal folders, because renamed copies change the MD5 split). Put this file next to diagnose_and_fix_branchA.py
and pipeline.py.

    python run_zero_sum_real.py --raw_dir raw-890 --ref_dir reference-890 --tag zs_uieb
    python run_zero_sum_real.py --raw_dir <RESIDE hazy folder> --ref_dir <RESIDE gt folder> --tag zs_reside
    python run_zero_sum_real.py --raw_dir <LOL-v2 Real_captured\\Test\\Low> --ref_dir <LOL-v2 Real_captured\\Test\\Normal> --tag zs_lol55

Outputs: <tag>_tune_results.csv and <tag>_test_results.csv (same columns as your originals).
"""
import pipeline as P
import diagnose_and_fix_branchA as D

_orig_kernel = P.build_fractional_kernels


def _zero_sum_kernel(alpha, *a, **kw):
    k = _orig_kernel(alpha, *a, **kw).copy()
    k[1, 1] -= k.sum()
    return k


def variant_zero_sum(deg):
    P.build_fractional_kernels = _zero_sum_kernel
    try:
        return P.afr_tf(deg)
    finally:
        P.build_fractional_kernels = _orig_kernel


D.VARIANTS = {
    'current (unchanged)': D.variant_current,
    'no detail stage': D.variant_no_detail_stage,
    'zero-sum high-pass detail': variant_zero_sum,
}

if __name__ == '__main__':
    D.main()
