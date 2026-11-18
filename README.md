# Supplementary material: "Detail or Tone? A Diagnostic Study of an Apparent Adaptive Fractional-Order Gain in Image Restoration"

Python 3.14.6; NumPy 2.5.3, SciPy 1.18.1, OpenCV 5.0.0, scikit-image 0.26.0, PyTorch 2.14.0, lpips 0.1.4, statsmodels 0.15.0, pandas 3.0.5, matplotlib 3.11.2. Run scripts from the `code/` folder.

## What produced what

| Manuscript item | Script (code/) | Result files (results/) |
|---|---|---|
| Tables 1 and 4 (synthetic battery, fusion ablation) | run_canonical_tables.py --seed-mode distinct | synthetic/canonical_per_image.csv, canonical_tables.csv |
| Tables 2 and 3, Fig. 3 (adaptive vs fixed, sweep) | run_table2b_no_detail.py, run_table2c_alpha_sweep.py --seed-mode distinct | Table 2 is derived from synthetic/canonical_per_image.csv; synthetic/table2c_sweep.csv, table2c_best_summary.csv |
| Table 8 (synthetic controls) | run_zero_sum_control.py, run_zero_sum_fixed.py (both --seed-mode distinct) | synthetic/zero_sum_control_*.csv, zero_sum_fixed_*.csv |
| Section 5.1, Table 7, Fig. 4 | check_normalised_D.py (and the tap sums of Eq. 16) | printed output |
| Section 5.6, Tables 10 and 11 (synthetic) | kernel_audit.py, stage_on_other_bases.py | synthetic/novelty_expB_kernels.csv |
| Section 5.6, Table 12 (real sets, held-out) | run_gl_real.py, then summarise_gl_real.py <tag> | real/gl_lol_*, gl_uieb_*, gl_reside_*_results.csv |
| Section 5.7, Table 13 | stage_on_other_bases.py | synthetic/novelty_expB_bases.csv |
| Section 5.4, Fig. 6 | brightness_offset.py | printed output |
| Table 5 (UIEB, RESIDE-SOTS, n = 500) | run_large_scale_eval.py | real/results_UIEB.csv, results_RESIDE-SOTS-outdoor.csv |
| Tables 6, 9, S1: current pipeline, Branch A disabled, gate 0.7, no detail | diagnose_and_fix_branchA.py (UIEB, RESIDE-SOTS, LOL-v2) | real/uieb_*_results_variants.csv, reside_*_results.csv, lol_*_results.csv |
| Tables 6, 9, S1: zero-sum (true high-pass) rows | run_zero_sum_real.py (wrapper around diagnose_and_fix_branchA.py that replaces the kernel by the zero-sum kernel, Eq. 17) | real/zs_*_results.csv |

## Seeds and splits
- Synthetic cases: seed = 10*i + j (i = 0-based image index in the order listed in Section 3.1, j = 0-based index in haze, underwater, lowlight).
- Real sets: MD5 of the file's base name including extension, hexadecimal digest as an integer, modulo 100; below 60 is tune, the rest held-out; then the first 150 images in sorted file-name order of each set. For LOL-v2 use the original Low and Normal folders (the hash is taken of names such as low00690.png).

## Notes
- results/real/lol_*_results.csv and reside_*_results.csv contain the four variants (current, Branch A disabled, gate 0.7, no detail); zs_*_results.csv contain current, no detail and the zero-sum variant. The LOL-v2 files used in the paper are the 55/45 split.
- The files in results/synthetic are those of the final run (distinct seeds). All numbers in the manuscript tables were recomputed from them.
