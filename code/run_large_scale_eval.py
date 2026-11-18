"""
Generic large-scale benchmark evaluation script.

Works for ANY paired dataset (degraded/A images + ground-truth images with
matching filenames in two folders), so the same script handles RESIDE SOTS,
LOL-v2 (real or synthetic), and the full EUVP sample -- just point it at the
right folders. This mirrors the exact logic already validated on the 23-pair
EUVP subset (validate_real_underwater.m / run_real_underwater.py), scaled to
handle arbitrarily many images and named per-dataset for clean reporting.

USAGE:
    python run_large_scale_eval.py --degraded_dir data/RESIDE_SOTS/hazy \
                                    --gt_dir data/RESIDE_SOTS/clear \
                                    --dataset_name RESIDE-SOTS \
                                    --degradation_type haze

    python run_large_scale_eval.py --degraded_dir data/LOLv2_real_test/low \
                                    --gt_dir data/LOLv2_real_test/high \
                                    --dataset_name LOLv2-real \
                                    --degradation_type lowlight

    python run_large_scale_eval.py --degraded_dir data/EUVP_sample500/A \
                                    --gt_dir data/EUVP_sample500/GT \
                                    --dataset_name EUVP-full-sample500 \
                                    --degradation_type underwater
"""
import argparse
import glob
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
import pandas as pd
from scipy import stats

import pipeline as P

try:
    import lpips
    import torch
    _LPIPS_AVAILABLE = True
    _lpips_model = None
except ImportError:
    _LPIPS_AVAILABLE = False


def get_lpips_model():
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net='alex')
        _lpips_model.eval()
    return _lpips_model


def compute_lpips(ref_f, test_f):
    """ref_f, test_f: HxWx3 float64 in [0,1]. Returns None if lpips not installed."""
    if not _LPIPS_AVAILABLE:
        return None
    model = get_lpips_model()
    def to_tensor(x):
        t = torch.from_numpy(x.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        return t * 2 - 1  # LPIPS expects [-1, 1]
    with torch.no_grad():
        d = model(to_tensor(ref_f), to_tensor(test_f))
    return float(d.item())


def match_files(degraded_dir, gt_dir):
    """Match degraded/GT files by filename, with a fallback for RESIDE-style
    naming where the hazy file has extra parameters appended, e.g.
    hazy: 1400_0.9_0.16.jpg  <->  clear: 1400.png (same leading number)."""
    deg_files = sorted(glob.glob(os.path.join(degraded_dir, '*')))
    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*')))
    gt_by_exact = {os.path.splitext(os.path.basename(g))[0]: g for g in gt_files}
    # RESIDE-style: index GT files by their basename with no extension (already
    # covers the simple case); additionally index degraded files' leading
    # numeric/underscore-delimited prefix for the fallback match.
    pairs = []
    unmatched = []
    for df in deg_files:
        base = os.path.splitext(os.path.basename(df))[0]
        if base in gt_by_exact:
            pairs.append((df, gt_by_exact[base]))
            continue
        # fallback: take everything before the first underscore
        prefix = base.split('_')[0]
        if prefix in gt_by_exact:
            pairs.append((df, gt_by_exact[prefix]))
        else:
            unmatched.append(df)
    if unmatched:
        print(f'NOTE: {len(unmatched)} degraded files had no matching GT file '
              f'(tried exact name and underscore-prefix match). Example: '
              f'{os.path.basename(unmatched[0])}')
    return pairs


def load_img(path, max_side=640):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--degraded_dir', required=True)
    ap.add_argument('--gt_dir', required=True)
    ap.add_argument('--dataset_name', required=True)
    ap.add_argument('--degradation_type', required=True, choices=['haze', 'underwater', 'lowlight'])
    ap.add_argument('--max_images', type=int, default=None, help='cap for a quick test run')
    ap.add_argument('--out_csv', default=None)
    args = ap.parse_args()

    pairs = match_files(args.degraded_dir, args.gt_dir)
    if args.max_images:
        pairs = pairs[:args.max_images]
    print(f'{args.dataset_name}: {len(pairs)} matched pairs found')
    if len(pairs) == 0:
        print('\nERROR: No matching degraded/GT pairs found. Nothing to evaluate.')
        print(f'Checked --degraded_dir "{args.degraded_dir}" against --gt_dir "{args.gt_dir}".')
        print('Run: dir "%s" and dir "%s"' % (args.degraded_dir, args.gt_dir))
        print('and check the filenames actually correspond to each other, then try again.')
        return
    if not _LPIPS_AVAILABLE:
        print('NOTE: lpips not installed -- run `pip install lpips torch` to include it. Continuing without it.')

    methods = {
        'Degraded (input)': lambda x: x,
        'CLAHE': P.baseline_clahe,
        'MSR': P.baseline_msr,
        'DCP-Dehaze': P.baseline_dcp_dehaze,
        'Fixed-order Fractional (ablation)': lambda x: P.baseline_gamma_fixed_fractional(x, 1.0),
        'Proposed AFR-TF': P.afr_tf,
    }

    rows = []
    for i, (dpath, gpath) in enumerate(pairs):
        deg = load_img(dpath)
        gt = load_img(gpath)
        if deg is None or gt is None:
            continue
        if deg.shape != gt.shape:
            gt = cv2.resize(gt, (deg.shape[1], deg.shape[0]))

        for mname, fn in methods.items():
            out = fn(deg)
            m = P.compute_metrics(gt, out)
            lp = compute_lpips(gt, out)
            rows.append(dict(dataset=args.dataset_name, image=os.path.basename(dpath),
                              method=mname, PSNR=m['PSNR'], SSIM=m['SSIM'], LPIPS=lp))
        if (i + 1) % 25 == 0:
            print(f'  {i+1}/{len(pairs)} done', flush=True)

    df = pd.DataFrame(rows)
    out_csv = args.out_csv or f'results_{args.dataset_name}.csv'
    df.to_csv(out_csv, index=False)
    print(f'\nSaved: {out_csv}')

    print('\n=== Summary (mean +/- std) ===')
    print(df.groupby('method')[['PSNR', 'SSIM'] + (['LPIPS'] if _LPIPS_AVAILABLE else [])].agg(['mean', 'std']).round(3))

    print('\n=== Core ablation: Proposed vs Fixed-order (paired) ===')
    prop = df[df.method == 'Proposed AFR-TF'].set_index('image')['PSNR']
    fixed = df[df.method == 'Fixed-order Fractional (ablation)'].set_index('image')['PSNR']
    common = prop.index.intersection(fixed.index)
    diff = prop.loc[common] - fixed.loc[common]
    t, p = stats.ttest_rel(prop.loc[common], fixed.loc[common])
    print(f'n={len(common)}  mean_diff={diff.mean():+.3f}dB  p={p:.5f}  wins={(diff>0).sum()}/{len(common)}')
    print(f'\nCompare this p-value and effect size against the n=14 synthetic result and the')
    print(f'n=23 EUVP-subset result already in the manuscript to see whether the larger,')
    print(f'more realistic sample changes the conclusion.')


if __name__ == '__main__':
    main()
