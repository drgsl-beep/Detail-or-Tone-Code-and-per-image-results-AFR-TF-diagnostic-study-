"""
Principled diagnosis + candidate fix + HELD-OUT validation for the real-data
gap found on UIEB (Proposed AFR-TF underperforms doing nothing by -1.94dB).

METHODOLOGY (this matters -- read before running):
  1. UIEB's 500 images are split deterministically into a TUNE set (60%) and
     a held-out TEST set (40%), by hashing the filename (same split every run).
  2. All diagnosis and candidate-fix selection happens ONLY on the TUNE set.
  3. Exactly one candidate is chosen based on TUNE-set performance.
  4. That chosen candidate is then evaluated ONCE on the held-out TEST set
     AND on the fully separate EUVP set (if you have it), with no further
     adjustment. This is what makes an improvement real rather than curve-
     fitting: if it doesn't hold on data it never influenced, it isn't real,
     and this script will tell you so rather than hide it.

USAGE:
    python diagnose_and_fix_branchA.py --raw_dir UIEB_raw --ref_dir UIEB_reference
    (add --euvp_a_dir / --euvp_gt_dir if you still have the EUVP folders, to
    also validate on that fully independent set)

    Add --no_lpips to skip LPIPS entirely (e.g. if you have no internet
    access to fetch the AlexNet backbone weights on first run, or lpips
    isn't installed and you don't want to install it right now).

FILENAME MATCHING:
    match_pairs() tries three strategies, in order, for each raw file:
      1. Exact stem match: "img001.png" <-> "img001.png"
      2. RESIDE-style prefix match: "hazy_0.9_0.16.png" -> prefix "hazy"
         matched against a ref file named exactly "hazy.png"
      3. Trailing-numeric-ID match: "low00690.png" <-> "normal00690.png"
         (different word prefix, same trailing digit run) -- this is what
         LOLv2's Real_captured Low/Normal folders use.
    Any raw file that still can't be matched after all three strategies is
    reported by name via a WARNING instead of being silently dropped.

LPIPS (new):
    Computed in-memory, in the same loop that already computes PSNR/SSIM --
    there is no need to save restored images to disk first. Requires
    `pip install lpips torch`. Uses the AlexNet backbone (net='alex'), which
    downloads a small set of ImageNet-pretrained weights the first time it
    runs, so you need normal internet access on that first run. Runs on GPU
    automatically if available, otherwise CPU (slower, but fine for a
    battery of this size).

    Because LPIPS is "lower is better" (unlike PSNR/SSIM), all "vs_degraded"
    comparisons for LPIPS are computed as (degraded_LPIPS - LPIPS), so that,
    consistent with the PSNR convention already used in this script, a
    positive number always means "the candidate is better than doing
    nothing."

    The script reports, for the TUNE set, which candidate is best by PSNR
    and which is best by LPIPS, and states plainly whether they agree --
    this is the "does LPIPS confirm or contradict PSNR" check requested
    alongside the metric itself.
"""
import argparse
import glob
import hashlib
import os
import re
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
import pandas as pd
from scipy import stats

import pipeline as P


# ---------------------------------------------------------------------
# LPIPS setup (optional; script still runs without it via --no_lpips)
# ---------------------------------------------------------------------

_LPIPS_MODEL = None
_LPIPS_DEVICE = None


def init_lpips():
    """Load the LPIPS AlexNet backbone once. Returns (model, device) or
    (None, None) if unavailable, printing exactly why."""
    global _LPIPS_MODEL, _LPIPS_DEVICE
    if _LPIPS_MODEL is not None:
        return _LPIPS_MODEL, _LPIPS_DEVICE
    try:
        import torch
        import lpips
    except ImportError as e:
        print(f'LPIPS unavailable: {e}. Install with: pip install lpips torch')
        return None, None
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = lpips.LPIPS(net='alex').to(device)
        model.eval()
        _LPIPS_MODEL, _LPIPS_DEVICE = model, device
        print(f'LPIPS (AlexNet) loaded OK on {device}.')
        return model, device
    except Exception as e:
        print(f'LPIPS unavailable: failed to initialize AlexNet backbone ({e}). '
              f'This is usually a network problem fetching the pretrained weights '
              f'on first run -- check your internet access and retry.')
        return None, None


def compute_lpips(model, device, gt, out):
    """gt, out: HWC float64 arrays in [0,1], RGB. Returns a Python float."""
    import torch

    def to_tensor(img):
        t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        return (t * 2 - 1).to(device)  # lpips expects [-1, 1]

    with torch.no_grad():
        d = model(to_tensor(gt), to_tensor(out))
    return float(d.item())


def load_img(path, max_side=480):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def deterministic_split(pairs, tune_frac=0.6):
    tune, test = [], []
    for pair in pairs:
        raw_path = pair[0]
        h = int(hashlib.md5(os.path.basename(raw_path).encode()).hexdigest(), 16)
        (tune if (h % 100) < tune_frac * 100 else test).append(pair)
    return tune, test


def _trailing_numeric_id(stem):
    """Extract a trailing run of digits from a filename stem, e.g.
    'low00690' -> '00690', 'normal00690' -> '00690'. Returns None if the
    stem doesn't end in digits."""
    m = re.search(r'(\d+)$', stem)
    return m.group(1) if m else None


def match_pairs(raw_dir, ref_dir):
    raw_files = sorted(glob.glob(os.path.join(raw_dir, '**', '*.*'), recursive=True))
    raw_files = [f for f in raw_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    ref_files = sorted(glob.glob(os.path.join(ref_dir, '**', '*.*'), recursive=True))
    ref_files = [f for f in ref_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    ref_by_exact = {os.path.splitext(os.path.basename(r))[0]: r for r in ref_files}

    # Build a numeric-ID index over the ref files too (fallback #3).
    # If two ref files share a trailing numeric ID, the first one found wins
    # and this is reported, since that indicates an ambiguous dataset layout.
    ref_by_numeric_id = {}
    dup_numeric_ids = set()
    for r in ref_files:
        base = os.path.splitext(os.path.basename(r))[0]
        nid = _trailing_numeric_id(base)
        if nid is None:
            continue
        if nid in ref_by_numeric_id:
            dup_numeric_ids.add(nid)
        else:
            ref_by_numeric_id[nid] = r

    pairs = []
    unmatched = []
    for rf in raw_files:
        base = os.path.splitext(os.path.basename(rf))[0]

        # Strategy 1: exact stem match
        if base in ref_by_exact:
            pairs.append((rf, ref_by_exact[base]))
            continue

        # Strategy 2: RESIDE-style prefix match (e.g. "hazy_0.9_0.16" -> "hazy")
        prefix = base.split('_')[0]
        if prefix in ref_by_exact:
            pairs.append((rf, ref_by_exact[prefix]))
            continue

        # Strategy 3: trailing numeric ID match (e.g. "low00690" <-> "normal00690")
        nid = _trailing_numeric_id(base)
        if nid is not None and nid in ref_by_numeric_id:
            pairs.append((rf, ref_by_numeric_id[nid]))
            continue

        unmatched.append(rf)

    if dup_numeric_ids:
        print(f'WARNING: {len(dup_numeric_ids)} numeric ID(s) appear on more than one '
              f'ref file; only the first was used for each (IDs: '
              f'{sorted(dup_numeric_ids)[:5]}{"..." if len(dup_numeric_ids) > 5 else ""})')

    if unmatched:
        print(f'WARNING: {len(unmatched)} raw image(s) had no matching ref image '
              f'after all matching strategies (examples: '
              f'{[os.path.basename(f) for f in unmatched[:5]]}'
              f'{"..." if len(unmatched) > 5 else ""})')

    return pairs


# ---------------------------------------------------------------------
# Candidate variants of afr_tf to test (all applied only on the TUNE set)
# ---------------------------------------------------------------------

def variant_current(deg):
    return P.afr_tf(deg)

def variant_no_branchA_ever(deg):
    """Hard-disable Branch A entirely, regardless of confidence -- rely only
    on Branch B + adaptive detail. Tests whether Branch A is net-harmful on
    real underwater images even when 'correctly' triggered by high confidence."""
    return P.afr_tf(deg, conf_thresh=1.01)  # threshold above 1 => always below it => always bypassed

def variant_higher_gate(deg):
    """Raise the confidence threshold substantially (0.35 -> 0.7), so Branch A
    only participates when very confident, rather than moderately confident."""
    return P.afr_tf(deg, conf_thresh=0.7)

def variant_no_detail_stage(deg):
    """Disable the adaptive fractional-order detail stage (detail_strength=0),
    testing whether the detail stage specifically (not the branches/fusion)
    is responsible for the real-data loss."""
    return P.afr_tf(deg, detail_strength=0.0)


VARIANTS = {
    'current (unchanged)': variant_current,
    'Branch A always disabled': variant_no_branchA_ever,
    'higher confidence gate (0.7)': variant_higher_gate,
    'no detail stage': variant_no_detail_stage,
}


def evaluate(pairs, label, lpips_model=None, lpips_device=None):
    rows = []
    for i, (rp, gp) in enumerate(pairs):
        deg = load_img(rp)
        gt = load_img(gp)
        if deg is None or gt is None:
            continue
        if deg.shape != gt.shape:
            gt = cv2.resize(gt, (deg.shape[1], deg.shape[0]))
        deg_psnr = P.compute_metrics(gt, deg)['PSNR']
        deg_lpips = (compute_lpips(lpips_model, lpips_device, gt, deg)
                     if lpips_model is not None else np.nan)
        for vname, vfn in VARIANTS.items():
            out = vfn(deg)
            m = P.compute_metrics(gt, out)
            row = dict(split=label, image=os.path.basename(rp), variant=vname,
                       PSNR=m['PSNR'], degraded_PSNR=deg_psnr)
            if lpips_model is not None:
                row['LPIPS'] = compute_lpips(lpips_model, lpips_device, gt, out)
                row['degraded_LPIPS'] = deg_lpips
            rows.append(row)
        if (i + 1) % 25 == 0:
            print(f'  [{label}] {i+1}/{len(pairs)} done', flush=True)
    return pd.DataFrame(rows)


def summarize(df, has_lpips):
    """PSNR (and, if available, LPIPS) summary by variant, matching the
    existing 'vs_degraded' convention: positive = better than doing nothing,
    for BOTH metrics (LPIPS is inverted internally to make this true)."""
    summary = df.groupby('variant')['PSNR'].agg(['mean', 'std']).round(3)
    summary['vs_degraded_PSNR'] = df.groupby('variant').apply(
        lambda g: (g['PSNR'] - g['degraded_PSNR']).mean())
    if has_lpips:
        lpips_summary = df.groupby('variant')['LPIPS'].agg(['mean', 'std']).round(4)
        lpips_summary.columns = ['LPIPS_mean', 'LPIPS_std']
        summary = summary.join(lpips_summary)
        # Inverted so positive = improvement, consistent with vs_degraded_PSNR
        summary['vs_degraded_LPIPS'] = df.groupby('variant').apply(
            lambda g: (g['degraded_LPIPS'] - g['LPIPS']).mean())
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', required=True)
    ap.add_argument('--ref_dir', required=True)
    ap.add_argument('--euvp_a_dir', default=None)
    ap.add_argument('--euvp_gt_dir', default=None)
    ap.add_argument('--tune_max', type=int, default=150, help='cap tune-set size for speed')
    ap.add_argument('--test_max', type=int, default=150, help='cap held-out size for speed')
    ap.add_argument('--tag', default='diagnosis', help='prefix for output CSVs, e.g. reside -> reside_tune_results.csv')
    ap.add_argument('--no_lpips', action='store_true', help='skip LPIPS entirely')
    args = ap.parse_args()

    lpips_model, lpips_device = (None, None) if args.no_lpips else init_lpips()
    has_lpips = lpips_model is not None

    pairs = match_pairs(args.raw_dir, args.ref_dir)
    print(f'Found {len(pairs)} total matched pairs')
    if len(pairs) == 0:
        print('\nERROR: No matching pairs found. Check --raw_dir and --ref_dir folder names')
        print('and confirm the filenames actually correspond to each other, then try again.')
        return
    tune_pairs, test_pairs = deterministic_split(pairs)
    tune_pairs, test_pairs = tune_pairs[:args.tune_max], test_pairs[:args.test_max]
    print(f'Tune set: {len(tune_pairs)} images (used to pick a candidate)')
    print(f'Held-out test set: {len(test_pairs)} images (used only to confirm)')

    print('\n=== Evaluating all candidates on TUNE set ===')
    df_tune = evaluate(tune_pairs, 'tune', lpips_model, lpips_device)
    df_tune.to_csv(f'{args.tag}_tune_results.csv', index=False)

    summary = summarize(df_tune, has_lpips)
    print('\nTUNE set summary:')
    print(summary.to_string())

    best_variant_psnr = summary['vs_degraded_PSNR'].idxmax()
    print(f'\nBest candidate on TUNE set by PSNR: "{best_variant_psnr}" '
          f'(vs_degraded = {summary.loc[best_variant_psnr, "vs_degraded_PSNR"]:+.3f} dB)')

    if has_lpips:
        best_variant_lpips = summary['vs_degraded_LPIPS'].idxmax()
        print(f'Best candidate on TUNE set by LPIPS: "{best_variant_lpips}" '
              f'(vs_degraded = {summary.loc[best_variant_lpips, "vs_degraded_LPIPS"]:+.4f})')
        if best_variant_lpips == best_variant_psnr:
            print('  ==> PSNR and LPIPS AGREE on the best candidate.')
        else:
            print('  ==> PSNR and LPIPS DISAGREE on the best candidate. Report both '
                  'selections and let the held-out confirmation below decide which, '
                  'if either, holds up -- do not silently pick one.')

    # PSNR remains the metric used to select the candidate that goes on to
    # held-out confirmation, matching the rest of this manuscript's tables;
    # the LPIPS numbers above are reported alongside for comparison, not
    # substituted in as the selection criterion.
    best_variant = best_variant_psnr

    if best_variant == 'current (unchanged)':
        print('\nThe current method is already the best of the candidates tried on the')
        print('tune set. None of these specific architectural changes helps. This is a')
        print('valid, reportable outcome -- it narrows what the fix is NOT, even if it')
        print("doesn't hand us a fix.")
        return

    print(f'\n=== Confirming "{best_variant}" on HELD-OUT test set (never used to pick it) ===')
    df_test = evaluate(test_pairs, 'test', lpips_model, lpips_device)
    df_test.to_csv(f'{args.tag}_test_results.csv', index=False)

    cur = df_test[df_test.variant == 'current (unchanged)'].set_index('image')
    best = df_test[df_test.variant == best_variant].set_index('image')
    common = cur.index.intersection(best.index)

    diff = best.loc[common, 'PSNR'] - cur.loc[common, 'PSNR']
    t, p = stats.ttest_rel(best.loc[common, 'PSNR'], cur.loc[common, 'PSNR'])
    print(f'\nOn held-out test set: "{best_variant}" vs current (PSNR)')
    print(f'  mean_diff={diff.mean():+.3f}dB  p={p:.5f}  wins={(diff>0).sum()}/{len(common)}')
    if diff.mean() > 0 and p < 0.05:
        print('  ==> CONFIRMED on held-out data (PSNR). This is a real improvement, not overfitting.')
    else:
        print('  ==> NOT confirmed on held-out data (PSNR) (not significant or wrong direction).')
        print('      Report the TUNE-set finding as exploratory only -- do not claim it as a fix.')

    if has_lpips:
        # Inverted so positive = improvement (lower LPIPS is better)
        diff_l = cur.loc[common, 'LPIPS'] - best.loc[common, 'LPIPS']
        t_l, p_l = stats.ttest_rel(cur.loc[common, 'LPIPS'], best.loc[common, 'LPIPS'])
        try:
            w_stat, w_p = stats.wilcoxon(cur.loc[common, 'LPIPS'], best.loc[common, 'LPIPS'])
        except ValueError:
            w_stat, w_p = float('nan'), float('nan')
        print(f'\nOn held-out test set: "{best_variant}" vs current (LPIPS, positive = improvement)')
        print(f'  mean_diff={diff_l.mean():+.4f}  paired-t p={p_l:.5f}  '
              f'Wilcoxon p={w_p:.5f}  wins={(diff_l>0).sum()}/{len(common)}')
        if diff_l.mean() > 0 and p_l < 0.05:
            print('  ==> LPIPS CONFIRMS the PSNR-based improvement on held-out data.')
        elif diff_l.mean() > 0:
            print('  ==> LPIPS points the same direction as PSNR but is not statistically '
                  'significant at this sample size.')
        else:
            print('  ==> LPIPS CONTRADICTS the PSNR-based improvement on held-out data. '
                  'Report this discrepancy explicitly -- do not suppress it because PSNR '
                  'looked good.')

    if args.euvp_a_dir and args.euvp_gt_dir:
        print(f'\n=== Also confirming on fully independent EUVP set ===')
        euvp_pairs = match_pairs(args.euvp_a_dir, args.euvp_gt_dir)
        df_euvp = evaluate(euvp_pairs, 'euvp', lpips_model, lpips_device)
        df_euvp.to_csv(f'{args.tag}_euvp_results.csv', index=False)
        cur_e = df_euvp[df_euvp.variant == 'current (unchanged)'].set_index('image')
        best_e = df_euvp[df_euvp.variant == best_variant].set_index('image')
        common_e = cur_e.index.intersection(best_e.index)
        diff_e = best_e.loc[common_e, 'PSNR'] - cur_e.loc[common_e, 'PSNR']
        t_e, p_e = stats.ttest_rel(best_e.loc[common_e, 'PSNR'], cur_e.loc[common_e, 'PSNR'])
        print(f'On EUVP (PSNR): mean_diff={diff_e.mean():+.3f}dB  p={p_e:.5f}  '
              f'wins={(diff_e>0).sum()}/{len(common_e)}')
        if has_lpips:
            diff_el = cur_e.loc[common_e, 'LPIPS'] - best_e.loc[common_e, 'LPIPS']
            t_el, p_el = stats.ttest_rel(cur_e.loc[common_e, 'LPIPS'], best_e.loc[common_e, 'LPIPS'])
            print(f'On EUVP (LPIPS, positive = improvement): mean_diff={diff_el.mean():+.4f}  '
                  f'p={p_el:.5f}  wins={(diff_el>0).sum()}/{len(common_e)}')


if __name__ == '__main__':
    main()
