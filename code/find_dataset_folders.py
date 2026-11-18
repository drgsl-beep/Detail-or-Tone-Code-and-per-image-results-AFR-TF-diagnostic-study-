"""
find_dataset_folders.py — finds the RESIDE-SOTS (outdoor) and LOL-v2 (Real_captured, Test) image
folders under the current directory and prints the exact commands to run.

Run from C:\\afrtf_eval:
    python find_dataset_folders.py
"""
import os

def dirs_named(root, name):
    out = []
    for dp, dn, _ in os.walk(root):
        for d in dn:
            if d.lower() == name.lower():
                out.append(os.path.join(dp, d))
    return out

def count_imgs(d):
    try:
        return sum(1 for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')))
    except OSError:
        return 0

print("=" * 70)
print("RESIDE-SOTS (outdoor)")
hazy = [d for d in dirs_named('SOTS_extracted', 'hazy') if 'outdoor' in d.lower()]
if not hazy:
    print("  Could not find an 'outdoor\\hazy' folder under SOTS_extracted.")
    print("  Send me the output of:  dir SOTS_extracted /s /b /ad")
for h in hazy:
    parent = os.path.dirname(h)
    gt = next((os.path.join(parent, n) for n in os.listdir(parent) if n.lower() in ('gt', 'clear', 'clean')), None)
    print(f"  hazy folder: {h}  ({count_imgs(h)} images)")
    print(f"  gt folder  : {gt}  ({count_imgs(gt) if gt else 0} images)")
    if gt:
        print("\n  >>> RUN 3 (copy this whole line):")
        print(f'  python run_zero_sum_real.py --raw_dir "{h}" --ref_dir "{gt}" --tag zs_reside')

print("\n" + "=" * 70)
print("LOL-v2 (Real_captured, Test)")
cands = []
for t in dirs_named('LOLv2_extracted', 'Test') + dirs_named('LOLv2_extracted', 'Test_matched'):
    if 'real' in t.lower():
        low = next((os.path.join(t, n) for n in os.listdir(t) if n.lower() == 'low'), None)
        nor = next((os.path.join(t, n) for n in os.listdir(t) if n.lower() in ('normal', 'high')), None)
        if low and nor:
            cands.append((t, low, nor))
if not cands:
    print("  Could not find a Real_captured Test folder with Low and Normal subfolders.")
    print("  Send me the output of:  dir LOLv2_extracted /s /b /ad")
# IMPORTANT: use the ORIGINAL Test folder (names like low00694.png), not Test_matched.
# The tune/held-out split is an MD5 hash of the file name, so renamed files give a different split.
original = [c for c in cands if not c[0].lower().endswith('test_matched')]
for t, low, nor in (original or cands):
    print(f"  folder: {t}\n  Low   : {low} ({count_imgs(low)} images)\n  Normal: {nor} ({count_imgs(nor)} images)")
    print("\n  >>> RUN 2 (copy this whole line; expect 55 tune / 45 held-out):")
    print(f'  python run_zero_sum_real.py --raw_dir "{low}" --ref_dir "{nor}" --tag zs_lol55')
    break
