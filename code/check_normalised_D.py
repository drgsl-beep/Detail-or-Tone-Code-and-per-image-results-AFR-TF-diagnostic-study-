"""Section 5.1 check, using the authors' own pipeline.py (same folder).
Normalising D by max|D| cancels the DC-gain magnitude, so one fixed order gives nearly the same
tone curve at any alpha. Run:  python check_normalised_D.py
"""
import numpy as np, cv2
from skimage import data, img_as_float
import pipeline as P

NAMES = ["astronaut","chelsea","coffee","rocket","immunohistochemistry","retina","brick",
         "grass","text","page","moon","hubble_deep_field","horse","camera"]

def gray_of(name):
    img = img_as_float(getattr(data, name)())
    if img.ndim == 2: img = np.stack([img]*3, -1)
    img = img[..., :3].astype(np.float64)
    return cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)/255.0

def fixed_D(g, a):
    D = cv2.filter2D(g, -1, P.build_fractional_kernels(a), borderType=cv2.BORDER_REFLECT)
    return D / (np.abs(D).max() + P.EPS)

r_fix, r_ad, smin, smax = [], [], [], []
for n in NAMES:
    g = gray_of(n)
    r = np.corrcoef(fixed_D(g, .7).ravel(), fixed_D(g, 1.6).ravel())[0, 1]
    amap, _ = P.adaptive_alpha_map(g, 0.7, 1.6)
    Da = P.fractional_detail_layer(g, amap); Da = Da/(np.abs(Da).max()+P.EPS)
    ra = np.corrcoef(fixed_D(g, 1.6).ravel(), Da.ravel())[0, 1]
    s = 1 - 5*amap + amap**2
    r_fix.append(r); r_ad.append(ra); smin.append(s.min()); smax.append(s.max())
    print(f"{n:22s} r(0.7 vs 1.6)={r:.4f}  r(fixed 1.6 vs adaptive)={ra:.4f}  s in [{s.min():.2f},{s.max():.2f}]")
print(f"\nMEAN r(normalised D, alpha 0.7 vs 1.6) = {np.mean(r_fix):.3f} (min {np.min(r_fix):.3f}, max {np.max(r_fix):.3f})")
print(f"MEAN r(normalised D, fixed 1.6 vs adaptive) = {np.mean(r_ad):.3f} (min {np.min(r_ad):.3f}, max {np.max(r_ad):.3f})")
