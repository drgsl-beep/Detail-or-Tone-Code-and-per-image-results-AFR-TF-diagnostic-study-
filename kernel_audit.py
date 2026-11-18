"""
kernel_audit.py -- DC gain and brightness coupling of fractional-order masks (Section 5.6, Table 10).

Masks: the implemented compact 3x3 mask of pipeline.py; truncated Grunwald-Letnikov (GL) stencils
with K taps summed over 4 or 8 directions; and their zero-sum versions (centre tap replaced by minus
the sum of the others). The DC gain of a truncated GL stencil is d * (-1)^(K-1) * C(alpha-1, K-1),
with d directions, so it vanishes only at integer orders.
Run in the folder that holds pipeline.py:   python kernel_audit.py
"""
import numpy as np, cv2
from skimage import data, img_as_float
import pipeline as P
NAMES=["astronaut","chelsea","coffee","rocket","immunohistochemistry","retina","brick","grass","text","page","moon","hubble_deep_field","horse","camera"]
def gray_of(n):
    img=img_as_float(getattr(data,n)()); 
    if img.ndim==2: img=np.stack([img]*3,-1)
    img=img[...,:3]; return cv2.cvtColor((img*255).astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float64)/255.0
def gl(alpha,K):
    w=[1.0]
    for k in range(1,K): w.append(w[-1]*(k-1-alpha)/k)
    return np.array(w)
def mask_gl(alpha,K=3,dirs=4,zero=False):
    r=K-1; m=np.zeros((2*r+1,2*r+1)); w=gl(alpha,K)
    D=[(1,0),(-1,0),(0,1),(0,-1)]+([(1,1),(1,-1),(-1,1),(-1,-1)] if dirs==8 else [])
    for dx,dy in D:
        for k in range(K): m[r+k*dy,r+k*dx]+=w[k]
    if zero: m[r,r]-=m.sum()
    return m
def mask_impl(alpha,zero=False):
    k=P.build_fractional_kernels(alpha).copy()
    if zero: k[1,1]-=k.sum()
    return k
G=[gray_of(n) for n in NAMES]
def stats(kf):
    r=[]; 
    for g in G:
        D=cv2.filter2D(g,-1,kf,borderType=cv2.BORDER_REFLECT); r.append(np.corrcoef(D.ravel(),g.ravel())[0,1])
    return np.mean(r),np.std(r,ddof=1)

if __name__ == "__main__":
    print(f"{'mask':44s}{'alpha':>6s}{'DC gain':>9s}{'mean r':>9s}{'sd':>7s}")
    rows=[('implemented 3x3 (centre tap 1)',lambda a:mask_impl(a)),
    ('implemented, centre tap zero-summed',lambda a:mask_impl(a,True)),
    ('GL 4-direction, K=3 (5x5)',lambda a:mask_gl(a,3,4)),
    ('GL 8-direction, K=3 (5x5)',lambda a:mask_gl(a,3,8)),
    ('GL 4-direction, K=5 (9x9)',lambda a:mask_gl(a,5,4)),
    ('GL 8-direction, K=5 (9x9)',lambda a:mask_gl(a,5,8))]
    for name,f in rows:
        for a in (0.7,1.0,1.6):
            k=f(a); m,s=stats(k); print(f"{name:44s}{a:6.1f}{k.sum():9.2f}{m:9.3f}{s:7.3f}")
    