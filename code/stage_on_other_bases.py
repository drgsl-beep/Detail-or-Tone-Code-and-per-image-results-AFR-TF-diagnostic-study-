"""
stage_on_other_bases.py -- Sections 5.6-5.7: (1) adaptive vs fixed order for the implemented and for
standard GL masks, with and without the zero-sum correction; (2) the implemented stage attached to other
base images (unprocessed input, CLAHE output, AFR-TF fused base). Synthetic battery, distinct seeds.
Run in the folder that holds pipeline.py, run_table2b_no_detail.py and kernel_audit.py:
    python stage_on_other_bases.py
Writes novelty_expB_kernels.csv and novelty_expB_bases.csv and prints the summary.
"""
import numpy as np, cv2, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy import stats
import pipeline as P, run_table2b_no_detail as t
from kernel_audit import mask_gl, mask_impl, NAMES
EPS=1e-6
t.SEED,t.SEED_MODE=0,'distinct'
bat=t.load_battery(["skimage:"+n for n in NAMES]); t._IDX.clear(); t._IDX.update({id(r):i for i,r in enumerate(bat.values())})
g8=lambda x: cv2.cvtColor((x*255).astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float64)/255.0
def det_adaptive(gray,mk):
    amap,_=P.adaptive_alpha_map(gray,0.7,1.6); lo,hi=float(amap.min()),float(amap.max())
    cs=np.linspace(lo,hi,6); da=cs[1]-cs[0]; num=np.zeros_like(gray); den=np.zeros_like(gray)
    for c in cs:
        w=np.clip(1-np.abs(amap-c)/da,0,1); num+=w*cv2.filter2D(gray,-1,mk(c),borderType=cv2.BORDER_REFLECT); den+=w
    return num/(den+1e-6)
def det_fixed(gray,mk,a): return cv2.filter2D(gray,-1,mk(a),borderType=cv2.BORDER_REFLECT)
def apply(base,D): D=D/(np.abs(D).max()+EPS); return np.clip(base*(1+0.45*D[...,None]),0,1)
KERN={'implemented':lambda a:mask_impl(a),'implemented zero-sum':lambda a:mask_impl(a,True),
      'GL 4-dir K=3':lambda a:mask_gl(a,3,4),'GL 4-dir K=3 zero-sum':lambda a:mask_gl(a,3,4,True),
      'GL 8-dir K=3':lambda a:mask_gl(a,3,8),'GL 8-dir K=3 zero-sum':lambda a:mask_gl(a,3,8,True)}
rows=[]; rowsB=[]
for deg in t.DEGRADATIONS:
    for n,ref in bat.items():
        d=P.make_degraded(ref,deg,seed=t.case_seed(ref,deg))
        base=t._branch_fusion(d); gb=g8(base)
        rows.append(dict(deg=deg,img=n,kernel='no stage',kind='none',PSNR=P.compute_metrics(ref,base)['PSNR']))
        for kn,mk in KERN.items():
            rows.append(dict(deg=deg,img=n,kernel=kn,kind='adaptive',PSNR=P.compute_metrics(ref,apply(base,det_adaptive(gb,mk)))['PSNR']))
            rows.append(dict(deg=deg,img=n,kernel=kn,kind='fixed1',PSNR=P.compute_metrics(ref,apply(base,det_fixed(gb,mk,1.0)))['PSNR']))
        # Experiment B: other bases, implemented stage
        for bn,b in (('unprocessed input',d),('CLAHE output',P.baseline_clahe(d)),('AFR-TF fused base',base)):
            gbb=g8(b); out=apply(b,det_adaptive(gbb,KERN['implemented']))
            outz=apply(b,det_adaptive(gbb,KERN['implemented zero-sum']))
            p0=P.compute_metrics(ref,b)['PSNR']
            rowsB.append(dict(deg=deg,img=n,base=bn,offset=b.mean()-ref.mean(),eff_impl=P.compute_metrics(ref,out)['PSNR']-p0,eff_zero=P.compute_metrics(ref,outz)['PSNR']-p0,
                              r=np.corrcoef(det_adaptive(gbb,KERN['implemented']).ravel(),gbb.ravel())[0,1]))
    print('done',deg,flush=True)
df=pd.DataFrame(rows); dfB=pd.DataFrame(rowsB)
df.to_csv('novelty_expB_kernels.csv',index=False); dfB.to_csv('novelty_expB_bases.csv',index=False)
def holm(p):
    p=np.asarray(p,float); o=np.argsort(p); a=np.zeros(len(p)); m=0
    for k,i in enumerate(o): m=max(m,min(1,p[i]*(len(p)-k))); a[i]=m
    return a
print('\n=== EXP B1: adaptive minus fixed(alpha=1) and adaptive minus no-stage, per kernel (dB; Holm over 3 degradations) ===')
piv=df.pivot_table(index=['deg','img'],columns=['kernel','kind'],values='PSNR')
for kn in KERN:
    for lab,b in (('adaptive-fixed1',(kn,'fixed1')),('adaptive-nostage',('no stage','none'))):
        res=[]
        for deg in t.DEGRADATIONS:
            x=(piv.loc[deg][(kn,'adaptive')]-piv.loc[deg][b]).values
            res.append((deg,x.mean(),stats.ttest_1samp(x,0).pvalue,int((x>0).sum())))
        h=holm([r[2] for r in res])
        print(f"{kn:24s}{lab:18s}"+"  ".join(f"{r[0][:5]} {r[1]:+.3f} (p_holm {hh:.3f}, {r[3]}/14)" for r,hh in zip(res,h)))
print('\n=== EXP B2: implemented stage on other bases ===')
for bn,g in dfB.groupby('base'):
    rho=stats.spearmanr(g.offset,g.eff_impl); rz=stats.spearmanr(g.offset,g.eff_zero)
    print(f"{bn:20s} mean r(D,brightness)={g.r.mean():+.3f} | stage effect: implemented {g.eff_impl.mean():+.2f} dB, zero-sum {g.eff_zero.mean():+.2f} dB | Spearman(offset,effect) implemented rho={rho.statistic:+.2f} p={rho.pvalue:.4f}; zero-sum rho={rz.statistic:+.2f} p={rz.pvalue:.3f}")
