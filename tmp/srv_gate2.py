"""GATE v2: register d1 -> BraTS template, apply template brain mask (auto skull
removal), then identify. Goal: boost query(ceT1) side above v1's weak 62%."""
import os, glob, csv, numpy as np, nibabel as nib
from scipy.ndimage import zoom, affine_transform
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor

G = 64
OPT = 24
BR = "/root/work/brats"
COMP = "/root/work/comp"

def load(p):
    if not os.path.exists(p) and p.endswith(".gz"): p = p[:-3]
    return np.asanyarray(nib.load(p).dataobj).astype(np.float32)

def reorient_brats(v): return v[::-1, ::-1, :]

def ds(v, g): return zoom(v, g / np.array(v.shape), order=1).astype(np.float32)

def ncc(a, b):
    fg = (a > 0.02) | (b > 0.02)
    if fg.sum() < 16: return 0.0
    x, y = a[fg], b[fg]; x = x - x.mean(); y = y - y.mean()
    return float((x @ y) / ((np.linalg.norm(x) * np.linalg.norm(y)) + 1e-6))

def rigmat(p):
    rot = np.eye(3)
    for ax, ang in enumerate(p[:3]):
        c, s = np.cos(ang), np.sin(ang); r = np.eye(3)
        a, b = [i for i in range(3) if i != ax]; r[a,a],r[a,b],r[b,a],r[b,b]=c,-s,s,c
        rot = rot @ r
    return rot, p[3:6]

def apply_rig(v, p):
    rot, sh = rigmat(p); c = (np.asarray(v.shape)-1)/2.0
    return affine_transform(v, rot, offset=c-rot@c+sh, order=1, mode="constant", cval=0.0)

def register(vol, tpl):
    g = vol.shape[0]
    zt = ds(tpl, OPT); zv = ds(vol, OPT)
    def neg(p): return -ncc(zt, apply_rig(zv, p))
    bp, bf = np.zeros(6), neg(np.zeros(6))
    for st in (np.zeros(6),[0.3,0,0,0,0,0],[0,0.3,0,0,0,0],[0,0,0.3,0,0,0],[0.3,0.3,0,0,0,0]):
        r = minimize(neg, np.array(st,float), method="Powell", options={"maxiter":80,"xtol":0.03,"ftol":0.005})
        if r.fun < bf: bf, bp = r.fun, r.x
    sp = bp.copy(); sp[3:6] *= g/OPT
    return apply_rig(vol, sp)

# globals filled per-process
_TPL1=_TPL2=_MASK=None
def _init(tpl1, tpl2, mask):
    global _TPL1,_TPL2,_MASK; _TPL1,_TPL2,_MASK=tpl1,tpl2,mask

def feat_lib(d):
    sid = os.path.basename(d)
    try:
        f1 = ds(reorient_brats(load(glob.glob(d+"/*t1c.nii")[0])), G)
        f2 = ds(reorient_brats(load(glob.glob(d+"/*t2w.nii")[0])), G)
        def mk(f):
            f=f.copy(); m=_MASK; fg=f[m]
            f[~m]=0; f[m]=(fg-fg.mean())/(fg.std()+1e-6); return f.reshape(-1).astype(np.float32)
        return sid, mk(f1), mk(f2)
    except Exception: return sid, None, None

def feat_d1(args):
    role, path, mod = args
    try:
        v = ds(load(path), G)
        tpl = _TPL1 if mod=="t1c" else _TPL2
        vr = register(v, tpl)
        m=_MASK; fg=vr[m]; vr[~m]=0; vr[m]=(fg-fg.mean())/(fg.std()+1e-6)
        return role, vr.reshape(-1).astype(np.float32)
    except Exception as e: return role, None

def norm(M): return M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-9)

if __name__ == "__main__":
    dirs = sorted(glob.glob(BR+"/BraTS-GLI-*"))
    # template from first 200 subjects (BraTS already atlas-aligned)
    sub = dirs[:200]
    acc1=[]; acc2=[]
    for d in sub:
        try:
            acc1.append(ds(reorient_brats(load(glob.glob(d+"/*t1c.nii")[0])),G))
            acc2.append(ds(reorient_brats(load(glob.glob(d+"/*t2w.nii")[0])),G))
        except Exception: pass
    TPL1=np.mean(acc1,0); TPL2=np.mean(acc2,0)
    MASK = (TPL1 > TPL1.max()*0.06) | (TPL2 > TPL2.max()*0.06)
    print(f"templates built, brain-mask voxels={int(MASK.sum())}", flush=True)

    with ProcessPoolExecutor(max_workers=18, initializer=_init, initargs=(TPL1,TPL2,MASK)) as ex:
        res = list(ex.map(feat_lib, dirs, chunksize=4))
    sids=[r[0] for r in res if r[1] is not None]
    L1=norm(np.stack([r[1] for r in res if r[1] is not None]))
    L2=norm(np.stack([r[2] for r in res if r[1] is not None]))
    print(f"library: {L1.shape}", flush=True)

    pairs=list(csv.DictReader(open(COMP+"/dataset1/train_pairs.csv")))
    np.random.seed(0); sample=[pairs[i] for i in np.random.permutation(len(pairs))[:50]]
    qa=[("q%d"%i, COMP+"/"+p["query_image"], "t1c") for i,p in enumerate(sample)]
    ta=[("t%d"%i, COMP+"/"+p["target_image"], "t2w") for i,p in enumerate(sample)]
    with ProcessPoolExecutor(max_workers=18, initializer=_init, initargs=(TPL1,TPL2,MASK)) as ex:
        qf=dict(ex.map(feat_d1, qa, chunksize=1)); tf=dict(ex.map(feat_d1, ta, chunksize=1))

    cons=0; n=0; qr1=0; tr1=0
    for i in range(len(sample)):
        fq=qf.get("q%d"%i); ft=tf.get("t%d"%i)
        if fq is None or ft is None: continue
        fq=fq/(np.linalg.norm(fq)+1e-9); ft=ft/(np.linalg.norm(ft)+1e-9)
        sq=L1@fq; st=L2@ft; iq=int(sq.argmax()); it=int(st.argmax()); n+=1
        cons += sids[iq]==sids[it]
    print(f"\nv2 SELF-CONSISTENCY q-subj==t-subj: {cons}/{n} = {cons/max(n,1):.2%}")
