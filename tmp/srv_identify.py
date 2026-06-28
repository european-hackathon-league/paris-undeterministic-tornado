"""Production BraTS-identity matcher.
Each query(ceT1)->signature over library t1c; each gallery(T2)->signature over
library t2w. Score(q,g)=signature dot (peaks at shared subject). Hungarian.
Validate on d1 train (has labels -> real MRR)."""
import os, glob, csv, numpy as np, nibabel as nib
from scipy.ndimage import zoom
from scipy.optimize import linear_sum_assignment
from concurrent.futures import ProcessPoolExecutor
BR="/root/work/brats"; COMP="/root/work/comp"; G=64

def load(p):
    if not os.path.exists(p) and p.endswith(".gz"): p=p[:-3]
    return np.asanyarray(nib.load(p).dataobj).astype(np.float32)
def reo(v): return v[::-1,::-1,:]
def ds(v): return zoom(v,G/np.array(v.shape),order=1).astype(np.float32)
_M=None
def _i(m):
    global _M;_M=m
def zn(f):
    f=f.copy(); fg=f[_M]
    if fg.size<10: return None
    f[~_M]=0; f[_M]=(fg-fg.mean())/(fg.std()+1e-6); return f.reshape(-1)
def lib(d):
    try: return os.path.basename(d), zn(ds(reo(load(glob.glob(d+"/*t1c.nii")[0])))), zn(ds(reo(load(glob.glob(d+"/*t2w.nii")[0]))))
    except Exception: return os.path.basename(d),None,None
def imgfeat(a):
    role,path=a
    try:
        f=zn(ds(load(path)))
        return role, (None if f is None else f)
    except Exception: return role,None
def nm(M): return M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-9)

def build_lib():
    dirs=sorted(glob.glob(BR+"/BraTS-GLI-*"))
    a1=[];a2=[]
    for d in dirs[:200]:
        try: a1.append(ds(reo(load(glob.glob(d+"/*t1c.nii")[0])))); a2.append(ds(reo(load(glob.glob(d+"/*t2w.nii")[0]))))
        except Exception: pass
    T1=np.mean(a1,0);T2=np.mean(a2,0); M=(T1>T1.max()*0.06)|(T2>T2.max()*0.06)
    with ProcessPoolExecutor(max_workers=18,initializer=_i,initargs=(M,)) as ex:
        res=list(ex.map(lib,dirs,chunksize=4))
    ok=[r for r in res if r[1] is not None]
    SID=[r[0] for r in ok]; L1=nm(np.stack([r[1] for r in ok])); L2=nm(np.stack([r[2] for r in ok]))
    return M,SID,L1,L2

def sigs(paths, L, M):
    with ProcessPoolExecutor(max_workers=18,initializer=_i,initargs=(M,)) as ex:
        feats=dict(ex.map(imgfeat,[("%d"%i,p) for i,p in enumerate(paths)],chunksize=1))
    F=[]
    for i in range(len(paths)):
        f=feats.get("%d"%i)
        F.append(np.zeros(L.shape[1]) if f is None else f)
    F=nm(np.stack(F))
    return F@L.T   # (N, n_lib) similarity signatures

def mrr_hungarian(S):
    n=S.shape[0]
    row,col=linear_sum_assignment(-S); assigned=np.empty(n,int); assigned[row]=col
    Sb=S.copy(); Sb[np.arange(n),assigned]=S.max()+1e6
    rr=[]
    for i in range(n):
        order=np.argsort(-Sb[i],kind="mergesort"); rr.append(1.0/(int(np.where(order==i)[0][0])+1))
    return float(np.mean(rr)), float((assigned==np.arange(n)).mean())

if __name__=="__main__":
    M,SID,L1,L2=build_lib()
    print(f"library: {L1.shape}",flush=True)
    pairs=list(csv.DictReader(open(COMP+"/dataset1/train_pairs.csv")))
    qpaths=[COMP+"/"+p["query_image"] for p in pairs]
    tpaths=[COMP+"/"+p["target_image"] for p in pairs]
    QS=sigs(qpaths,L1,M); print("query sigs done",flush=True)
    TS=sigs(tpaths,L2,M); print("target sigs done",flush=True)
    print(f"\n=== d1 TRAIN (350-pool) BraTS-identity method ===")
    # (a) raw signature dot
    print(f"  [raw dot]      MRR={mrr_hungarian(QS@TS.T)[0]:.4f}")
    # (b) z-scored signatures (remove common-mode popular-subject component)
    def zr(X): return (X-X.mean(1,keepdims=True))/(X.std(1,keepdims=True)+1e-9)
    QZ=zr(QS); TZ=zr(TS)
    print(f"  [zscore dot]   MRR={mrr_hungarian(QZ@TZ.T)[0]:.4f}")
    # (c) double-centered (remove popular-subject bias on both axes)
    def dc(X):
        X=X-X.mean(0,keepdims=True); return X
    print(f"  [colcenter dot] MRR={mrr_hungarian(zr(QS-QS.mean(0))@zr(TS-TS.mean(0)).T)[0]:.4f}")
    # (d) binary argmax-subject equality (hard identity match)
    qarg=QS.argmax(1); targ=TS.argmax(1)
    Sb=(qarg[:,None]==targ[None,:]).astype(float)
    print(f"  [argmax-eq]    MRR={mrr_hungarian(Sb)[0]:.4f}   q-subj==t-subj true pairs: {(qarg==targ).mean():.2%}")
    np.save("/root/work/d1_QS.npy",QS); np.save("/root/work/d1_TS.npy",TS)
