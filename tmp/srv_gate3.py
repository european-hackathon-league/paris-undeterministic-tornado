"""GATE v3: global-cosine prefilter (top-K) -> rigid registration NCC rerank,
intra-modal (d1 query ceT1 vs BraTS t1c; d1 target T2 vs BraTS t2w)."""
import os, glob, csv, numpy as np, nibabel as nib
from scipy.ndimage import zoom, affine_transform
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor

BR="/root/work/brats"; COMP="/root/work/comp"
G=64; OPT=32; TOPK=15

def load(p):
    if not os.path.exists(p) and p.endswith(".gz"): p=p[:-3]
    return np.asanyarray(nib.load(p).dataobj).astype(np.float32)
def reo(v): return v[::-1,::-1,:]
def ds(v,g): return zoom(v,g/np.array(v.shape),order=1).astype(np.float32)
def znorm(f,m):
    f=f.copy(); fg=f[m]; f[~m]=0; f[m]=(fg-fg.mean())/(fg.std()+1e-6); return f

def ncc(a,b):
    fg=(a>0.05)&(b!=0)
    if fg.sum()<16: return 0.0
    x,y=a[fg],b[fg]; x=x-x.mean(); y=y-y.mean()
    return float((x@y)/((np.linalg.norm(x)*np.linalg.norm(y))+1e-6))
def rigmat(p):
    rot=np.eye(3)
    for ax,ang in enumerate(p[:3]):
        c,s=np.cos(ang),np.sin(ang); r=np.eye(3)
        a,b=[i for i in range(3) if i!=ax]; r[a,a],r[a,b],r[b,a],r[b,b]=c,-s,s,c; rot=rot@r
    return rot,p[3:6]
def apply_rig(v,p):
    rot,sh=rigmat(p); c=(np.asarray(v.shape)-1)/2.0
    return affine_transform(v,rot,offset=c-rot@c+sh,order=1,mode="constant",cval=0.0)
def reg_ncc(mov, fix):
    def neg(p): return -ncc(fix, apply_rig(mov,p))
    bf=neg(np.zeros(6))
    for st in (np.zeros(6),[0,0,0.3,0,0,0],[0,0,-0.3,0,0,0],[0.25,0,0,0,0,0]):
        r=minimize(neg,np.array(st,float),method="Powell",options={"maxiter":50,"xtol":0.05,"ftol":0.02})
        bf=min(bf,r.fun)
    return -bf

# globals
_L1=_L2=_SID=_MASK=_LIBV1=_LIBV2=None
def _init(L1,L2,SID,MASK,LV1,LV2):
    global _L1,_L2,_SID,_MASK,_LIBV1,_LIBV2; _L1,_L2,_SID,_MASK,_LIBV1,_LIBV2=L1,L2,SID,MASK,LV1,LV2

def identify(args):
    role,path,mod=args
    try:
        vfull=ds(load(path),G)
        f=znorm(vfull,_MASK).reshape(-1); f=f/(np.linalg.norm(f)+1e-9)
        L=_L1 if mod=="t1c" else _L2; LV=_LIBV1 if mod=="t1c" else _LIBV2
        s=L@f
        cand=np.argsort(-s)[:TOPK]
        movs=ds(load(path),OPT)
        best=None
        for c in cand:
            fix=LV[c]  # candidate volume at OPT res
            sc=reg_ncc(movs,fix)
            if best is None or sc>best[1]: best=(c,sc)
        return role,_SID[int(best[0])], float(best[1]), _SID[int(cand[0])]
    except Exception as e:
        return role,None,0.0,None
def norm(M): return M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-9)

if __name__=="__main__":
    dirs=sorted(glob.glob(BR+"/BraTS-GLI-*"))
    acc1=[];acc2=[]
    for d in dirs[:200]:
        try: acc1.append(ds(reo(load(glob.glob(d+"/*t1c.nii")[0])),G)); acc2.append(ds(reo(load(glob.glob(d+"/*t2w.nii")[0])),G))
        except Exception: pass
    T1=np.mean(acc1,0); T2=np.mean(acc2,0); MASK=(T1>T1.max()*0.06)|(T2>T2.max()*0.06)
    # build library: global feats (G) + OPT-res volumes for registration
    def lib(d):
        try:
            v1=reo(load(glob.glob(d+"/*t1c.nii")[0])); v2=reo(load(glob.glob(d+"/*t2w.nii")[0]))
            return os.path.basename(d), znorm(ds(v1,G),MASK).reshape(-1), znorm(ds(v2,G),MASK).reshape(-1), ds(v1,OPT), ds(v2,OPT)
        except Exception: return os.path.basename(d),None,None,None,None
    with ProcessPoolExecutor(max_workers=18) as ex:
        res=list(ex.map(lib,dirs,chunksize=4))
    ok=[r for r in res if r[1] is not None]
    SID=[r[0] for r in ok]; L1=norm(np.stack([r[1] for r in ok])); L2=norm(np.stack([r[2] for r in ok]))
    LV1=np.stack([r[3] for r in ok]); LV2=np.stack([r[4] for r in ok])
    print(f"library {L1.shape}, OPT vols {LV1.shape}",flush=True)
    pairs=list(csv.DictReader(open(COMP+"/dataset1/train_pairs.csv")))
    np.random.seed(0); sample=[pairs[i] for i in np.random.permutation(len(pairs))[:50]]
    qa=[("q%d"%i,COMP+"/"+p["query_image"],"t1c") for i,p in enumerate(sample)]
    ta=[("t%d"%i,COMP+"/"+p["target_image"],"t2w") for i,p in enumerate(sample)]
    with ProcessPoolExecutor(max_workers=18,initializer=_init,initargs=(L1,L2,SID,MASK,LV1,LV2)) as ex:
        qf={r[0]:r for r in ex.map(identify,qa,chunksize=1)}
        tf={r[0]:r for r in ex.map(identify,ta,chunksize=1)}
    cons=0; cons_pre=0; n=0
    for i in range(len(sample)):
        rq=qf.get("q%d"%i); rt=tf.get("t%d"%i)
        if not rq or not rt or rq[1] is None or rt[1] is None: continue
        n+=1
        cons += rq[1]==rt[1]            # after registration rerank
        cons_pre += rq[3]==rt[3]        # global-cosine argmax only
    print(f"\nv3 SELF-CONSISTENCY  rerank: {cons}/{n} = {cons/max(n,1):.2%}   (pre-rerank cosine: {cons_pre/max(n,1):.2%})")
