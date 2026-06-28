"""GATE v1.5: v1 (no registration, d1 native already aligned to BraTS-flipped) +
template brain mask to strip skull from d1. Resolution sweep."""
import os, glob, csv, numpy as np, nibabel as nib
from scipy.ndimage import zoom
from concurrent.futures import ProcessPoolExecutor

BR="/root/work/brats"; COMP="/root/work/comp"
G=72

def load(p):
    if not os.path.exists(p) and p.endswith(".gz"): p=p[:-3]
    return np.asanyarray(nib.load(p).dataobj).astype(np.float32)
def reo(v): return v[::-1,::-1,:]
def ds(v): return zoom(v, G/np.array(v.shape), order=1).astype(np.float32)

_MASK=None
def _init(m):
    global _MASK; _MASK=m
def mk(f):
    f=f.copy(); fg=f[_MASK]; f[~_MASK]=0
    f[_MASK]=(fg-fg.mean())/(fg.std()+1e-6); return f.reshape(-1).astype(np.float32)
def lib(d):
    sid=os.path.basename(d)
    try: return sid, mk(ds(reo(load(glob.glob(d+"/*t1c.nii")[0])))), mk(ds(reo(load(glob.glob(d+"/*t2w.nii")[0]))))
    except Exception: return sid,None,None
def d1f(a):
    role,path=a
    try: return role, mk(ds(load(path)))   # d1 native (already in BraTS-flipped pose), masked
    except Exception: return role,None
def norm(M): return M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-9)

if __name__=="__main__":
    dirs=sorted(glob.glob(BR+"/BraTS-GLI-*"))
    acc1=[];acc2=[]
    for d in dirs[:200]:
        try: acc1.append(ds(reo(load(glob.glob(d+"/*t1c.nii")[0])))); acc2.append(ds(reo(load(glob.glob(d+"/*t2w.nii")[0]))))
        except Exception: pass
    T1=np.mean(acc1,0); T2=np.mean(acc2,0)
    MASK=(T1>T1.max()*0.06)|(T2>T2.max()*0.06)
    print(f"G={G} mask={int(MASK.sum())}",flush=True)
    with ProcessPoolExecutor(max_workers=18,initializer=_init,initargs=(MASK,)) as ex:
        res=list(ex.map(lib,dirs,chunksize=4))
    sids=[r[0] for r in res if r[1] is not None]
    L1=norm(np.stack([r[1] for r in res if r[1] is not None]))
    L2=norm(np.stack([r[2] for r in res if r[1] is not None]))
    print(f"library {L1.shape}",flush=True)
    pairs=list(csv.DictReader(open(COMP+"/dataset1/train_pairs.csv")))
    np.random.seed(0); sample=[pairs[i] for i in np.random.permutation(len(pairs))[:50]]
    qa=[("q%d"%i,COMP+"/"+p["query_image"]) for i,p in enumerate(sample)]
    ta=[("t%d"%i,COMP+"/"+p["target_image"]) for i,p in enumerate(sample)]
    with ProcessPoolExecutor(max_workers=18,initializer=_init,initargs=(MASK,)) as ex:
        qf=dict(ex.map(d1f,qa,chunksize=1)); tf=dict(ex.map(d1f,ta,chunksize=1))
    cons=0;n=0; qgap=[]
    for i in range(len(sample)):
        fq=qf.get("q%d"%i); ft=tf.get("t%d"%i)
        if fq is None or ft is None: continue
        fq=fq/(np.linalg.norm(fq)+1e-9); ft=ft/(np.linalg.norm(ft)+1e-9)
        sq=L1@fq; st=L2@ft; iq=int(sq.argmax()); it=int(st.argmax()); n+=1
        cons+= sids[iq]==sids[it]
        qgap.append((float(np.sort(sq)[-1]),float(np.sort(sq)[-2]),float(np.sort(st)[-1])))
    print(f"\nv1.5 SELF-CONSISTENCY: {cons}/{n} = {cons/max(n,1):.2%}")
    qg=np.array(qgap); print(f"mean q_top1={qg[:,0].mean():.3f} q_2nd={qg[:,1].mean():.3f} t_top1={qg[:,2].mean():.3f}")
