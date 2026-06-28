"""Diagnostic: use confident target(T2->t2w) matches as pseudo-truth subject,
then measure QUERY(ceT1->t1c) recall@K against that. Tells us if query's true
subject is even retrievable by intensity cosine."""
import os, glob, csv, numpy as np, nibabel as nib
from scipy.ndimage import zoom
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
    f=f.copy(); fg=f[_M]; f[~_M]=0; f[_M]=(fg-fg.mean())/(fg.std()+1e-6); return f.reshape(-1)
def lib(d):
    try: return os.path.basename(d), zn(ds(reo(load(glob.glob(d+"/*t1c.nii")[0])))), zn(ds(reo(load(glob.glob(d+"/*t2w.nii")[0]))))
    except Exception: return os.path.basename(d),None,None
def d1(a):
    role,path=a
    try: return role, zn(ds(load(path)))
    except Exception: return role,None
def nm(M): return M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-9)
if __name__=="__main__":
    dirs=sorted(glob.glob(BR+"/BraTS-GLI-*"))
    a1=[];a2=[]
    for d in dirs[:200]:
        try: a1.append(ds(reo(load(glob.glob(d+"/*t1c.nii")[0])))); a2.append(ds(reo(load(glob.glob(d+"/*t2w.nii")[0]))))
        except Exception: pass
    T1=np.mean(a1,0);T2=np.mean(a2,0); M=(T1>T1.max()*0.06)|(T2>T2.max()*0.06)
    with ProcessPoolExecutor(max_workers=18,initializer=_i,initargs=(M,)) as ex:
        res=list(ex.map(lib,dirs,chunksize=4))
    ok=[r for r in res if r[1] is not None]; SID=[r[0] for r in ok]
    L1=nm(np.stack([r[1] for r in ok])); L2=nm(np.stack([r[2] for r in ok]))
    idx={s:i for i,s in enumerate(SID)}
    pairs=list(csv.DictReader(open(COMP+"/dataset1/train_pairs.csv")))
    np.random.seed(1); sample=[pairs[i] for i in np.random.permutation(len(pairs))[:80]]
    qa=[("q%d"%i,COMP+"/"+p["query_image"]) for i,p in enumerate(sample)]
    ta=[("t%d"%i,COMP+"/"+p["target_image"]) for i,p in enumerate(sample)]
    with ProcessPoolExecutor(max_workers=18,initializer=_i,initargs=(M,)) as ex:
        qf=dict(ex.map(d1,qa,chunksize=1)); tf=dict(ex.map(d1,ta,chunksize=1))
    # pseudo-truth from confident targets
    recallK={1:0,5:0,15:0,50:0,200:0}; n=0; tcos=[]
    qrank_when_t_conf=[]
    for i in range(len(sample)):
        fq=qf.get("q%d"%i); ft=tf.get("t%d"%i)
        if fq is None or ft is None: continue
        fq=fq/(np.linalg.norm(fq)+1e-9); ft=ft/(np.linalg.norm(ft)+1e-9)
        st=L2@ft; it=int(st.argmax()); tc=float(st[it]); g2=float(np.sort(st)[-2])
        if tc<0.45 or tc-g2<0.05: continue   # only confident target matches as truth
        truth=it; n+=1; tcos.append(tc)
        sq=L1@fq; order=np.argsort(-sq)
        rank=int(np.where(order==truth)[0][0])+1
        qrank_when_t_conf.append(rank)
        for K in recallK:
            if rank<=K: recallK[K]+=1
    print(f"confident-target pseudo-truth pairs: {n} (mean t_cos={np.mean(tcos):.3f})")
    print("QUERY recall vs pseudo-truth subject:")
    for K in sorted(recallK): print(f"  recall@{K:<3d} = {recallK[K]}/{n} = {recallK[K]/max(n,1):.1%}")
    print(f"median query rank of true subject = {int(np.median(qrank_when_t_conf))}")
