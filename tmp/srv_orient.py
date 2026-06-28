import subprocess, glob
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True, errors="replace").stdout
print("=== comp extract status ==="); print(sh("cat /root/work/comp_done.flag 2>/dev/null || echo extracting; find /root/work/comp -name '*.nii' | wc -l; find /root/work/comp -name 'train_pairs.csv'"))

import nibabel as nib, numpy as np, csv
# one BraTS t1c affine
b = glob.glob("/root/work/brats/BraTS-GLI-00000-000/*t1c.nii")[0]
bi = nib.load(b)
print("BraTS affine:\n", np.round(bi.affine,3), "shape", bi.shape)
# one d1 query affine
tp = "/root/work/comp/dataset1/train_pairs.csv"
import os
if os.path.exists(tp):
    r = next(csv.DictReader(open(tp)))
    qp = "/root/work/comp/"+r["query_image"]
    if not os.path.exists(qp) and qp.endswith(".gz"): qp = qp[:-3]
    di = nib.load(qp)
    print("d1 query affine:\n", np.round(di.affine,3), "shape", di.shape)
    # relative orientation: which axis/flip maps BraTS voxel space to d1 voxel space
    import numpy.linalg as la
    M = la.inv(di.affine) @ bi.affine  # BraTS-vox -> world -> d1-vox
    print("BraTS-vox -> d1-vox linear (rounded):\n", np.round(M[:3,:3],2), "\n offset", np.round(M[:3,3],1))
else:
    print("comp not extracted yet")
