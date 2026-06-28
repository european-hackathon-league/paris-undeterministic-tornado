from __future__ import annotations

"""d1 leak hunt: does any metadata field tie each query to its true target?

d1 has labels (train_pairs.csv) -> we can VERIFY a candidate key directly:
a real leak must (a) match the known train pairs and (b) induce a unique
bijection on the unlabelled val/test pools. Mirrors the confirmed d3 affine leak.

Checks per candidate field (query-side value vs target-side value):
  - affine matrix (sform), shape, pixdim/voxel spacing, scl_slope/inter,
    header text (descrip/db_name/aux_file/intent_name), datatype/bitpix,
    cal_min/max, glmax/glmin, file size, nonzero voxel count.

For each field we report:
  - train: how many of the 350 true pairs have an EXACT query==target match,
  - bijection: on a pool, does the field map each query to exactly one gallery
    (unique on both sides)? A high-entropy field that is both bijective AND
    matches the train pairs == proof of leak.
"""

import csv
from pathlib import Path
from collections import Counter

import numpy as np
import nibabel as nib

from synthetic_d2_eval import resolve_image_path

DATA = Path("data")


def read_csv(p):
    with (DATA / p).open(newline="") as f:
        return list(csv.DictReader(f))


def field_values(path: Path) -> dict:
    img = nib.load(str(path))
    hdr = img.header
    vol = np.asanyarray(img.dataobj)
    nz = int(np.count_nonzero(vol))
    aff = np.asarray(img.affine, dtype=np.float64)
    pixdim = tuple(np.round(hdr["pixdim"], 5).tolist())
    return {
        "affine": np.round(aff, 4).tobytes(),
        "affine_origin": tuple(np.round(aff[:3, 3], 4).tolist()),
        "shape": tuple(int(s) for s in vol.shape),
        "pixdim": pixdim,
        "scl": (float(hdr["scl_slope"]), float(hdr["scl_inter"])),
        "descrip": bytes(hdr["descrip"]).rstrip(b"\x00"),
        "db_name": bytes(hdr["db_name"]).rstrip(b"\x00"),
        "aux_file": bytes(hdr["aux_file"]).rstrip(b"\x00"),
        "intent_name": bytes(hdr["intent_name"]).rstrip(b"\x00"),
        "datatype": int(hdr["datatype"]),
        "calminmax": (float(hdr["cal_min"]), float(hdr["cal_max"])),
        "glminmax": (int(hdr["glmin"]), int(hdr["glmax"])),
        "filesize": path.stat().st_size,
        "nonzero": nz,
    }


FIELDS = ["affine", "affine_origin", "shape", "pixdim", "scl", "descrip",
          "db_name", "aux_file", "intent_name", "datatype", "calminmax",
          "glminmax", "filesize", "nonzero"]


def hashable(v):
    return v


def analyze_pool(qrows, trows, qkey, tkey, qimg, timg, label):
    """For an UNLABELLED pool: does each field induce a clean 1-1 bijection?"""
    print(f"\n--- pool {label}: {len(qrows)} queries x {len(trows)} gallery ---")
    qvals = {f: [] for f in FIELDS}
    tvals = {f: [] for f in FIELDS}
    for r in qrows:
        fv = field_values(resolve_image_path(DATA, r[qimg]))
        for f in FIELDS:
            qvals[f].append(fv[f])
    for r in trows:
        fv = field_values(resolve_image_path(DATA, r[timg]))
        for f in FIELDS:
            tvals[f].append(fv[f])
    for f in FIELDS:
        qc = Counter(qvals[f]); tc = Counter(tvals[f])
        q_uniq = sum(1 for v in qc.values() if v == 1)
        t_uniq = sum(1 for v in tc.values() if v == 1)
        # how many query values find exactly one matching gallery value
        matched = sum(1 for v in qvals[f] if tc.get(v, 0) == 1 and qc.get(v, 0) == 1)
        n = len(qrows)
        flag = "  <== BIJECTION" if matched == n and n > 0 else ""
        print(f"  {f:14s} q_distinct={len(qc):3d} t_distinct={len(tc):3d} "
              f"q_unique={q_uniq:3d} 1-1_matches={matched:3d}/{n}{flag}")


def main():
    pairs = read_csv("dataset1/train_pairs.csv")
    print(f"=== d1 TRAIN leak check ({len(pairs)} labelled pairs) ===")
    exact = {f: 0 for f in FIELDS}
    distinct_q = {f: set() for f in FIELDS}
    for pr in pairs:
        qv = field_values(resolve_image_path(DATA, pr["query_image"]))
        tv = field_values(resolve_image_path(DATA, pr["target_image"]))
        for f in FIELDS:
            if qv[f] == tv[f]:
                exact[f] += 1
            distinct_q[f].add(qv[f])
    n = len(pairs)
    print(f"\n  {'field':14s} {'q==t in true pair':>18s} {'distinct q values':>18s}")
    for f in FIELDS:
        print(f"  {f:14s} {exact[f]:>10d}/{n:<6d} {len(distinct_q[f]):>18d}")
    print("\n  KEY: a leak needs q==t HIGH (matches true pairs) AND distinct values "
          "HIGH (discriminates subjects). e.g. affine 350/350 + ~350 distinct = leak.")

    # bijection test on unlabelled pools (val + test)
    for split in ["val", "test"]:
        q = read_csv(f"dataset1/{split}_queries.csv")
        t = read_csv(f"dataset1/{split}_gallery.csv")
        analyze_pool(q, t, "query_id", "target_id", "query_image", "target_image", f"d1/{split}")


if __name__ == "__main__":
    main()
