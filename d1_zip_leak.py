from __future__ import annotations

"""Zip archive-order / timestamp leak test, verified on d1 labels.

The zip stores decompressed .nii (no gzip mtime), but each entry has:
  - an archive ORDER (infolist position / header_offset)
  - a date_time (2-second resolution) + the raw file mtime if an extended
    timestamp extra field is present.

Hypothesis: organizers wrote each subject's files in a consistent order, so the
k-th query corresponds to the k-th gallery (write-order alignment), or a query
pairs to the gallery with the nearest timestamp. d1 has labels -> we can check
whether either recovers train_pairs exactly.
"""

import csv
import struct
import zipfile
from pathlib import Path

ZIP = Path("data/ehl-paris-medical-image-retrieval.zip")


def read_csv(p):
    with Path("data", p).open(newline="") as f:
        return list(csv.DictReader(f))


def ext_mtime(info: zipfile.ZipInfo):
    """Parse the unix mtime from an extended-timestamp extra field (0x5455) if present."""
    extra = info.extra
    i = 0
    while i + 4 <= len(extra):
        hid, sz = struct.unpack("<HH", extra[i:i + 4])
        body = extra[i + 4:i + 4 + sz]
        if hid == 0x5455 and len(body) >= 5:
            flags = body[0]
            if flags & 1:  # mtime present
                return struct.unpack("<i", body[1:5])[0]
        i += 4 + sz
    return None


def main():
    zf = zipfile.ZipFile(ZIP)
    infos = zf.infolist()
    # map nii.gz path in csv -> nii path in zip
    meta = {}  # zip_name -> (order_index, date_time tuple, ext_mtime, offset)
    for idx, info in enumerate(infos):
        meta[info.filename] = (idx, info.date_time, ext_mtime(info), info.header_offset)

    def zname(csv_path):
        p = csv_path
        if p.endswith(".nii.gz"):
            p = p[:-3]
        return p

    pairs = read_csv("dataset1/train_pairs.csv")
    print(f"=== d1 train: {len(pairs)} labelled pairs ===")
    miss = 0
    rows = []
    for pr in pairs:
        qn, tn = zname(pr["query_image"]), zname(pr["target_image"])
        if qn not in meta or tn not in meta:
            miss += 1
            continue
        rows.append((pr, meta[qn], meta[tn]))
    print(f"resolved {len(rows)}/{len(pairs)} (missing {miss})")

    # how many distinct ext_mtimes / date_times -> entropy
    qt_mtime = [r[1][2] for r in rows]
    print(f"ext_mtime present: {sum(1 for x in qt_mtime if x is not None)}/{len(rows)}; "
          f"distinct query ext_mtimes: {len(set(qt_mtime))}")

    # TEST 1: write-order alignment. Sort queries by archive order, sort gallery by
    # archive order; does the k-th query's true target equal the k-th gallery?
    q_all = read_csv("dataset1/train_pairs.csv")
    q_order = sorted(q_all, key=lambda r: meta[zname(r["query_image"])][0])
    g_order = sorted(q_all, key=lambda r: meta[zname(r["target_image"])][0])
    aligned = sum(1 for k in range(len(q_order)) if q_order[k]["pair_id"] == g_order[k]["pair_id"])
    print(f"\nTEST1 write-order alignment (k-th query == k-th gallery): {aligned}/{len(q_order)}")

    # TEST 2: timestamp nearest-neighbour. Pair each query to the gallery with the
    # closest ext_mtime (or date_time); count exact-pair recovery.
    def ts(info_tuple):
        return info_tuple[2] if info_tuple[2] is not None else \
            int(zipfile_datetime_to_epoch(info_tuple[1]))

    def zipfile_datetime_to_epoch(dt):
        import calendar
        return calendar.timegm((dt[0], dt[1], dt[2], dt[3], dt[4], dt[5], 0, 0, 0))

    gallery = [(r["pair_id"], ts(r2)) for r, _, r2 in rows]
    correct = 0
    deltas = []
    for pr, qm, tm in rows:
        qts = ts(qm)
        best = min(gallery, key=lambda g: abs(g[1] - qts))
        if best[0] == pr["pair_id"]:
            correct += 1
        # delta to the TRUE target
        deltas.append(abs(ts(tm) - qts))
    print(f"TEST2 timestamp-NN pair recovery: {correct}/{len(rows)}")
    import statistics
    print(f"   true-pair |Δtimestamp| sec: min={min(deltas)} median={statistics.median(deltas):.0f} "
          f"max={max(deltas)} (small/constant => pairing signal)")

    # TEST 3: header_offset adjacency within each folder (interleaving)
    # show first few entries in archive order to eyeball structure
    print("\nfirst 6 query entries in archive order (order, mtime, name):")
    qs = sorted(q_all, key=lambda r: meta[zname(r["query_image"])][0])[:6]
    for r in qs:
        m = meta[zname(r["query_image"])]
        print(f"  ord={m[0]:5d} mtime={m[2]} {r['query_image'].split('/')[-1]}")
    print("first 6 gallery entries in archive order:")
    gsl = sorted(q_all, key=lambda r: meta[zname(r["target_image"])][0])[:6]
    for r in gsl:
        m = meta[zname(r["target_image"])]
        print(f"  ord={m[0]:5d} mtime={m[2]} {r['target_image'].split('/')[-1]}")


if __name__ == "__main__":
    main()
