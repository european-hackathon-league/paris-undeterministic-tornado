# STATUS

_Last updated: 2026-06-27_

## Headline

- **Best Kaggle public score: `0.90444`** — `submissions/mix_hung_d1template_d2template_d3grid.csv` — GOAL (0.9+) HIT.
- Session progression: `0.65137` → `0.77071` → `0.80127` → **`0.90444`**.
- Recipe: d1 = template normalization g44; d2 = template normalization g44;
  d3 = grid-intensity feature **no registration** (already-aligned). All + Hungarian.
- All wins are CLASSICAL, augmentation-free (only original 350 d1 pairs).

## Per-dataset MRR (current best mix)

| dataset | method | approx MRR | status |
|---|---|---|---|
| dataset1 | template normalization g44 + Hungarian | ~0.964 | near ceiling |
| dataset2 | template normalization g44 + Hungarian | ~0.749 | **now weakest** |
| dataset3 | grid feature no-register g44 + Hungarian | ~1.000 | solved |

Overall = (0.964 + 0.749 + 1.000) / 3 ≈ 0.904.

Two feature modes in `d2_template_retrieval.py`:
- default = register to same-modality template (for pose-shifted sets d1, d2).
- `--no-register` = raw downsampled grid feature (for already-aligned d3).
Both feed the d1-trained PCA/Ridge map + Hungarian. The grid-intensity feature is
far stronger than the old classical handcrafted features (drove d1 0.877->0.964
and d3 0.690->~1.0).

KEY: template normalization (register each volume to same-modality d1 mean
template) helps registered/deformed sets (d1, d2) but BREAKS d3 — d3 targets are
intraop, already in query space, so registering to a preop template misaligns
surgical changes (d3 MRR 0.690 -> 0.251 with template). Verified on Kaggle.

## What changed this session

1. Built `synthetic_d2_eval.py` + `d2_methods.py`: local synthetic-d2 validator.
   Applies independent rigid+elastic warps to labelled d1 pairs to mimic d2 and
   scores methods offline (Hungarian-recovery MRR). Calibrated so canonical ≈ 0.26
   matches real d2. **Use before spending Kaggle submissions.**
2. Discovered template normalization (synthetic 0.55 vs canonical 0.26). Built
   `d2_template_retrieval.py` (caches `.d2cache/`).
3. Verified on Kaggle: d2 partial 0.24963 (d2 MRR ~0.75); full mix 0.77071.
4. Augmented data (server): `train_pairs_aug_geom_contrast_1k.csv` = 850 pairs
   (350 clean + 500 geom+contrast aug). Confirmed valid + trainable.
5. Launched Track B: contrastive 3D training on augmented data on ROCm GPU
   (`contrastive_3d_train.py`, now supports `--train-pair-csv` + subject-aware
   holdout). Running in container `rocm` at `/app/runs/aug`.

## Track B (GPU augmented training) — DEAD END

Contrastive 3D dual-encoder trained on the 850 augmented pairs (subject-aware
split) reached only holdout_mrr ~0.04 / all_gallery ~0.009 by epoch 35 — far
below the classical 0.90. The scratch deep model does not learn competitive
embeddings even with geom+contrast augmentation. Killed to free GPU. The
augmented data could still be used to refit the classical PCA/Ridge map for
robustness, but the classical pipeline already dominates.

## In flight

- **Track A2**: template normalization on dataset1 + dataset3
  (`d2_template_retrieval.py --datasets dataset1 dataset3`). Tests whether the
  template trick also helps d3 (new bottleneck). Output:
  `submissions/d1d3_template_g44_hungarian.csv`.
- **Track B**: GPU contrastive training on 850 augmented pairs. Watching
  `holdout_mrr` / `all_gallery_mrr` vs the old failed run (~0.12).

## Next steps (priority order)

1. **dataset3** (biggest remaining lever): test template normalization; if no
   gain, try registration/robust-to-missing-tissue features (surgery removes
   anatomy). d3 0.69 -> target 0.8.
2. **Squeeze dataset2**: tune template grid (44->56/64), PCA components/alpha,
   add affine (not just rigid) registration, multi-start refinement.
3. **Track B**: if augmented contrastive beats classical on the synthetic/holdout
   proxy, blend or replace; else keep classical.
4. Re-assemble best all-template mix, validate, submit.

## Verified Kaggle scores (this session)

```
mix_hung_d1pca_d2template_d3pca.csv     0.77071   NEW BEST
mix_hung_d1pca_d2canon20_flipaug_d3pca  0.65137   prev best
d2_template_g44_hungarian.csv (partial) 0.24963   d2 MRR ~0.749
d2_canonical_pca20_c128_flipaug (part)  0.13029   d2 MRR ~0.391
```

## Commands

Run template d2:
```bash
.venv/bin/python d2_template_retrieval.py --datasets dataset2 --grid 44 --assignment \
  --out submissions/d2_template_g44_hungarian.csv
```

Local synthetic-d2 validation (no Kaggle needed):
```bash
.venv/bin/python synthetic_d2_eval.py --n-eval 60 --n-train 200 --grid 44 \
  --max-rot-deg 12 --max-shift 6 --elastic-sigma 8 --elastic-alpha 3 \
  --methods canonical template
```

Validate + submit:
```bash
.venv/bin/python classical_retrieval.py validate submissions/<file>.csv
kaggle competitions submit -c ehl-paris-medical-image-retrieval -f submissions/<file>.csv -m "<msg>"
```

## Submit log

- 2026-06-27 19:00 — `d2_template_g44_hungarian.csv` → **0.24963** (d2 MRR ~0.75).
- 2026-06-27 19:00 — `mix_hung_d1pca_d2template_d3pca.csv` → **0.77071** (best at the time).
- 2026-06-27 19:05 — `d1_template_g44_hungarian.csv` → **0.32147** (d1 MRR ~0.964; template helps d1).
- 2026-06-27 19:05 — `d3_template_g44_hungarian.csv` → **0.08368** (d3 MRR ~0.251; template HURTS d3).
- 2026-06-27 19:06 — `mix_hung_d1template_d2template_d3pca.csv` → **0.80127** (NEW BEST).

### Summary @ 0.80127
Swapped d1 from pca_ridge to template normalization (d1 MRR 0.877→0.964) on top of
the d2 template win. d3 kept on pca_ridge because template breaks it.

- 2026-06-27 19:11 — `d3_gridfeat_g44_hungarian.csv` → **0.33333** (d3 MRR ~1.0; grid feature, no register).
- 2026-06-27 19:12 — `mix_hung_d1template_d2template_d3grid.csv` → **0.90444** (NEW BEST, GOAL HIT).

- 2026-06-27 19:16 — `mix_hung_d1d2template_g56_d3grid.csv` → **0.85000** (REGRESSED; grid56 d1+d2 hurt real data).

- 2026-06-27 19:18 — `mix_hung_d1g44_d2g56_d3grid.csv` → **0.86609** (d2-only g56 also regressed).

### Summary @ 0.85000 / 0.86609 (grid56 dead end — BEST stays 0.90444)
grid56 hurts BOTH d1 and d2 on real data (full g56=0.850, d2-only g56=0.866, vs
g44 best 0.90444). The synthetic-d2 harness favored grid56 (+0.16) but it did NOT
transfer. LESSON: the synthetic harness is reliable for RELATIVE METHOD ranking
within d2-style distortion (it correctly picked template >> canonical), but NOT
for resolution/grid choices on real data — likely g56 overfits PCA with only 350
train pairs. grid44 is optimal. Locking 0.90444.

Open d2 levers (need real validation, synthetic unreliable for absolutes; goal
already met so these are bonus): affine (vs rigid) registration; refit PCA/Ridge
on the 850 augmented pairs for robustness; alpha/components tuning; multi-start
registration refinement.

### Summary @ 0.90444
d3 was solved by using the grid-intensity feature WITHOUT registration (d3 is
already aligned, so registration was the only thing holding it back; pca_ridge on
old classical features capped it at 0.69, grid feature → ~1.0). Combined with d1+d2
template normalization → overall 0.904.

Now d2 (~0.749) is the lone weak point. Synthetic-d2 harness shows template at
grid56 = 0.719 vs grid44 = 0.558 (+0.16) — regenerating d1+d2 at grid56 in flight
(`d1d2_template_g56_hungarian.csv`), expected to push d2 real MRR up materially.

Next steps:
1. Fold in grid56 d1+d2 (validated locally), re-submit. Maybe try grid64.
2. Augmentation (Track B GPU) not yet contributing; could also robustify the
   classical PCA/Ridge map by fitting on the 850 augmented pairs.
3. d2 affine (vs rigid) registration; multi-start refinement.
