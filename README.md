# EHL Paris — Cross-modal Brain MRI Retrieval

Hackathon submission for the **EHL Paris: Medical Image Retrieval** Kaggle
competition. Given a query **T1 post-contrast** brain MRI, rank a gallery of **T2**
MRIs so the same subject ranks first, across three datasets of increasing
geometric difficulty.

**Final public score: `0.91981`** (up from a 0.651 baseline). The pipeline is
**fully classical** — no deep learning, no augmentation in the final model — built
on intensity normalization, registration to learned templates, a PCA+Ridge
cross-modal map, and Hungarian one-to-one assignment.

> Full method writeup: [`docs/architecture.md`](docs/architecture.md).
> Augmentation experiments (a negative result): [`docs/augmentation.md`](docs/augmentation.md).

---

## The problem

| Pool | Geometry | Labels |
|---|---|---|
| **dataset1** | registered preop pairs; query & target share a grid | 350 labeled pairs (the *only* training data) |
| **dataset2** | query & target **independently** rigid+elastic warped | none |
| **dataset3** | preop→intraop, target resampled into query space; surgery alters tissue | none |

Score = `(MRR1 + MRR2 + MRR3) / 3`. In each pool #queries == #targets and the
match is one-to-one — retrieval is a **linear assignment problem** (→ Hungarian).

## Pipeline

```
volume → intensity normalize → resample to 44³ grid → φ (flatten, center, L2)
                                      │
        ┌─────────────────────────────┼───────────────────────────────┐
   dataset1/2: register to            dataset3: NO registration
   same-modality template             (already aligned; registering hurts)
   (T1=mean d1 queries, T2=mean d1 targets)
   d2 also: deformable Demons + slab
        └─────────────────────────────┼───────────────────────────────┘
                                      │
            PCA(query) + Ridge → PCA(target)   ·   cosine similarity
                                      │
                    Hungarian assignment → per-query ranking
```

Key ideas (details in [`docs/architecture.md`](docs/architecture.md)):

1. **Intensity normalization** — robust percentile window, foreground mask, 44³ grid.
2. **PCA + Ridge cross-modal map** — learned on the 350 labeled d1 pairs; maps T1
   codes → T2 codes (`α=100`, 128 components).
3. **Template normalization** — rigidly register each volume to its same-modality
   d1 mean template (intramodal NCC, robust). The d2 fix: MRR 0.39 → 0.75; also
   lifts d1 0.877 → 0.964. **Skip it for d3** (already aligned — registering drops
   d3 to 0.251; raw grid feature → ~1.0 instead).
4. **Deformable Demons + slab fusion** — the real lever for d2's elastic warp;
   the only thing beyond rigid that *transferred* to the leaderboard
   (0.90444 → 0.91874 → 0.91981).
5. **Hungarian assignment** — exploit the bijection; assign then rank.

## Repo layout

| File | Role |
|---|---|
| `d2_template_retrieval.py` | Main entry: d1/d2/d3 retrieval via template normalization (`build_model`, `score_pool`). `--no-register` for d3. |
| `d2_methods.py` | Feature extractors, PCA+Ridge map (`_fit_pca_ridge`), rigid registration (`_register_to_template`). |
| `deform_lib.py` | The 0.91981 lever: deformable Demons (`_deformable_to_template`) + per-slab features + fusion. |
| `d2_deform_slab_submit.py` | Entry: builds the winning deformable+slab d2 part. |
| `alpha_mix.py` | Ridge-`α` mix builder (fast refit + rescore from cached features). |
| `synthetic_d2_eval.py` | Offline synthetic-dataset2 validator — rank methods without spending Kaggle submissions. |
| `generate_augmented_train_data.py` | Augmentation pipeline (see `docs/augmentation.md`). |
| `docs/` | `architecture.md`, `augmentation.md`. |
| `assets/` | Example query/target pairs for dataset2 and dataset3. |
| `sample_submission.csv` | Kaggle submission format reference. |
| `archive/` | Parked research and dead ends (see `archive/README.md`). |

## How to run

Requires Python ≥ 3.12. Scripts carry inline [uv](https://docs.astral.sh/uv/)
dependency headers (nibabel, numpy, scipy, scikit-learn), so `uv run <script>`
auto-installs deps. Place the competition data at
`ehl-paris-medical-image-retrieval/` (gitignored).

```bash
DATA=ehl-paris-medical-image-retrieval

# dataset1 + dataset2 — rigid template normalization
uv run d2_template_retrieval.py --data-root $DATA \
  --datasets dataset1 dataset2 --grid 44 --assignment \
  --out submissions/_part_d1d2_template.csv

# dataset2 — deformable Demons + slab (the winning d2 part)
uv run d2_deform_slab_submit.py --data-root $DATA --grid 44 \
  --smooth 3.0 --slab-k 10 --slab-s 16 \
  --out submissions/_part_d2_deformslab.csv

# dataset3 — grid feature, NO registration
uv run d2_template_retrieval.py --data-root $DATA \
  --datasets dataset3 --grid 44 --no-register --assignment \
  --out submissions/_part_d3_grid.csv

# Final mix = d1 (template) + d2 (deform+slab) + d3 (grid), concatenated.
```

Validate methods offline before spending a Kaggle submission:

```bash
uv run synthetic_d2_eval.py --n-eval 60 --n-train 250 --grid 44 \
  --max-rot-deg 12 --max-shift 6 --elastic-sigma 8 --elastic-alpha 3 \
  --methods canonical template
```

Normalized features are cached in `.d2cache/`; deformable fields in `.deformcache/`
(both regenerable, gitignored).

## Results

| Submission | Public | Note |
|---|---|---|
| canon20 flipaug (prior best) | 0.65137 | baseline |
| d1 pca · d2 **template** · d3 pca | 0.77071 | d2 fix |
| d1 **template** · d2 template · d3 pca | 0.80127 | template helps d1 |
| d1 template · d2 template · d3 **grid** | 0.90444 | grid for d3 |
| d1 template · d2 **deformable** · d3 grid | 0.91874 | elastic reg transfers |
| d1 template · d2 **deform + slab** · d3 grid | **0.91981** | **best** |

Per-dataset MRR at best mix: `MRR1 ≈ 0.964`, `MRR2 ≈ 0.75`, `MRR3 ≈ 1.0`.

## What didn't work (see `archive/`)

- **Deep contrastive 3D model** on augmented pairs — holdout MRR ≈0.04, far below
  classical 0.90.
- **Augmentation refit** of the classical map — *hurt* real d2 (0.749 → 0.628).
- **BrainIAC foundation-model embeddings** (cosine/adapter/patch) — below classical.
- **Pose-invariant descriptors**, **SIFT-Rank**, **MIND**, **deedsBCV re-rank**,
  **Sinkhorn / fusion reranking**, **grid 56** — all parked.

Lesson: only **registration** improvements transferred to the real leaderboard;
reranking/fusion and data-distribution changes did not.
