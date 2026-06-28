# Augmentation

Augmented training data was generated (on a GPU server) to test whether a refit /
trained model could beat the classical pipeline on the hard dataset2. Augmentation
is applied to the positive pairs in `dataset1/train_pairs.csv`. For each augmented
copy, T1 and T2 are transformed **independently** — this preserves the same-subject
positive pair but removes the perfect geometric match between modalities, mimicking
dataset2.

Generator: [`generate_augmented_train_data.py`](../generate_augmented_train_data.py).

## What was generated

| Item | Value |
|---|---|
| Original train pairs | 350 |
| Augmented pairs total | 1750 |
| Final train pairs in CSV | 2100 |
| Augmented NIfTI files | 3500 |
| Missing paths check | 0 |
| Unique pair ids | 2100 |

Recipe: for the first 250 source pairs, 5 augmented copies each; for the remaining
100 source pairs, 5 augmented copies each → `350 × 5 = 1750` augmented pairs,
`1750 × 2 = 3500` augmented NIfTI files.

## Augmentation sets

| Set | Pairs | Copies | Added |
|---|---|---|---|
| `geom_contrast` | 250 | 2 | original `aug00–aug01` |
| `geom_contrast_stronger` | 250 | 3 | `aug02–aug04` |
| `geom_contrast_stronger` | 100 | 5 | `aug00–aug04` for the remaining 100 |

## Transforms and ranges

| Transform | `geom_contrast` | `stronger` | Effect |
|---|---|---|---|
| Rotation | ±18° | ±22° | rotate 3D volume about x/y/z |
| Translation | ±12 vox | ±16 vox | shift volume in space |
| Scale | 0.90–1.10 | 0.85–1.15 | mild stretch/shrink |
| Elastic deformation | p=0.45, 3–8 vox | p=0.60, 4–10 vox | smooth nonlinear local displacement |
| Bias field | p=0.30 | p=0.40 | smooth brightness gradient (MRI inhomogeneity) |
| Gamma contrast | p=0.30, γ 0.75–1.35 | p=0.40, γ 0.75–1.35 | nonlinear contrast change |
| Intensity scale/shift | p=0.30 | p=0.40 | linear brightness change |
| Gaussian noise | p=0.30 | p=0.40 | small random noise |

## Result — augmentation did NOT transfer

This was an **honest negative result**:

- **Refitting the classical PCA/Ridge map** on the synthetic-augmented pairs *hurt*
  real d2: MRR **0.749 → 0.628** (overall 0.90444 → 0.86412). The harness predicted
  +0.11; it did not transfer — the map overfit the synthetic deform/contrast
  distribution.
- **Training a deep contrastive 3D dual-encoder** on the 850 augmented pairs
  (subject-aware split, ROCm GPU) reached only holdout MRR ≈0.04 by epoch 35 — far
  below the classical 0.90. Killed to free the GPU.

**Conclusion:** the registration-based classical pipeline (see
[`architecture.md`](architecture.md)) dominates. Only improvements to *registration*
(rigid → deformable template normalization) transferred to the real leaderboard;
data-distribution changes (augmentation refit, learned embeddings) did not.

> This matches the harness's known scope: trust it for **geometry / method ranking**,
> not for data-distribution choices.
