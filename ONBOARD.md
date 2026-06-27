# Project Onboarding

This repository is for the Kaggle competition `ehl-paris-medical-image-retrieval`.
The active objective is to reach a public/private score of `0.9+`. Current best
verified public score is `0.80127` (`mix_hung_d1template_d2template_d3pca.csv`:
template normalization for d1+d2, pca_ridge for d3). See `STATUS.md` for the live
status, submit log, and next steps.

## Latest Breakthrough (template normalization for dataset2)

dataset2 was the bottleneck (MRR ~0.31-0.39). The fix: **template normalization**.
Because dataset1 train pairs are registered, the mean of the train queries (T1)
and the mean of the train targets (T2) form two templates that share one grid.
Rigidly registering each dataset2 volume to its same-modality template
(intramodal intensity NCC -> robust, unlike cross-modal) lands query and target
in a common frame where the dataset1 PCA/Ridge map works. This is O(N), not
O(N^2). It raised d2 MRR from ~0.39 to ~0.75 and overall from 0.65137 to 0.77071.

Key scripts:
- `synthetic_d2_eval.py` + `d2_methods.py`: local synthetic-d2 validator. Applies
  independent rigid+elastic warps to labelled d1 pairs to mimic d2, so methods
  can be scored offline (Hungarian-recovery MRR) WITHOUT spending Kaggle
  submissions. Calibrated: at rot12/shift6/elastic3 the canonical baseline scores
  ~0.26 (matches real d2), template scores ~0.55. The proxy correctly predicted
  the real win. USE THIS before submitting any new d2 method.
- `d2_template_retrieval.py`: production template normalization. Caches normalized
  features in `.d2cache/`. `--datasets dataset2 --assignment`.

Negative results from the harness (did NOT beat canonical, saved submissions):
pose-invariant histogram descriptors (too lossy), rigid edge-NCC pairwise
registration re-rank (elastic warp defeats rigid; local minima).

After this, **dataset3 (MRR ~0.69) is now the weakest dataset.**

## Workspace

- Repo root: `/Users/mikhail.fadin/paris`
- Dataset root: `ehl-paris-medical-image-retrieval/`
- Current branch: `main`
- Local Python env: `.venv/`
- Cached classical features: `.classical_cache/`
- Submission outputs: `submissions/`
- Runtime artifacts and large files should stay out of git.

The dataset CSVs reference `.nii.gz`, but the local files are often stored as
uncompressed `.nii`. Existing scripts handle this fallback.

## Dataset Structure

```text
ehl-paris-medical-image-retrieval/
  dataset1/
    train_pairs.csv
    val_queries.csv
    val_gallery.csv
    test_queries.csv
    test_gallery.csv
    images/
  dataset2/
    val_queries.csv
    val_gallery.csv
    test_queries.csv
    test_gallery.csv
    images/
  dataset3/
    val_queries.csv
    val_gallery.csv
    test_queries.csv
    test_gallery.csv
    images/
```

Task: for each T1 post-contrast query volume, rank all T2 gallery volumes from
the same dataset and split.

Counts:

```text
dataset1 train pairs: 350
dataset1 val/test:    40 / 100
dataset2 val/test:    40 / 100
dataset3 val/test:    20 / 77
full submission rows: 377
```

Dataset meaning:

- `dataset1`: registered preoperative pairs. This is the only labelled training
  set and is the source domain.
- `dataset2`: same source setting, but query and target were independently
  transformed with random rigid translation/rotation and nonlinear deformation.
  This is the main bottleneck.
- `dataset3`: preoperative-to-intraoperative pairs. Target is in roughly the
  same physical space, but surgery can remove/shift anatomy.

Metric:

```text
score = (dataset1_MRR + dataset2_MRR + dataset3_MRR) / 3
```

Partial submissions are useful. If submitting only one dataset, multiply the
displayed Kaggle score by `3` to estimate that dataset's public MRR.

## Environment

Offline classical env:

```bash
./setup_local_env_offline.sh
source .venv/bin/activate
python -c "import numpy, scipy, sklearn, nibabel; print('env ok')"
```

Use `.venv/bin/python` explicitly when running local scripts from automation:

```bash
.venv/bin/python classical_retrieval.py validate submissions/mix_hung_d1pca_d2canonpca20_d3pca.csv
```

GPU work has been done through the remote Jupyter machine. Do not save tokens,
passwords, or root credentials in repo files. If credentials are needed, recover
them from the user/chat context, not from committed files.

## Core Scripts

- `classical_retrieval.py`
  Main classical feature pipeline. Loads NIfTI, normalizes volumes, extracts
  features, caches them, trains PCA/Ridge, predicts rankings, and validates CSVs.

- `assignment_rerank.py`
  Applies one-to-one Hungarian assignment over score matrices. This was a major
  improvement and should be treated as part of the current baseline.

- `canonical_pca_retrieval.py`
  Dataset2-specific PCA-axis canonicalization. It canonicalizes foreground mask
  axes, extracts canonical features, trains PCA/Ridge on dataset1, and ranks d2.
  This is the best current dataset2 direction.

- `diagnostic_submissions.py`
  Cheap leakage/order/header/file-size diagnostics. These did not beat the main
  model, but document useful negative evidence.

- `brainiac_cosine_retrieval.py`, `brainiac_adapter_train.py`,
  `brainiac_patch_retrieval.py`
  BrainIAC experiments. Plain cosine, adapter training, and patch-token matching
  underperformed the classical/canonical pipeline.

- `contrastive_3d_train.py`
  Scratch 3D contrastive model with strong augmentation. Smoke-tested on GPU,
  but the first full run did not learn competitive embeddings.

- `mind_retrieval.py`, `lesion_retrieval.py`
  Dataset2 self-similarity/lesion-focused diagnostics. Both underperformed the
  canonical/PCA baseline.

- `jupyter_exec.py`
  Local helper for executing code on the remote Jupyter kernel.

## Verified Kaggle Scores

Current best full submission:

```text
submissions/mix_hung_d1pca_d2rankfusion_canon20_grid_d3pca.csv
public score: 0.62522
```

Important full scores:

```text
mix_hung_d1pca_d2rankfusion_canon20_grid...  0.62522  current best
mix_hung_d1pca_d2canonpca20_d3pca.csv        0.62491
mix_hung_d1pca_d2pca075grid025_d3pca.csv     0.59523
all_pca_ridge_c128_a100_hungarian.csv        0.59436
all_pca_ridge_c128_a100.csv                  0.55714
all_fusion_default_plus_pca_c128_a100.csv    0.50153
brainiac_cosine_submission.csv               0.15121
```

Important partial scores:

```text
d1_pca_ridge_c128_a100.csv                   0.29244 displayed, approx d1 MRR 0.87732
d2_canonical_pca20_c128_hungarian.csv        0.10383 displayed, approx d2 MRR 0.31149
d2_rankfusion_canon20_080_pca075grid025...   0.10414 displayed, approx d2 MRR 0.31242
d2_canonical_pca24_c128_hungarian.csv        0.08255 displayed, approx d2 MRR 0.24765
d2_blend_pca075_grid025_hungarian.csv        0.07415 displayed, approx d2 MRR 0.22245
d2_pca_ridge_c128_a100_hungarian.csv         0.07328 displayed, approx d2 MRR 0.21984
d3_pca_ridge_c128_a100_hungarian.csv         0.23016 displayed, approx d3 MRR 0.69048
```

The current bottleneck is still `dataset2`. Dataset1 and dataset3 are much
better after PCA/Ridge plus Hungarian assignment.

## Current Best Recipe

Current best full submission combines:

- dataset1: `pca_ridge_c128_a100` + Hungarian assignment
- dataset2: rank fusion of canonical PCA-axis `size=20` and old pca/grid blend,
  followed by Hungarian assignment
- dataset3: `pca_ridge_c128_a100` + Hungarian assignment

File:

```text
submissions/mix_hung_d1pca_d2rankfusion_canon20_grid_d3pca.csv
```

Validation:

```bash
.venv/bin/python classical_retrieval.py validate submissions/mix_hung_d1pca_d2rankfusion_canon20_grid_d3pca.csv
```

Submit:

```bash
kaggle competitions submit \
  -c ehl-paris-medical-image-retrieval \
  -f submissions/mix_hung_d1pca_d2rankfusion_canon20_grid_d3pca.csv \
  -m "mix hung d1 pca d2 rankfusion canon20 grid d3 pca"
```

## Ready But Not Yet Fully Used

These files are generated and valid. `d2_canonical_pca24...` has already been
submitted and is worse than `d2_canonical_pca20...`; keep it as negative
evidence, not as a current candidate.

```text
submissions/d2_canonical_pca24_c128_hungarian.csv
```

The rank-fusion dataset2 file has already been folded into the current best full
submission.

## Reproduction Commands

Generate the current best d1/d3 Hungarian baseline:

```bash
.venv/bin/python assignment_rerank.py \
  --method pca_ridge \
  --pca-components 128 \
  --pca-alpha 100 \
  --out submissions/all_pca_ridge_c128_a100_hungarian.csv
```

Generate the current best dataset2 canonical diagnostic:

```bash
.venv/bin/python canonical_pca_retrieval.py \
  --datasets dataset2 \
  --size 20 \
  --components 128 \
  --alpha 100 \
  --assignment \
  --out submissions/d2_canonical_pca20_c128_hungarian.csv
```

Validate a partial dataset2 submission:

```bash
.venv/bin/python classical_retrieval.py validate \
  --allow-partial submissions/d2_canonical_pca20_c128_hungarian.csv
```

Validate a full submission:

```bash
.venv/bin/python classical_retrieval.py validate \
  submissions/mix_hung_d1pca_d2rankfusion_canon20_grid_d3pca.csv
```

Check latest Kaggle scores:

```bash
kaggle competitions submissions -c ehl-paris-medical-image-retrieval
```

## What Worked

- PCA/Ridge cross-modal mapping trained on dataset1.
- One-to-one Hungarian assignment. This raised the full score from `0.55714` to
  `0.59436`.
- Dataset2 PCA-axis canonicalization. This raised d2 displayed partial score
  from about `0.074` to `0.104`.
- Dataset2 rank fusion of canonical20 with the older pca/grid blend. This gave
  a small but verified full-score increase from `0.62491` to `0.62522`.
- Dataset-specific mixing. Do not force one method across all three datasets.

## What Did Not Work

- Plain BrainIAC cosine: very low full score (`0.15121`).
- BrainIAC adapter training: poor holdout/all-gallery metrics.
- BrainIAC patch-token matching for d2: below classical/canonical baseline.
- Scratch 3D contrastive model: did not learn useful all-gallery retrieval in
  the first full run.
- Dataset2 order/header/file-size/sample-submission leakage checks: negative.
- MIND-like self-similarity and lesion-focused handcrafted descriptors: below
  canonical/PCA baseline.
- Fusion default plus PCA full submission: below plain PCA/Ridge Hungarian.
- Dataset2 canonical `size=24`: worse than canonical `size=20`.

## Work Principles For The Next Agent

1. Always validate CSVs before submitting to Kaggle.

2. Use partial dataset submissions to isolate where a change helps. Remember:
   displayed partial score times `3` approximates that dataset's MRR.

3. Be conservative with Kaggle submissions. Submit diagnostics only when they
   test a new hypothesis or a strong variant of a proven path.

4. Treat dataset2 as the main bottleneck. Work on rotation/deformation
   invariance, canonicalization, registration, or robust assignment before
   spending time on dataset1.

5. Keep dataset1 and dataset3 baselines stable unless there is specific evidence
   of a better method. Current d1/d3 are comparatively strong.

6. Do not commit large artifacts, datasets, caches, model checkpoints, tokens, or
   remote credentials. Keep source scripts and small CSV submissions only.

7. Prefer reproducible scripts over notebook-only work. If using remote GPU,
   copy the final script back into repo and document generated artifacts.

8. Do not trust local CV alone. Dataset2 and dataset3 are domain shifts; Kaggle
   partial diagnostics are the authoritative signal.

9. If continuing dataset2 work, first try:
   - add caching to `canonical_pca_retrieval.py` so size/sign/feature variants
     can be iterated faster;
   - test more canonical feature/rank blends before starting another deep model;
   - prioritize transforms around `size=20`, because `size=24` regressed.

10. Keep `ONBOARD.md`, `SUM.md`, and `SUBMISSION_PLAN.md` updated when scores or
    best files change.
