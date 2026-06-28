# NEXT_AGENT.md

_Last updated: 2026-06-28 (Claude session)_

## Goal (active)

Reach **0.95 public** on Kaggle (`ehl-paris-medical-image-retrieval`). Current
verified best = **0.90444**. The only lever is **dataset2** (d1≈0.964, d3≈1.0 are
maxed; d2≈0.749 is the bottleneck). 0.95 overall needs **d2 ≈ 0.886**.

User directives this session (priority):
- **Do NOT pursue BrainIAC / any deep encoder.** User rejected it: a foundation
  encoder optimizes the wrong invariance (washes out the subject fingerprint we
  need for same-subject retrieval). Server/BrainIAC track is OFF.
- Push d2 with **classical** ideas only. Target the user named: d2 → 0.93.
- Submit via `kaggle` CLI once a candidate beats 0.904.

## TWO HARD BLOCKERS (need the user)

1. **Kaggle credentials are missing.** No `~/.kaggle/kaggle.json`, no env vars,
   nothing in repo. `kaggle` CLI (2.2.3) is installed in `.venv` but cannot
   authenticate. **Nothing can be submitted until the user drops
   `~/.kaggle/kaggle.json` in place.** All candidate CSVs are staged and validated
   (see below) — submission is a one-liner once creds exist.
2. (Resolved/dropped) Server access was denied by the sandbox classifier AND the
   user told us to drop the deep track, so ignore the server.

## Environment notes (important)

- **Data root is `data/`** (NOT `ehl-paris-medical-image-retrieval/`). All scripts
  default to the wrong path — always pass `--data-root data`.
- `.venv` (Python 3.14) was missing `scipy`, `scikit-learn`, `requests`,
  `websocket-client`; all now installed.
- `.d2cache/` is populated (754 normalized features at g44) → regenerating d2/d1
  partials is now FAST (no re-registration).
- Validator arg order: `python classical_retrieval.py --data-root data validate <csv>`
  (global `--data-root` goes BEFORE the `validate` subcommand).

## Staged submissions (validated, 377 rows, 0 errors)

- `submissions/mix_hung_d1template_d2template_d3grid.csv` — **the 0.904 floor**
  (d1 template + d2 template + d3 grid-noreg, all Hungarian). Submit FIRST to
  confirm reproduction.
- `submissions/mix_d1template_d2sinkhorn_d3grid.csv` — **candidate**: same but d2
  uses Sinkhorn re-ranking. Differs from floor on **32/140 d2 top-1 assignments**,
  d1/d3 untouched. Submit to measure if Sinkhorn helps real d2.

## What was tried this session

### NEW WIN: Sinkhorn re-ranking of the d2 score matrix (parameter-free)

The d2 MRR lever is the PERMUTATION (Hungarian forces assigned→rank1, so alpha and
residual ranking are inert — confirmed in STATUS). Sinkhorn normalizes the cosine
matrix into a doubly-stochastic (soft-bijection) matrix BEFORE Hungarian.

Production-equivalent eval (`d2_prod_eval.py`, Hungarian-boost MRR, n_eval=100,
calibrated synthetic d2: rot12/shift6/elastic σ8 α3):

| method | boost_mrr |
|---|---|
| template (baseline; ≙ real d2 0.749) | 0.614 |
| template_sinkhorn τ=0.05 | **0.656** |
| template_sinkhorn τ=0.03 / 0.10 | 0.647 / 0.647 (robust across τ) |
| template_dc (double-center) | 0.624 |
| slab (rigid per-height map) | 0.629 |
| template + slab (fusion) | ~0.658 |
| slab_flex (windowed z-match) | 0.525 ❌ |
| meanpool over reg-hypotheses | 0.421 ❌ |
| max over reg-hypotheses (mh_template, harness raw) | 0.437 ❌ |

**Why Sinkhorn is trustworthy** (unlike grid56/augfit which regressed on real
data): it is a **parameter-free re-normalization at inference** — it cannot overfit
the 350 training pairs. Robust across τ. This is the category STATUS says transfers.
Estimated real-d2 effect: **0.749 → ~0.79** (relative +6.8%). Real-d2 effect is
UNCONFIRMED (blocked on Kaggle creds).

### NEGATIVE results (do NOT repeat)

- **Multi-hypothesis registration** (`mh_template`, max cosine over K frames): 0.44
  < 0.56. Max inflates impostors.
- **Mean-pool** embeddings over reg hypotheses: 0.42. Bad frames dilute the mean.
- **Flexible z slab matching** (best target slab in window): 0.525. Same impostor-
  inflation failure mode as max/meanpool. ANY "best-match maximization" hurts.
- (From STATUS, earlier sessions) affine/deformable reg, grid56, synthetic+real
  augmentation map refit, rich features, alpha tuning — all regressed or neutral.

### The user's slab idea — partial merit

Split volume into 10 axial slabs, per-slab cross-modal PCA/Ridge. RIGID slab
matching (0.629) slightly BEATS whole-volume template (0.614) and FUSES well with
template (~0.658). The FLEXIBLE-window variant fails. Worth keeping the rigid slab
representation as a fusion signal; drop the flexible matching.

## In flight at handoff

Final stack eval running in background → results in `/tmp/final_stack_eval.txt`:
```
.venv/bin/python d2_prod_eval.py --data-root data --n-eval 100 --n-train 250 \
  --grid 44 --max-rot-deg 12 --max-shift 6 --elastic-sigma 8 --elastic-alpha 3 \
  --methods template template_sinkhorn template_slab template_slab_sinkhorn slab_sinkhorn
```
Testing whether fusion(template+slab)→Sinkhorn STACKS above 0.656. **Read that file
first.** If `template_slab_sinkhorn` clearly beats `template_sinkhorn`, wire that as
the d2 method instead.

## Honest assessment of the 0.93 / 0.95 goal

**0.93 on d2 is very likely NOT reachable by re-ranking.** Best classical lever
(Sinkhorn) gets ~+0.04 → d2 ~0.79 → overall ~0.917. Stacking slab may add a little.
The +0.18 jump to 0.93 would require fundamentally better cross-modal similarity
under elastic warp, i.e. actually UNDOING the deformation.

**Highest-ceiling untried classical idea: proper regularized deformable
registration** (ANTs/elastix/SimpleITK-SyN class, intramodal volume→template).
d2's distortion is literally rigid+elastic; a good non-linear registration inverts
it and could plausibly reach the 0.95+ that (per user) others hit. STATUS's
"deformable washes shape" was a crude home-grown attempt; a properly *regularized*
low-DOF deformable is different. NOT YET TRIED here. This is the recommended next
big swing. (SimpleITK is not installed — would need install, may need network.)

## Recommended next steps (priority)

1. **Get `~/.kaggle/kaggle.json` from user**, then submit (one at a time, validate
   first):
   ```
   .venv/bin/python classical_retrieval.py --data-root data validate <csv>
   .venv/bin/kaggle competitions submit -c ehl-paris-medical-image-retrieval -f <csv> -m "<msg>"
   ```
   Order: (a) the 0.904 floor to confirm; (b) the Sinkhorn candidate to measure d2.
2. Read `/tmp/final_stack_eval.txt`; if the slab stack wins, regenerate the d2
   partial with that method and rebuild the candidate mix.
3. Bigger swing for 0.93+: implement regularized deformable registration to the
   template; validate on `d2_prod_eval.py` BEFORE submitting. Beware: only LARGE,
   distribution-independent gains transfer (hardened lesson — see STATUS).

## Key files added/changed this session

- `d2_methods.py` — added: `_double_center`, `_sinkhorn`, `_template_scores`
  (caches S), `mh_template`, `template_dc`, `template_sinkhorn` (+τ variants),
  `meanpool` variants, `slab`/`slab_flex`/`slab_sinkhorn`/`template_slab`(+sinkhorn).
- `d2_template_retrieval.py` — added `--rerank {none,dc,sinkhorn}` (applies to the
  score matrix before Hungarian). Used to make the Sinkhorn candidate.
- `d2_prod_eval.py` — NEW. Reports **Hungarian-boost MRR** (the real submission
  metric), not raw_mrr. Use this for all d2 method decisions.
- Partials: `submissions/_part_d1d2_template_g44.csv`,
  `_part_d2_sinkhorn_g44.csv`, `_part_d3_grid_g44.csv`.

## Reproduce the staged mixes

```bash
# floor partials (d1+d2 template, d3 grid) — fast now (.d2cache warm)
.venv/bin/python d2_template_retrieval.py --data-root data --datasets dataset1 dataset2 \
  --splits val test --grid 44 --assignment --out submissions/_part_d1d2_template_g44.csv
.venv/bin/python d2_template_retrieval.py --data-root data --datasets dataset3 \
  --splits val test --grid 44 --assignment --no-register --out submissions/_part_d3_grid_g44.csv
# sinkhorn d2 partial
.venv/bin/python d2_template_retrieval.py --data-root data --datasets dataset2 \
  --splits val test --grid 44 --assignment --rerank sinkhorn --out submissions/_part_d2_sinkhorn_g44.csv
# assemble: d1 = first 140 data rows of the d1d2 partial, then d2 (sinkhorn or template), then d3
```
