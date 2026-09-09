# ABVT3R — SSL/DINOv2 Biomass Pipeline (Stage A + Stage B)

Adapts ideas from Hao et al. 2025, *"Self-Supervised and Multi-Task Learning
Framework for Rapeseed Above-Ground Biomass Estimation"* (Agriculture
15(23):2516) to this project: DINOv2 self-supervised ViT features for
biomass regression, with a multi-task auxiliary target as a regularizer.
This is not the same setup as the paper — the differences and how each was
handled are called out explicitly below, not glossed over.

## Why two stages, not one

The paper trains on N=833 labeled top-down photos, unfreezes the entire ViT
backbone during fine-tuning, and pre-trains its own SSL corpus from ~5,000
public plant images. None of that transfers directly to our n=38 (see
`Pipeline.md` step 8a for why 38, not 41):

- **n=38 vs. n=833.** Even the paper calls 833 "small" in its own
  limitations. Full backbone fine-tuning at 22x less data is a serious
  overfitting risk — this project's own from-scratch ANN needed real
  debugging work to be trustworthy at this n (see `Pipeline.md` step 8a's
  target-normalisation note), and that network has ~50 parameters, not
  DINOv2 ViT-B/14's ~86M.
- **Capture geometry.** The paper uses one fixed-height top-down photo per
  plant. Our rig captures 12 (or 9, for a partial specimen) side/azimuthal
  views around the pot — no top-down view at all.
- **No dry weight.** `dataset/README.md` is explicit: fresh mass only, no
  drying step. The paper's MTL pairing (Fresh Weight + Dry Weight) isn't
  reproducible as given.
- **Compute.** This machine's GPU is a 4GB RTX 2050 — `CLAUDE_CONTEXT.md`'s
  own GPU table marks that tier "dev/testing only," not the RTX 4060 lab
  machine earmarked for DINOv2 work.

**Stage A** (this document's main content, done and validated below) sidesteps
all of this: use the public, frozen, pre-trained DINOv2 checkpoint as a
feature extractor — no fine-tuning, no from-scratch SSL pretraining, no
GPU-memory risk — and test whether its features carry biomass signal at all
on our data. **Stage B** is the fuller replication (fine-tuning, MTL), scoped
but not run — it needs the lab GPU machine and is a bigger commitment.

---

## Prerequisites (one-time)

```bash
cd ~/ABVT3R
source abvt310/bin/activate
pip install torch torchvision
```
No `--index-url` needed — plain `pip install torch torchvision` resolves a
CUDA-enabled build automatically when a compatible GPU is present (confirmed
working on the 4GB RTX 2050 here: torch 2.14.0+cu130, `torch.cuda.is_available()
== True`) and falls back to CPU otherwise. Inference-only workloads (Stage A)
run fine either way.

The DINOv2 backbone downloads from `torch.hub` on first use (~330MB from
`dl.fbaipublicfiles.com`, cached at `~/.cache/torch/hub/` afterward — only
needs internet access once).

---

## Stage A: frozen DINOv2 features vs. classical geometry

### What it tests

Does a DINOv2 ViT-B/14 — pre-trained by Meta on general web images, never
fine-tuned, never shown a single one of our plants — carry biomass-relevant
visual signal beyond what our procedure_alpha reconstruction's geometric
features (volume, surface area, height, ...) already capture? This is the
paper's core hypothesis (domain-general SSL features > hand-engineered
features), tested here without any of the fine-tuning risk.

### Step 1: extract embeddings

```bash
python neural_geometry/dino_features.py
```
For every specimen with RGB views under `acquisition/dataset/specimens/{id}/rgb/`,
encodes each view with the frozen backbone and mean-pools the per-view CLS
tokens into one 768-dim embedding per specimen (`DINOv2Encoder.encode_multiview`
in `neural_geometry/backbone/dinov2_encoder.py`). Caches both the aggregated
embedding and the per-view CLS tokens to `neural_geometry/dino_features/{id}.npz`.
Re-run any time new specimens are imported — already-cached ones are skipped
unless you pass `--force`. Pure inference: ~2.5s/specimen on the RTX 2050,
~2 minutes for all 41 specimens.

**Fix baked in while building this:** `DINOv2Encoder` asserts input
dimensions are exact multiples of its 14px patch size. Our capture
resolution (512×424) isn't a multiple of 14 in either dimension, so
`encode()` failed outright the first time it was run against real images
(this scaffold had architecture but had never actually been exercised on
real data). `_preprocess()` now resizes every view to a fixed 518×518
(37×14) before encoding — a standard DINOv2 inference resolution — so this
is fixed permanently, not a one-off workaround.

### Step 2: compare against the classical baseline

```bash
python biomass_engine/train_dino.py
```
Runs three LOOCV Random Forest configurations on the same primary n=38 set
`train_all.py` uses (excludes V009-V011 — see `Pipeline.md` step 8a):

1. **Geometric only** — the existing `RF_FEATURES` baseline (reference point).
2. **DINOv2 only** — the 768-dim embedding, PCA-reduced to *k* components.
3. **Geometric + DINOv2** — both concatenated.

PCA is refit inside every LOOCV fold on the training embeddings only — never
on the held-out specimen — at `n=37` training points and a 768-dim
embedding, skipping this would make the whole comparison meaningless (the
reduction itself would have seen the answer). Several `k` values are tried
(3/5/8/12) and reported side by side rather than picking one after the fact.
Results land in `evaluation_suite/reports/dino_stage_a_metrics.txt`.

### Results (run 2026-09-09, n=38)

| Config | R² | MAE (g) |
|---|---|---|
| Geometric only (baseline) | 0.451 | 330.1 |
| DINOv2 only, k=5 | 0.407 | 328.0 |
| **Geometric + DINOv2, k=5** | **0.455** | **320.2** |

Full table across all *k* in the report file. Two honest read on this:

- **DINOv2 alone — no 3D reconstruction, no depth camera, just 12 RGB
  crops through a frozen general-purpose ViT — comes within 0.04 R² of the
  entire classical reconstruction pipeline's result.** That's the
  headline finding: a model that has never seen a Kinect point cloud, never
  seen our pot/shoot split, never seen this species, gets most of the way
  to what months of classical-pipeline engineering achieves.
- **The combined result (0.455 vs. 0.451) is a marginal improvement, and at
  n=38 it is almost certainly within noise** — this project's own bootstrap
  CIs on RF (see `Pipeline.md` step 8a) run about ±0.2-0.3 wide at this
  sample size. Report it as "DINOv2 features are competitive with, not
  proven to beat, our engineered geometry" — consistent with how every
  other small-n result in this project has been handled, not as a win.

### What Stage A does *not* test

No fine-tuning happened — the backbone weights are frozen throughout, so
this says nothing about whether DINOv2 features could be fine-tuned to do
better (or worse, from overfitting) on our data. It also doesn't test the
paper's MTL regularization claim, which specifically requires a shared
backbone learning two correlated targets jointly during training — that
only makes sense once weights are actually being updated. Both require
Stage B.

---

## Stage B: fine-tuning (scoped, not run — needs the lab GPU machine)

### What already exists in the repo

- `neural_geometry/backbone/dinov2_encoder.py` — the `DINOv2Encoder` used
  above; supports unfreezing (`freeze=False`) for fine-tuning too.
- `neural_geometry/pretraining/dino_pretraining.py` — a from-scratch DINO
  SSL training loop, currently pointed at *our own* `SPECIMENS_DIR` images.
  **Don't use this as-is** — at ~450-500 total images (41 specimens × ~12
  views) it's far too small a corpus for effective from-scratch SSL. The
  paper's approach (curate ~5,000 public plant images, e.g. CVPPP + Plant
  Seedlings + VegAnn, explicitly excluding the target species) is the
  better model, and it's *cheaper* than what's scaffolded here: Stage A
  already shows the public `dinov2_vitb14` checkpoint alone is a strong
  starting point, so **skip re-pretraining entirely** unless a specific
  gap is found that domain-adaptive continued pretraining would fix.
- `neural_geometry/volumetric/volumetric_transformer.py` — a
  `VolumetricTransformer` with a biomass regression head already wired in
  (Level 2 / DeepVoxels architecture).
- `neural_geometry/sam3d/sam3d_pipeline.py` — segmentation/backprojection/
  soil-removal for Level 3.

None of these connect to `dataset/ground_truth.csv` or have a training loop
against real labels — there's no `train_level3.py` equivalent to
`biomass_engine/train_all.py` yet. That script is the main piece of new
work Stage B needs.

### Recipe, adapted from the paper for our constraints

1. **Backbone**: start from the public `dinov2_vitb14` checkpoint (already
   working, see Stage A) — do not attempt from-scratch SSL pretraining at
   our data scale.
2. **Input**: per specimen, all 12 (or 9) RGB views, not a single top-down
   photo — reuse `DINOv2Encoder.encode_multiview` from Stage A, but with
   `freeze=False` so gradients flow, or freeze the backbone entirely and
   fine-tune only a small head on top of the pooled embedding (a much safer
   starting point at n=38 than the paper's full-backbone unfreeze).
3. **MTL substitute**: the paper predicts FW+DW jointly. We don't have DW.
   Substitute `net_weight_g` (primary target) + `shoot_volume` (this
   project's own reconstructed geometric proxy, already computed by
   `procedure_alpha` — see `Pipeline.md` step 8a's feature-engineering
   note) as the second head. This is a genuine variant on the paper, not
   just a workaround — it uses information (3D reconstruction) the paper's
   authors didn't have, since they only had 2D RGB.
4. **Loss**: Smooth L1 per task (paper's choice, sensible for small-n
   regression with occasional outliers — see V010/V008's leverage
   diagnostics in `Pipeline.md` step 8a for why that matters here
   specifically), summed across tasks.
5. **Differential learning rate**: paper uses backbone LR 5e-5, head LR
   5e-4. Worth keeping if the backbone is unfrozen at all — but given n=38,
   strongly consider keeping the backbone fully frozen first (equivalent to
   Stage A's setup but with a proper trained MTL head instead of RF) before
   risking a full unfreeze.
6. **Validation protocol**: LOOCV, matching every other model in this
   project (train_all.py, train_dino.py) — the paper's 5-fold CV assumes
   enough data per fold to be stable; at n=38, LOOCV is the more defensible
   choice, same reasoning as `train_all.py`'s design.
7. **What to build**: `biomass_engine/train_dino_mtl.py` (or similar) —
   loads specimens the same way `train_dino.py` does, builds an MLP (or
   thin transformer head) on top of `DINOv2Encoder`'s pooled embedding with
   two output heads, trains with the paper's loss/LR recipe, evaluates via
   LOOCV. Doesn't exist yet.

### Before starting Stage B

- Move to the lab RTX 4060 machine (`CLAUDE_CONTEXT.md`'s GPU table) —
  fine-tuning gradients through even a frozen-backbone-with-trainable-head
  setup will want more headroom than 4GB, and a full unfreeze needs it.
- Re-read `Pipeline.md` step 8a's leverage/bootstrap-CI diagnostics before
  reporting any Stage B number — the same small-n caveats apply, more so
  once ~86M parameters (even mostly frozen) are in the loop.
- If more specimens get collected before this starts, re-run Stage A first
  — its result may shift meaningfully with a few more V-batch or new-species
  specimens, and that's a much cheaper signal to check than committing to a
  fine-tuning run.
