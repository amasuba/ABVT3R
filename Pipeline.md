# ABVT3R — Reconstruction Pipeline: Step-by-Step

How to go from field-collected Kinect data to a 3D mesh + reconstruction
stats + biomass estimate + evaluation metrics, entirely from the terminal.

## Pipeline architecture

Two independent reconstruction methods feed a shared biomass-regression
stage, plus an evaluation arm that compares them to each other (not to
ground-truth geometry — see step 8). Solid arrows are data dependencies;
each stage's outputs are consumed as inputs by the next.

```
                      dataset/plants/{ID}/{images,depth}/  (raw Kinect capture)
                      dataset/ground_truth.csv              (scale weights)
                                    │
                                    ▼
                 acquisition/dataset/import_farm_dataset.py      [step 1]
                                    │
                    ┌───────────────┴────────────────┐
                    ▼                                 ▼
   acquisition/dataset/specimens/{ID}/      acquisition/dataset/ground_truth/
     {rgb,depth}/*.npy                        registry.csv
                    │                                 │
                    ▼                                 │
   ╔══════════════════════════════╗                   │
   ║ CLASSICAL ARM (procedure_alpha) ║                 │
   ║  depth → point cloud → SOR/MLS  ║  [step 2]       │
   ║  → coarse+ICP registration      ║                 │
   ║  → voxel meshing → pot/shoot    ║                 │
   ║    segmentation                 ║                 │
   ╚══════════════════════════════╝                   │
                    │                                 │
                    ▼                                 │
    procedure_alpha/outputs/                          │
      mesh_specimen_{ID}.ply, reconstruction_stats_*.txt
                    │                                 │
                    ├─────────────────┬────────────────┤
                    ▼                 ▼                ▼
      ╔═══════════════════╗   evaluation_suite/   biomass_engine/
      ║ NeRF ARM           ║   efficiency_report.py  train_mango.py
      ║ (neural_geometry)  ║   [step 8b]              predict_batch.py
      ║  build_transforms.py ║                        [step 5, 8a]
      ║  (reuses procedure_alpha's own  ║                    │
      ║   ICP-composed poses — no        ║                    ▼
      ║   independent calibration)       ║          biomass_engine/trained/
      ║  → ns-train nerfacto  [step 7]  ║            RF_model_mango,
      ║  → ns-eval (PSNR/SSIM/LPIPS)    ║            ANN_model_mango
      ║    [step 8c]                    ║                    │
      ║  → ns-export pointcloud         ║                    ▼
      ╚═══════════════════╝          biomass_engine/visualisation/
                    │                  results_dashboard.py  [step 6]
                    ▼                  evaluation_suite/comparison.py
      evaluation_suite/geometry_comparison.py
        Chamfer / F-score / HD95 / Normal Consistency
        — CLASSICAL mesh vs. NeRF point cloud, method
          AGREEMENT only, no ground-truth geometry exists
          [step 8d]
```

Two things worth internalising before running anything:

- **The NeRF arm never touches the classical arm's registration** except by
  reusing its already-computed camera poses (`build_transforms.py` derives
  poses from `procedure_alpha`'s own coarse+ICP math, not an independent
  calibration) — so the two arms are genuinely separate reconstructions of
  the same 12 views, not one feeding the other's geometry.
- **Only the biomass-regression stage has real ground truth** (measured
  plant weight, in `dataset/ground_truth.csv`). The geometry-comparison
  stage (step 8d) has no independently-scanned reference mesh, so its
  metrics measure agreement between the two methods, not accuracy —
  see step 8d for why this distinction matters and how it's reported.

## 0. Prerequisites (one-time)

All of this runs inside the `abvt310` virtualenv (Python 3.10) — it's the
one with `pylibfreenect2` and now `open3d` installed.

```bash
cd ~/ABVT3R
source abvt310/bin/activate
```

Everything below assumes this venv is active.

## 1. Import field-collected data into the pipeline's layout

Field captures land in `dataset/plants/{plant_id}/{images,depth}/cam{A,B}_{angle:03d}.png`
(the dual-Kinect rig format) plus `dataset/ground_truth.csv`. The
reconstruction pipeline instead reads from
`acquisition/dataset/specimens/{specimen_id}/` as `.npy` arrays. Bridge the
two with:

```bash
python acquisition/dataset/import_farm_dataset.py
```

This converts every plant folder under `dataset/plants/` into
`acquisition/dataset/specimens/{plant_id}/{rgb,depth}/` and merges
`dataset/ground_truth.csv` into `acquisition/dataset/ground_truth/registry.csv`.

To import a single plant instead of all of them:

```bash
python acquisition/dataset/import_farm_dataset.py --plant M001
```

## 2. Run the 3D reconstruction

The rig protocol is: 6 manual repositions, two Kinects mounted 180° apart,
both firing at each stop — 12 true viewing angles from 6 physical moves.
That's the **dual-camera** protocol, so pass `--dual`:

```bash
python run_pipeline.py alpha --specimen M001 --dual
```

Run it for every imported plant:

```bash
for p in M001 M002 M003 M004 M005; do
    python run_pipeline.py alpha --specimen "$p" --dual
done
```

Or let the runner do all specimens under `acquisition/dataset/specimens/`
at once — it auto-detects which protocol each specimen used from its
`metadata.json`, so this is safe even with a mix of old and new data:

```bash
python run_pipeline.py alpha --all
```

Each run takes roughly 2.5–4 minutes (preprocessing → ICP registration →
mesh generation). Progress prints to the terminal as it goes.

## 3. Find the outputs

Everything lands in `procedure_alpha/outputs/`, named by specimen:

| File | What it is |
|---|---|
| `reconstruction_stats_specimen_{ID}.txt` | Human-readable summary: volume, surface area, height/width/depth, mesh quality, ICP RMSE per view, pot/shoot split, view legend |
| `merged_cloud_specimen_{ID}.ply` | The fused, coloured point cloud (coloured by height — see note below) |
| `merged_cloud_byview_specimen_{ID}.ply` | Same fused cloud, coloured by **which of the 12 captures** each point came from — shows how the individual views merged |
| `merged_cloud_segmented_specimen_{ID}.ply` | Same fused cloud, coloured **pot (brown) vs shoot (green)** — see segmentation note below |
| `mesh_specimen_{ID}.ply` / `.obj` | The final reconstructed mesh — **uncoloured/grey**, by design (see note below) |
| `final_vertices_specimen_{ID}.npy` / `final_triangles_specimen_{ID}.npy` | Raw mesh arrays |
| `merged_points_specimen_{ID}.npy` | Fused point cloud before meshing, as raw XYZ (no colour) |
| `surface_normals_specimen_{ID}.npy` | Per-vertex normals |

**Pot/shoot segmentation:** there's no reliable colour signal to separate
plant from pot on this rig (see below), so the split is geometric — it finds
the height (Y) band with the lowest point density between the pot cluster
and the canopy above, restricted to a plausible transition zone (15-60% up
the total height). It's a heuristic, not true segmentation, and can break
on a specimen with a bad capture (e.g. one anomalously noisy/misregistered
view skews the whole height range) — always sanity-check
`merged_cloud_segmented_specimen_{ID}.ply` visually and the "Pot height" /
"Shoot height" stats before trusting the shoot-only volume for biomass.
The stats file's "Shoot volume" is the more appropriate figure to compare
against ground-truth net (above-ground) weight — the plain "Volume" figure
includes the pot and soil.

**Why the mesh has no colour:** this pipeline is a reproduction of Odwa
Nombambela's thesis methodology, and his own report documents that Kinect
RGB texture mapping onto the mesh failed due to infrared interference from
the depth sensors — it's listed as an unsolved problem, with stereo cameras
suggested as the eventual fix. The "coloured" renders in his report's
appendix are the **point clouds**, coloured by height with a viridis-style
gradient (dark at the pot, bright green/yellow at the canopy top) — not
real photographed colour. `merged_cloud_specimen_{ID}.ply` reproduces that
exact look; the mesh stays grey to match his results faithfully.

Quick look at the numbers:

```bash
cat procedure_alpha/outputs/reconstruction_stats_specimen_M001.txt
```

## 4. View the results

**Coloured point cloud** (this is the one that looks like the thesis
appendix — height-coloured, green canopy over a dark pot):

```bash
python3 -c "
import open3d as o3d
pcd = o3d.io.read_point_cloud('procedure_alpha/outputs/merged_cloud_specimen_M001.ply')
o3d.visualization.draw_geometries([pcd])
"
```

**By-view coloured cloud** (see how the 12 individual captures merged —
each of the 12 views gets its own distinct colour):

```bash
python3 -c "
import open3d as o3d
pcd = o3d.io.read_point_cloud('procedure_alpha/outputs/merged_cloud_byview_specimen_M001.ply')
o3d.visualization.draw_geometries([pcd])
"
```
Cross-reference colours to angles/cameras using the "View Legend" section
at the bottom of that specimen's `reconstruction_stats_*.txt`.

**Pot/shoot segmented cloud** (brown pot vs green shoot):

```bash
python3 -c "
import open3d as o3d
pcd = o3d.io.read_point_cloud('procedure_alpha/outputs/merged_cloud_segmented_specimen_M001.ply')
o3d.visualization.draw_geometries([pcd])
"
```

**Mesh** (grey — see the note in step 3 on why):

```bash
python3 -c "
import open3d as o3d
mesh = o3d.io.read_triangle_mesh('procedure_alpha/outputs/mesh_specimen_M001.ply')
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])
"
```

All windows: left-drag to rotate, scroll to zoom, right-drag to pan.

Alternatively, open the `.ply` or `.obj` file in any external tool
(MeshLab, CloudCompare, Blender).

## 5. Optional: biomass prediction dashboard

Once reconstruction stats exist for your specimens, the trained RF/ANN
models can turn them into biomass estimates:

```bash
python run_pipeline.py dashboard
```

## 6. Optional: cross-method evaluation report

Compares reconstruction-derived estimates against the ground-truth
registry:

```bash
python run_pipeline.py evaluate --export results/comparison.pdf
```

## 7. Experimental: NeRF comparison arm (Nerfstudio)

A second, independent reconstruction method to compare against the
classical `procedure_alpha` pipeline — trains a neural radiance field
(Nerfacto) directly from the RGB images, sidestepping the "RGB texture
mapping failed" limitation entirely since NeRF learns colour and geometry
jointly rather than projecting colour onto a depth-derived mesh.

This lives in a **separate venv**, `abvt_nerf/` (Python 3.10 + PyTorch
2.5.1+cu121 + Nerfstudio 1.1.5), already set up. To recreate it from
scratch on another machine:

```bash
python3.10 -m venv abvt_nerf
source abvt_nerf/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install nerfstudio
```

There are **two ways to get camera poses**, since there's no checkerboard
calibration data and the cameras are hand-repositioned rather than fixed:

### 7a. Assumed-geometry poses (no calibration needed)

Derives poses directly from the same rotation/centroid math
`procedure_alpha`'s own registration already computes (see
`neural_geometry/nerf/build_transforms.py` docstring for the derivation).

```bash
source abvt310/bin/activate   # build_transforms.py only needs procedure_alpha's deps
python neural_geometry/nerf/build_transforms.py --specimen M001
```
Writes `neural_geometry/nerf_data/M001_assumed/{transforms.json,images/}`.
Takes a few minutes — it now runs full coarse+ICP registration internally
(see below for why).

**Important fix (2026-07-26):** an earlier version of this script only used
`procedure_alpha`'s *coarse* registration stage (fixed rotation by nominal
capture angle), not the *fine ICP correction* that follows it. That's fine
for classical point-cloud fusion (ICP cleans up the residual either way),
but NeRF has no equivalent self-correction — a 50k-iteration test run on
the coarse-only poses came out **worse** than a 1000-iteration pilot (point
cloud extent nearly doubled, floaters got worse not better with more
training), which is what exposed the bug. Some views needed a genuine
~20-35° ICP rotation correction (hand rotation isn't exactly 30°, and the
two physical cameras aren't exactly 180° apart) — not noise. The script now
composes both registration stages, and this was verified numerically:
applying the derived pose to the raw preprocessed cloud reproduces
`procedure_alpha`'s actual registered point positions to floating-point
precision (~1e-15 max diff). If you have any transforms.json / trained
checkpoints from before this fix, treat their reconstructions as unreliable
and regenerate.

Train (switch to the NeRF venv). The GPU here has only ~4GB VRAM, so two
flags exist purely to dodge OOM, not to change training quality:
`--steps-per-eval-image`/`--steps-per-eval-all-images` are pushed out
because the periodic full-image eval render (not training itself)
overflows memory otherwise, and `--pipeline.model.eval-num-rays-per-chunk
4096` (default 32768) shrinks the render chunk size so the **final**
`ns-eval` run (step 8c) also fits in 4GB:

```bash
source abvt_nerf/bin/activate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ns-train nerfacto \
  --data neural_geometry/nerf_data/M001_assumed \
  --output-dir neural_geometry/nerf_outputs \
  --experiment-name M001_assumed_fixed_full \
  --max-num-iterations 30000 \
  --pipeline.datamanager.train-num-rays-per-batch 2048 \
  --pipeline.model.eval-num-rays-per-chunk 4096 \
  --viewer.quit-on-train-completion True \
  --vis tensorboard \
  --steps-per-eval-image 100000 \
  --steps-per-eval-all-images 100000 \
  --steps-per-save 5000 \
  nerfstudio-data
```
At ~0.4s/iteration this is ~3.3 hours for 30,000 iterations. For a quick
plumbing check instead (does the run start, do losses look sane — **not**
enough for the geometry-comparison arm in step 8d), drop
`--max-num-iterations` to `3000` (~20 min) and use a distinct
`--experiment-name` so you don't overwrite the full run.

**Update (2026-07-26): 30,000 iterations does not fix this — don't spend
the 3.3 hours expecting it to.** A first pass at 3,000 iterations
(PSNR=12.09, SSIM=0.303) exported a point cloud that was >90% floater
noise with no coherent cluster near the plant. The full 30,000-iteration
run came back nearly identical (PSNR=11.72, SSIM=0.311, 94% floater
discard, and the largest surviving cluster still >0.5m from the actual
plant location). Pulling the training-time PSNR curve from the tensorboard
logs explains why: **train PSNR reached 28-30dB by the end of the run
(near-perfect fit to the 12 training images) while eval PSNR on the
held-out view stayed flat at ~11.6-12.6dB the entire time, from step 500
to step 29,500.** That divergence is the textbook signature of sparse-view
overfitting — nerfacto memorised each training view individually via
near-camera "floater" density rather than learning consistent 3D
structure, so training longer just fits the training views harder without
ever improving generalisation. More iterations will not help; this is a
view-count/baseline problem, not an optimisation-budget problem, and it's
corroborated independently by step 7b's COLMAP note above predicting the
same 12-view wide-baseline capture would be a hard case for feature-based
SfM too. See step 8d for what this means for the geometry-comparison arm.

### 7b. COLMAP-estimated poses (real pose recovery, may fail)

Install COLMAP first (needs your password, run in your own terminal):
```bash
sudo apt install colmap
```

Then let Nerfstudio's own pipeline estimate poses via COLMAP feature
matching on the same 12 images:
```bash
source abvt_nerf/bin/activate
ns-process-data images \
  --data dataset/plants/M001/images \
  --output-dir neural_geometry/nerf_data/M001_colmap
```
**Expect this to struggle or fail outright** — our 12 images are sparse
and wide-baseline (two physically separate cameras, 30° steps between
shots), which is a hard case for COLMAP's feature-matching to find enough
overlap to solve for poses. If it fails, that failure is itself the
informative result (confirms assumed-geometry poses are the only viable
path for this rig without denser capture). If it succeeds, train the same
way as 7a but pointing `--data` at `neural_geometry/nerf_data/M001_colmap`.

**Confirmed (2026-07-27): it fails outright**, exactly as predicted —
`colmap mapper` couldn't find a good initial image pair even after
relaxing its initialization constraints, and errored with "failed to
create sparse model" before ever reaching pose estimation. This is
useful, citable corroborating evidence for the dissertation: two
completely independent multi-view geometry methods (COLMAP's
feature-matching SfM here, and nerfacto's volumetric fitting in step 7a)
both fail to make sense of this same 12-view, wide-baseline capture —
converging evidence that the bottleneck is the capture protocol itself,
not a bug or weakness in either algorithm.

### Export & test results

**Point cloud** (needed before viewing — the viewer command below only
works after this has been run at least once):
```bash
source abvt_nerf/bin/activate
CONFIG=$(find neural_geometry/nerf_outputs/M001_assumed_fixed_full/nerfacto -name config.yml | sort | tail -1)
ns-export pointcloud \
  --load-config "$CONFIG" \
  --output-dir neural_geometry/nerf_outputs/M001_assumed_fixed_full/pointcloud \
  --num-points 200000 \
  --remove-outliers True \
  --normal-method open3d \
  --save-world-frame True
```
(`$CONFIG` auto-picks the **most recent** run folder Nerfstudio created
under `neural_geometry/nerf_outputs/M001_assumed_fixed_full/nerfacto/` —
don't hand-copy a timestamp into these commands: `<` and `>` are live
shell redirection operators, so pasting a literal `<TIMESTAMP>` placeholder
tries to redirect stdin/stdout to files named that, and fails with
`bash: TIMESTAMP: No such file or directory` instead of doing anything
useful. If you have multiple old runs under that folder and want a
specific one rather than the latest, list them with `ls
neural_geometry/nerf_outputs/M001_assumed_fixed_full/nerfacto/` and set
`CONFIG=neural_geometry/nerf_outputs/M001_assumed_fixed_full/nerfacto/<pick-one>/config.yml`
directly — just don't leave angle brackets in what you actually run.)
`--normal-method open3d` avoids needing to retrain with normal prediction
enabled. `--save-world-frame True` matters for step 8d — it undoes
Nerfstudio's internal auto-orient/auto-scale so the exported cloud lands
back in the same coordinate frame as the poses in `transforms.json`
(still Z-flipped relative to `procedure_alpha`'s own convention — the
comparison script handles that flip, see step 8d).

View it the same way as the classical point clouds:
```bash
python3 -c "
import open3d as o3d
pcd = o3d.io.read_point_cloud('neural_geometry/nerf_outputs/M001_assumed_fixed_full/pointcloud/point_cloud.ply')
o3d.visualization.draw_geometries([pcd])
"
```

**Quantitative render quality** (PSNR/SSIM/LPIPS on held-out views —
the actual "how good is this NeRF" metric, analogous to
`procedure_alpha`'s ICP RMSE / mesh quality score):
```bash
CONFIG=$(find neural_geometry/nerf_outputs/M001_assumed_fixed_full/nerfacto -name config.yml | sort | tail -1)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ns-eval \
  --load-config "$CONFIG" \
  --output-path neural_geometry/nerf_outputs/M001_assumed_fixed_full/eval_metrics.json
```
**Fixed (2026-07-26):** this used to OOM on the 4GB GPU rendering a full
held-out image at once, even with the memory-fragmentation flag. The fix
is `--pipeline.model.eval-num-rays-per-chunk 4096` at **train** time (baked
into the step-7 command above, default is 32768) — it shrinks the render
chunk size the checkpoint was configured with, so `ns-eval` inherits it
and no longer needs its own flag. Confirmed working at both training
budgets: 3,000 iterations gave PSNR=12.09, SSIM=0.303, LPIPS=0.754;
30,000 iterations gave PSNR=11.72, SSIM=0.311, LPIPS=0.698 — essentially
unchanged despite 10x the training (`*_std` fields come back `NaN` at
both — the eval split only holds out 1 image at this dataset size, so
there's no variance to report). These are genuinely poor scores (a
well-trained nerfacto scene is usually 25-30+ dB PSNR), and the fact that
10x more training didn't move them is itself the important result — see
the tensorboard note in step 7 (train PSNR hit 28-30dB while eval PSNR
stayed flat throughout): this is sparse-view overfitting, not an
undertrained or broken eval command.

**Interactive exploration** (fly around the trained scene in a browser):
```bash
CONFIG=$(find neural_geometry/nerf_outputs/M001_assumed_fixed_full/nerfacto -name config.yml | sort | tail -1)
ns-viewer --load-config "$CONFIG"
```

**Training curves** (loss/PSNR over the run, since training used
`--vis tensorboard`):
```bash
tensorboard --logdir neural_geometry/nerf_outputs/M001_assumed_fixed_full
```
Then open the printed `http://localhost:6006` URL.

## 8. Evaluation metrics suite (for the dissertation)

Four separate scripts, one per evaluation concern. They're independent —
run whichever ones you need, in any order, as long as their inputs exist
(reconstructions from step 2, a trained NeRF checkpoint from step 7).

### 8a. Extended biomass regression metrics

Beyond plain MAE/RMSE (which `evaluate` in step 6 already gives you),
this adds the metrics a regression-validation section of a thesis is
expected to have: Bias (systematic over/under-prediction), nRMSE (scale-
comparable RMSE), Lin's Concordance Correlation Coefficient (agreement
with the 1:1 line — catches scale error R² can miss), and a Bland-Altman
plot (does error grow with plant size?).

```bash
source abvt310/bin/activate
python biomass_engine/train_mango.py
```
Retrains RF and ANN on all 10 Mango specimens (leave-one-out CV, since
n=10 is too small for a held-out test split), saves the models to
`biomass_engine/trained/{RF,ANN}_model_mango/`, prints a metrics table,
and saves `evaluation_suite/figures/mango_bland_altman.png`. Treat the ANN
result with real skepticism — 10 samples is far below where an MLP can be
expected to generalise (LOOCV R² was strongly negative at last run; RF is
the more trustworthy of the two here, and even RF's R² was negative,
reflecting how little training data exists). This is itself worth stating
plainly in the dissertation rather than hidden — small-n regression is
exactly the kind of result the reference metrics doc's "small-n
statistics" trap warns about over-interpreting.

Run `python biomass_engine/predict_batch.py` afterward to refresh the
per-specimen `Biomass (RF)` / `Biomass (ANN)` lines in
`reconstruction_stats_specimen_*.txt` from the freshly retrained models,
then `python run_pipeline.py dashboard` (step 5) to see them plotted.

### 8b. Efficiency report

Reconstruction wall-clock, model complexity (tree/parameter counts), and
NeRF training throughput/VRAM — the "how expensive is each method"
counterpart to the accuracy metrics, and the natural place to answer cost
questions a reconstruction-architecture thesis invites.

```bash
python evaluation_suite/efficiency_report.py
```
Reads `Processing time` lines already appended to every
`reconstruction_stats_specimen_*.txt` (step 2 writes these automatically),
loads the trained RF/ANN models to count trees/nodes/parameters, and
reports the NeRF throughput/VRAM figures observed during step 7's training
runs. Saves to `evaluation_suite/reports/efficiency_report.txt`. Also
flags the **views-vs-accuracy** question (does reconstruction quality
degrade gracefully with fewer than 12 views, which matters directly for
this rig's two-camera hardware constraint) as a scoped follow-up — not run
automatically, but supported: `ProcedureAlpha.run_specimen_dual(specimen_id,
half_angles_deg=<subset>)` takes an explicit angle subset if you want to
run that experiment.

### 8c. NeRF appearance metrics

Covered in step 7's "Export & test results" — `ns-eval` gives PSNR/SSIM/
LPIPS. Repeated here only as a pointer since it's as much an evaluation
metric as steps 8a/8b/8d.

### 8d. Cross-method geometry comparison

Compares the classical `procedure_alpha` mesh against the NeRF exported
point cloud on Chamfer Distance, F-score (at 1%/2%/5% of the bounding-box
diagonal), HD95, and Normal Consistency.

```bash
source abvt310/bin/activate   # open3d + sklearn, no torch needed
python evaluation_suite/geometry_comparison.py \
  --specimen M001 \
  --nerf-experiment M001_assumed_fixed_full
```
Expects the NeRF point cloud to already be exported (step 7's "Export &
test results" → point cloud). Saves
`evaluation_suite/reports/geometry_comparison_{ID}.txt`.

**Read the module docstring in `evaluation_suite/geometry_comparison.py`
before citing these numbers** — the short version:

- **This is method agreement, not ground-truth accuracy.** There is no
  independently-scanned reference geometry for these plants (no laser
  scan, no structured-light scan, no CAD model). If both methods share a
  bias, this comparison cannot see it. Report it as "classical/NeRF
  agreement," never as "reconstruction accuracy," in the dissertation.
- **No scale normalisation, ever.** Both point sets are supposed to
  already be metrically consistent — the NeRF poses come from
  `procedure_alpha`'s own registration math (step 7a), not an independent
  calibration. Rescaling to force agreement would hide exactly the
  metric-grounding error this comparison exists to catch. Only a rigid
  (rotation + translation) ICP refinement is applied, and its correction
  magnitude is printed as a sanity check — a large correction (the script
  warns above 10cm / 15°) means the two methods disagree about more than
  fine surface detail, and the numbers below it should be treated with
  suspicion.
- **Floater filtering is reported, not hidden.** NeRF exports commonly
  contain floater noise, worse with few views / short training. DBSCAN
  isolates the largest coherent cluster before comparing; the script
  prints what fraction was discarded and warns above 30%.

**Result for M001, and why no numeric CD/F-score/HD95 is reported as
citable here:** both the 3,000-iteration pilot and the full
30,000-iteration run (step 7) hit >90% floater discard, and the surviving
"largest cluster" sat nowhere near the classical mesh (ICP couldn't find
any correspondence within 15cm at 30k iterations — fitness 0.000). This
is **not** a training-budget problem: the tensorboard logs show train PSNR
reaching 28-30dB (near-perfect fit to the 12 training images) while eval
PSNR on the held-out view stayed flat at ~11.6-12.6dB for the entire
30,000-iteration run — classic sparse-view NeRF overfitting (memorising
each training view individually rather than learning consistent 3D
structure), not undertraining. More iterations won't fix it; the 12-view,
wide-baseline, hand-repositioned capture protocol is the bottleneck, and
step 7b's COLMAP note above predicts the same failure mode independently.

**Reproduced across independent runs (2026-07-27):** a second,
independently-trained 30,000-iteration checkpoint (different random
initialisation, run separately from the one above) gave the same result —
96.0% floater discard (vs. 94.2%), ICP fitness still 0.000 (no
correspondence within 15cm), F-score still 0.000 at every threshold. Two
separate training runs landing on the same failure mode rules out "one
unlucky run" as the explanation; this is a deterministic property of the
12-view capture protocol, not training-seed noise.

**What this means for the dissertation:** report the appearance metrics
(PSNR/SSIM/LPIPS, step 8c) and the train/eval PSNR divergence itself as
the finding — they're valid numbers that directly demonstrate *why* the
geometry comparison isn't meaningful for this rig, which is a more useful
and more defensible result than forcing a CD/F-score number onto geometry
that both ICP and DBSCAN independently flag as incoherent. Don't run
`geometry_comparison.py`'s numeric output through further interpretation
for M001 specifically; it's provided so the underlying evidence (floater
fraction, ICP fitness, cluster distances) is reproducible, not because the
distance numbers themselves are meaningful here. If a future capture
protocol adds meaningfully more views (denser angular sampling, narrower
baseline), re-run steps 7-8d — the script and pose derivation need no
changes, only better-conditioned input data.

---

**Notes**

- `--dual` only applies to the 6-step opposite-camera protocol
  (`camA`/`camB` sharing the same 6 loop angles). The older single-camera
  12-view protocol (e.g. `DG041_20260609_B02`) doesn't need it — just
  `--specimen ID` on its own.
- No `--legacy` here — that flag is for the old 4-view 90° `data_collection/`
  directory, unrelated to this rig.
- If `python run_pipeline.py alpha --specimen M001 --dual` errors with a
  missing depth file, re-run step 1 first — the specimen hasn't been
  imported from `dataset/plants/` yet.
- The NeRF arm (step 7) is entirely separate from steps 1-6 — it never
  touches `acquisition/dataset/specimens/` or `procedure_alpha/outputs/`,
  and lives in its own venv (`abvt_nerf/`, not `abvt310/`). Nothing about
  the classical pipeline changes if you skip it.
- **2026-07-26: dataset grew from 5 to 10 plants (M001-M010), and M001-M005
  are now different physical plants** — `dataset/plants/M001-M005` was
  overwritten with new captures reusing those IDs. The old specimens,
  `procedure_alpha/outputs/*_M001..M005*`, and the old M001 NeRF
  data/checkpoints were archived (not deleted) to
  `_archive_old_M001-M005_20260726/` at the repo root, then all 10 plants
  were freshly imported.
- **Resolved (2026-07-26): `dataset/ground_truth.csv` now has real weight
  rows for all 10 current Mango plants** (net weight, grams — the field
  scale's native unit; earlier Duranta-era code used kg, which is why
  `results_dashboard.py`/`comparison.py` both convert the registry's kg
  columns ×1000 for display). RF/ANN were retrained on this data via
  `biomass_engine/train_mango.py` (step 8a) — `RF_model_mango` /
  `ANN_model_mango` in `biomass_engine/trained/`. If you add more Mango
  plants later, re-run `import_farm_dataset.py` (step 1) then
  `train_mango.py` (step 8a) to pull in the new ground truth and retrain.
