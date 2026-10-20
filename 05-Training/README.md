# 05-Training

Fifth stage of the SOMA pipeline. Trains a physics-informed neural network that
predicts per-vertex residual displacements on top of Linear Blend Skinning (LBS)
for both the muscle and skin layers.

The released models (`RELEASE_V2`) are described in [Released model](#released-model-release_v2).

---

## Pipeline Overview

```
[SKIM dataset v2 at $SKIM_ROOT (or --data-root), per subject <root>/{S}/]
  preprocessed_vFinal_clean/   ← packed per-shot .npz (training-ready; no preprocessing needed)
  canonical_model/{S}_rig.npz  ← SOMA rig (rest joints + skin weights)
        |
        ↓
  01_end_to_end_training.py    ← train one model per subject (interactive or SLURM)
  sweep.py                     ← train (01) + evaluate (03) + log a row in sweep_results.csv
        |
        ↓ checkpoints/{S}_ABLATION_{ARCH}_{tag}_best_data{,_rigid}.pth + {S}_static_canon_{tag}.npy
        |
        ↓
  02_validate_training.py      ← interactive Viser viewer (prediction vs. markers, per shot)
  03_evaluate_metrics.py       ← metrics on the held-out shots
        |
        ↓
  06-Evaluation/               ← SMPL alignment + intersection metrics
```

---

## Files

| File | Purpose |
|------|---------|
| `01_end_to_end_training.py` | Training. Defaults = the released configuration |
| `01_end_to_end_training_job.sh` | SLURM wrapper for 01 |
| `02_validate_training.py` | Viser viewer; checkpoint table `CHECKPOINTS` at the top |
| `03_evaluate_metrics.py` | Held-out metrics (marker error, intersections, volume, roughness) |
| `sweep.py` | Named runs (`RUNS`): 01 + 03 per subject, results to `sweep_results.csv` |
| `forearm_detwist.py` | Removes the spurious Captury ForeArm twist (S2–S5); used by 01/02/03 and the v2 build |
| `muscle_binding.py` | Binds the individual muscle meshes to the muscle layer (Gaussian blend of the nearest layer vertices) and transfers its displacement; used by 02 (replaces the dataset's `individual_muscle_to_skin_binding.json`) |
| `build_rig.py` | Builds the SOMA rig: rest joints refitted to the bones + bone-heat skin weights (`rigs/{S}_rig.npz`) |
| `build_skim_v2.py`, `verify_skim_v2.py` | Build SKIM v2 from v1 (rig, de-twist, root, alignment and layer fixes baked in) and verify it |
| `preprocess_from_skim.py`, `pack_preprocessed.py` | Rebuild the packed training tensors from the residual npz (reproducibility only) |
| `00_preprocess_data.py` | Original Blender-JSON preprocessor (historical reference) |
| `{S}_validation_filepaths.json` | Train/validation split per subject (whole held-out shots) |
| `A_`–`H_*.py` | Blender visualization scripts (see below) |

---

## Step-by-Step Instructions

### Step 0 — Get the data (no preprocessing needed)

The released SKIM dataset is already **training-ready**: each subject ships packed per-shot frames
under `<S>/preprocessed_vFinal_clean/<shot>.npz` with arrays `pose` `(F, J*6)`,
`residuals` `(F, M, 3)`, `masks` `(F, M)` and `global_R` `(3, 3)`. Download the dataset and point
the scripts at it:

```bash
export SKIM_ROOT=/path/to/SKIM_dataset   # or pass --data-root /path/to/SKIM_dataset to each script
```

`01`, `02`, `03`, `preprocess_from_skim.py` and `pack_preprocessed.py` all read `--data-root`, falling
back to `$SKIM_ROOT`, and exit with an error if neither is set. 01/02/03 read `dataset_version` from
`<root>/metadata.json`: on v2 the rig, de-twist and layer fixes are already in the data; on v1 they are
applied at load time (see [SKIM v2 vs v1](#skim-v2-vs-v1)).

<details>
<summary>Rebuilding the training tensors (reproducibility only)</summary>

The packed frames are derived from the SKIM residual `.npz` (`<S>/shot_XXX.npz`):
`preprocess_from_skim.py` reconstructs the BVH pose in the training 6D convention and pairs it with
the scaled residuals + masks, then `pack_preprocessed.py` bundles each shot into one `.npz`.
SKIM v2 itself is built from v1 with `build_skim_v2.py` and checked with `verify_skim_v2.py`
(see [SKIM v2 vs v1](#skim-v2-vs-v1)).

```bash
python preprocess_from_skim.py --subjects S1 S2 S3 S4 S5   # residual npz -> per-frame .npy
python pack_preprocessed.py    --subjects S1 S2 S3 S4 S5   # pack -> per-shot .npz
```

> **Note on S1 BVH scale:** S1 BVH uses scale `1.0`; all other subjects use `0.001`
> (handled automatically inside the scripts).
</details>

### Pretrained models

The released checkpoints (`SOMA_checkpoints.tar`, distributed with the dataset) extract into
`checkpoints/`:

```bash
tar -xf SOMA_checkpoints.tar -C 05-Training
```

### Released model (`RELEASE_V2`)

One UNet per subject (same architecture and configuration for all subjects), trained on SKIM v2 for
40 epochs with seed 0. It is the `RELEASE_V2` run in `sweep.py`, i.e. the defaults of 01 plus `--seed 0`:

| Setting | Value |
|---|---|
| Loss preset | `soma_final`: w_data 5e4 · smooth 100/200 · biharmonic 100/200 · spring 1e4/1e4 · tangent 1/0.1 · vol 30/30 · crease 75/75 (musc/skin) · w_unobs 1e4 |
| Optimiser | lr 1e-4, batch 128, weight decay 0 |
| Data loss | posed (`--posed-data`), static per-marker offset removed in each marker's bone frame (`--residual-static-removal`), `--rigid-correction` |
| Rig | `--rig auto` → `<root>/<S>/canonical_model/<S>_rig.npz` (joints refitted to the bones, bone-heat skin weights; `build_rig.py`) |
| Volume / crease | `--vol-loss rel`, `--fat-posed`, `--crease-mode onesided`, `--rigid-bone` |
| Unobserved skin | `--unobs-normal`: hidden skin is pulled to LBS along the normal only |
| Marker filters (training only) | `--drop-isolated`, `--drop-inconsistent 0.05` |

Results on the held-out shots (mm unless noted; ↓ better). MPME-obs is the marker error over observed
markers; P90 is over all markers, where unobserved markers score 0 for LBS by construction:

| Subject | MPME-obs SOMA / LBS | P90 SOMA / LBS | fat inter (%) | musc inter (%) | skin roughness SOMA / LBS (°) |
|---|---|---|---|---|---|
| S1 | 12.32 / 28.58 | 35.62 / 44.59 | 0.932 | 0.101 | 1.62 / 1.10 |
| S2 |  9.40 / 23.62 | 27.97 / 34.75 | 0.031 | 0.013 | 2.03 / 1.06 |
| S3 |  9.72 / 27.70 | 30.14 / 40.91 | 0.044 | 0.063 | 1.86 / 0.99 |
| S4 | 11.97 / 26.77 | 46.36 / 34.57 | 0.507 | 0.072 | 1.51 / 0.85 |
| S5 | 10.85 / 20.97 | 41.14 / 29.22 | 0.027 | 0.043 | 1.53 / 0.90 |

The released model differs from the paper model (preset `paper_final`, MLP).

#### SKIM v2 vs v1

SKIM v2 is derived from v1 with these fixes baked into the data. 01/02/03 read `dataset_version` from
`metadata.json` and, on v1, apply the equivalent fixes at load time.

- **Rig:** rest joints refitted to the bone contacts and bone-heat skin weights (no zero-weight hand
  vertices); marker weights are the barycentric blend of the skin weights.
- **Residuals:** recomputed against the rebuilt rig, `residual = observed − (global_R · LBS + global_t)`,
  with `global_R/global_t` refitted per shot and stored for every subject.
- **Poses:** the spurious Captury forearm twist is removed (S2–S5; raw poses kept as `pose6d_raw` /
  `pose_raw`), and S3 shot_005/008, exported with a rotated rest skeleton, are re-expressed on the rig.
- **Root:** `root_position` follows the rig Hips (the raw BVH root tracks a different hip point; raw kept
  as `root_position_raw`).
- **Geometry:** S2–S5 barycentric marker maps fixed; S4 inner layers moved onto the skin layer.

---

### Step 1 — Train (`01_end_to_end_training.py`)

The defaults of 01 are the released configuration (preset `soma_final`, UNet, rig from the dataset,
posed data loss, static removal, rigid correction, marker filters; see [Released model](#released-model-release_v2)).

**Released models (all subjects, 40 epochs, seed 0):**
```bash
python sweep.py --only RELEASE_V2 --subjects S1,S2,S3,S4,S5 --epochs 40
```
This writes `checkpoints/{S}_ABLATION_UNET_RELEASE_V2_e40_*`, the same names as the pretrained models
from `SOMA_checkpoints.tar`, which it replaces; to keep both, train with another `--tag` (below).

**One subject, interactive:**
```bash
python 01_end_to_end_training.py --subject S1 --epochs 40 --tag my_run
```

**Cluster (SLURM):**
```bash
sbatch 01_end_to_end_training_job.sh --subject S1 --epochs 40   # args go to 01; SKIM_ROOT is inherited from the submitting shell
```

SLURM config: `gpu20` partition, 1 GPU, 16 CPUs, 32 GB RAM, 2-day time limit.

`python 01_end_to_end_training.py --help` lists every flag. Main ones:

| Flag | Default | Description |
|------|---------|-------------|
| `--subject` | `S1` | Subject to train on |
| `--arch` | `unet` | `linear`, `mlp` or `unet` |
| `--epochs` | `20` | Training epochs |
| `--preset` | `soma_final` | Loss-weight preset from `LAMBDAS_PRESETS` |
| `--tag` | preset name | Run name used in checkpoint / log names |
| `--seed` | `0` | Random seed |
| `--rig` | `auto` | `auto` = `<root>/<S>/canonical_model/<S>_rig.npz` on v2, else `rigs/<S>_rig.npz`; `none` = original rig |

On Windows keep `num_workers = 0` (the dataset is held in RAM and cannot be pickled to worker processes).

#### Loss Weight Presets (`LAMBDAS_PRESETS`)

| Preset | Description |
|--------|-------------|
| `"soma_final"` | Released configuration (default) |
| `"paper_final"` | Configuration of the submission |
| `"full"` | All losses enabled at high weights |
| `"data_only"` | Data term only |
| `"no_smooth_tan"`, `"no_physics"`, `"no_vol"`, `"no_bi_stretch"` | Ablations |

| Key | Purpose |
|-----|---------|
| `w_data` | Marker tracking via barycentric interpolation |
| `w_smooth_musc/skin` | Laplacian spatial smoothness |
| `w_biharmonic_musc/skin` | 2nd-order smoothness (wrinkle control) |
| `w_spring_musc/skin` | Stretch resistance |
| `w_tangent_musc/skin` | Tangential sliding prevention |
| `w_vol_musc/skin` | Volume preservation (Gauss quadrature) |
| `w_crease_musc/skin` | Crease (fold) penalty |
| `w_unobs` | Keeps unobserved skin close to LBS (along the normal) |

#### Architecture Options

- **`linear`** — Single matrix multiply `P × R`. Fast baseline (linear blendshapes).
- **`mlp`** — 2-layer FC with BatchNorm, LeakyReLU, Dropout.
- **`unet`** — Encoder-decoder with skip connections. Released models.

Learnable parameters `P_muscle` and `P_skin` are `(V×3) × (9×(J-1))` blendshape
matrices (23,752 vertices, 24 joints → 207 rotation parameters).

#### Outputs

```
checkpoints/{S}_ABLATION_{ARCH}_{tag}_best_data.pth        # weights at the best validation data loss
checkpoints/{S}_ABLATION_{ARCH}_{tag}_best_data_rigid.pth  # rigid correction head saved with it
checkpoints/{S}_static_canon_{tag}.npy                     # static marker offset removed from the targets
runs/{S}_ABLATION_{ARCH}_{tag}/                            # TensorBoard logs
```

---

### Step 2 — Validate (`02_validate_training.py`)

Interactive Viser viewer. Loads a checkpoint and renders the predicted muscle/skin deformation alongside
the markers for one shot. Edit `CONFIG` (subject, shot) and `ACTIVE_CHECKPOINT` at the top of the script:

```python
CONFIG = {"subject": "S1", "shot": "shot_003", "architecture": "unet", ...}
ACTIVE_CHECKPOINT = "release_v2_e40"   # key from the CHECKPOINTS dict
```

```bash
python 02_validate_training.py         # Viser server at http://localhost:8080
```

Held-out shots: S1 `shot_003/006/015`, S2–S3 `shot_006/012`, S4–S5 `shot_009/010`.

---

### Step 3 — Evaluate metrics (`03_evaluate_metrics.py`)

Metrics over the held-out shots (marker error, muscle/fat intersections, volume variation, roughness):

```bash
python 03_evaluate_metrics.py --subject S1 --arch unet \
    --checkpoint checkpoints/S1_ABLATION_UNET_RELEASE_V2_e40_best_data.pth
```

The rig, the static canon, the rigid head and the forearm de-twist are detected automatically.
Use **MPME-obs** (observed markers) and τ to compare marker error: the "all markers" MPME/P90 count
unobserved markers, where LBS scores 0 by construction.

---

## Visualization Scripts (Blender)

All scripts below run inside Blender's Python environment. Open Blender, paste
or load the script in the Scripting workspace, and run.

| Script | Purpose |
|--------|---------|
| `A_color_muscles.py` | Assigns integer IDs as `ID_Color` vertex attribute to selected muscle meshes |
| `B_heatmap.py` | Generates a deformation heatmap by comparing two mesh objects vertex-by-vertex |
| `C_animation_meshes.py` | Animates a mesh by loading OBJ frames from a directory (vertex cache) |
| `D_animation_heatmap.py` | Per-frame heatmap overlay during animation; can isolate individual muscles |
| `F_set_muscle_colors.py` | Sets muscle colors from `unified_muscle_ranges.json` on the active object |
| `H_animate_HIT.py` | Loads an OBJ frame sequence as animation with a configurable frame limit |

---

## Data Folder Structure

```
$SKIM_ROOT/{S}/
  canonical.npz                       # canonical markers, marker LBS weights, rig joints
  shot_{N}.npz                        # research form: residuals, mask, pose, alignment
  preprocessed_vFinal_clean/
    shot_{N}.npz                      # training form: pose (F, J*6), residuals (F, M, 3), masks (F, M), global_R
  canonical_model/
    {S}_rig.npz                       # SOMA rig (v2)
    generated_marker_barycentric_map.json
    lbs_skin/{S}_skin_lbs_weights_exported.npy
    ...
  layers/{tpose,apose}/               # skin / muscle / skeleton meshes, per-muscle OBJs
  raw/shot_{N}_captury/{S}_shot_{N}.bvh
```

See the dataset's own `README.md` / `metadata.json` for every array.

```
05-Training/
  checkpoints/                        # {S}_ABLATION_UNET_RELEASE_V2_e40_* from SOMA_checkpoints.tar (not tracked)
  runs/, sweep_logs/                  # TensorBoard and sweep logs (not tracked)
  sweep_results.csv                   # one row per sweep run (not tracked)
  {S}_validation_filepaths.json       # held-out frame ids (whole shots)
```
