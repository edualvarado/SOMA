# 05-Training

Fifth stage of the SOMA pipeline. Trains a physics-informed neural network that
predicts per-vertex residual displacements on top of Linear Blend Skinning (LBS)
for both the muscle and skin layers.

---

## Pipeline Overview

```
[04-Blender outputs, copied to static00/{S}/]
  preprocessed_vFinal_clean/   ← 00_preprocess_data.py converts raw JSON → .npy
        |
        ↓
  01_end_to_end_training.py    ← train the model (interactive or SLURM)
  01_end_to_end_training_job.sh
        |
        ↓ checkpoints/{run_name}/...epoch_{N}.pth
        |
        ↓
  02_validate_training.py      ← interactive Viser visualization + validation metrics
        |
        ↓
  03_evaluate_metrics.py       ← quantitative evaluation on validation set
        |
        ↓
  06-Evaluation/               ← SMPL alignment + intersection metrics
```

---

## Step-by-Step Instructions

### Step 0 — Pre-process raw data (`00_preprocess_data.py`)

Converts raw per-shot JSON files (BVH, residuals, masks, LBS markers) from
`static00/{S}/raw/` into per-frame `.npy` arrays ready for the DataLoader.

```bash
python 00_preprocess_data.py --subject S4
python 00_preprocess_data.py --subject S1 --base_dir /custom/path/S1
```

Inputs (from `static00/{S}/raw/shot_{N}_captury/`):

| File | Description |
|------|-------------|
| `{S}_shot_{N}.bvh` | Motion capture BVH (from 04-Blender) |
| `{S}_residuals_shot_{N}_world_lbs_scaled_tpose.json` | Scaled marker residuals (from 04-Blender) |
| `{S}_masked_residuals_shot_{N}_world_tpose.json` | Per-frame visibility masks (from 04-Blender) |
| `{S}_canonical_markers_lbs_shot_{N}_exported_tpose.json` | LBS-deformed canonical markers (from 04-Blender) |

Also requires (from `static00/{S}/canonical_model/`):
- `{S}_canonical_data_tpose.json` — T-pose marker positions
- `generated_marker_barycentric_map.json` — marker → mesh vertex barycentric mapping

Output: `static00/{S}/preprocessed_vFinal_clean/`

```
preprocessed_vFinal_clean/
  pose_rotations/    # shot_{N}_frame_{FFFF}.npy   — (J*6,) flattened 6D rotation vectors
  residuals/         # shot_{N}_frame_{FFFF}.npy   — (M, 3) marker residual vectors
  masks/             # shot_{N}_frame_{FFFF}.npy   — (M,) visibility masks
  canonical_lbs/     # shot_{N}_frame_{FFFF}.npy   — (M, 3) LBS-deformed canonical positions
```

> **Note on S1 BVH scale:** S1 BVH uses scale `1.0`; all other subjects use `0.001`.
> This is handled automatically inside the script.

---

### Step 1 — Train (`01_end_to_end_training.py`)

**Interactive (local):**
```bash
python 01_end_to_end_training.py
```

**Cluster (SLURM):**
```bash
sbatch 01_end_to_end_training_job.sh
```

SLURM config: `gpu20` partition, 1 GPU, 16 CPUs, 32 GB RAM, 2-day time limit.

#### Key Configuration (`CONFIG` dict, top of script)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `subject` | `"S1"` | Subject to train on |
| `architecture` | `"mlp"` | Model type: `"linear"`, `"mlp"`, or `"unet"` |
| `learning_rate` | `1e-4` | Adam learning rate |
| `epochs` | `20` | Training epochs |
| `batch_size` | `128` | Batch size |
| `weight_decay` | `0.0` | Adam weight decay |
| `train_split` | `0.9` | 90/10 train/val split |
| `preload_to_ram` | `False` | Load all `.npy` files to RAM for faster I/O |
| `overfit_single` | `False` | Debug: train on 1 frame only |
| `overfit_multiple` | `False` | Debug: train on first 300 frames only |
| `viz_enabled` | `False` | Enable live Viser visualization during training |

#### Loss Weight Presets (`LAMBDAS_PRESETS` dict, top of script)

Switch between ablation configurations by changing one line:

```python
ACTIVE_PRESET = "full"   # <-- change this
LAMBDAS = LAMBDAS_PRESETS[ACTIVE_PRESET]
```

| Preset | Description |
|--------|-------------|
| `"full"` | All losses enabled at high weights |
| `"no_smooth_tan"` | Ablation: remove E_smooth and E_tangential |
| `"no_physics"` | Ablation: remove E_biharmonic, E_spring, E_vol |
| `"no_vol"` | Ablation: remove volume preservation only |
| `"no_bi_stretch"` | Ablation: remove E_biharmonic and E_spring |

Each preset controls the following keys:

| Key | Purpose |
|-----|---------|
| `w_data` | Marker tracking via barycentric interpolation |
| `w_smooth_musc/skin` | Laplacian spatial smoothness |
| `w_biharmonic_musc/skin` | 2nd-order smoothness (wrinkle control) |
| `w_spring_musc/skin` | Stretch resistance |
| `w_tangent_musc/skin` | Tangential sliding prevention |
| `w_vol_musc/skin` | Volume preservation (Gauss quadrature) |

#### Architecture Options

- **`linear`** — Single matrix multiply `P × R`. Fast baseline (linear blendshapes).
- **`mlp`** — 2-layer FC with BatchNorm, LeakyReLU, Dropout. Default.
- **`unet`** — Encoder-decoder with skip connections. Most expressive.

Learnable parameters `P_muscle` and `P_skin` are `(V×3) × (9×(J-1))` blendshape
matrices (23,752 vertices, 24 joints → 207 rotation parameters).

#### Checkpoint Naming

Checkpoints are saved to `checkpoints/` with a name encoding the architecture and active preset:
```
ABLATION_{ARCH}_{PRESET}_epoch_{N}.pth
```
Example:
```
checkpoints/ABLATION_MLP_full_epoch_17.pth
```

The active preset and full lambda values are also logged to the console at startup.

---

### Step 2 — Validate (`02_validate_training.py`)

Interactive Viser visualizer. Loads a checkpoint and renders the predicted
muscle/skin deformation alongside the ground truth for a given shot.

Edit `CONFIG` (subject, shot) and `ACTIVE_CHECKPOINT` at the top of the script, then run:

```bash
python 02_validate_training.py
```

```python
# Switch checkpoints by changing one line:
ACTIVE_CHECKPOINT = "volume_skin_500"   # <-- key from CHECKPOINTS dict
CHECKPOINT_PATH = CHECKPOINTS[ACTIVE_CHECKPOINT]["path"]
```

Each entry in `CHECKPOINTS` has a `"path"` and a `"result"` field summarizing
the qualitative outcome when that checkpoint was evaluated.

Launches a Viser server at `http://localhost:8080` for interactive inspection.

---

### Step 3 — Evaluate metrics (`03_evaluate_metrics.py`)

Computes quantitative metrics (prism volume, marker tracking error) over the
validation set and launches an interactive Viser viewer.

```bash
python 03_evaluate_metrics.py \
    --subject S4 \
    --shot shot_001 \
    --checkpoint ./checkpoints/volume_both_clean/full_MLP_..._epoch_5.pth
```

The `--checkpoint` path can be any `.pth` file from the `checkpoints/` directory.
The `CHECKPOINTS` dict at the top of the script is a reference table of all
tested configurations, each with its path and a `"result"` note. Copy the
desired path from there and pass it via `--checkpoint`.

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
| `F_set_muscle_colors_new.py` | Sets muscle colors from `unified_muscle_ranges.json` on the active object |
| `H_animate_HIT.py` | Loads an OBJ frame sequence as animation with a configurable frame limit |

---

## Data Folder Structure

```
static00/{S}/
  raw/
    shot_{N}_captury/
      {S}_shot_{N}.bvh
      {S}_residuals_shot_{N}_world_lbs_scaled_tpose.json
      {S}_masked_residuals_shot_{N}_world_tpose.json
      {S}_canonical_markers_lbs_shot_{N}_exported_tpose.json

  preprocessed_vFinal_clean/          # Output of 00_preprocess_data.py
    pose_rotations/
    residuals/
    masks/
    canonical_lbs/

  canonical_model/
    {S}_canonical_data_tpose.json
    generated_marker_barycentric_map.json
    lbs_skin/
      {S}_skin_lbs_weights_exported.npy

  layers/
    tpose/
      musc_layer-{S}-TPose.obj        # Muscle mesh in T-pose
      skin_layer-{S}-TPose.obj        # Skin mesh in T-pose
      skel_layer-{S}-TPose.obj        # Skeleton mesh in T-pose

  validation/                         # Held-out evaluation sequences
```

```
05-Training/
  checkpoints/
    01_adding_smoothness/             # Development stage 1: E_data + E_smooth
    02_adding_stretch/                # Development stage 2: + E_spring
    03_adding_tan/                    # Development stage 3: + E_tangential
    04_bi_new/                        # Development stage 4: + E_biharmonic
    05_changing_tan/                  # Development stage 5: tangential tuning
    06_adding_volume/                 # Development stage 6: + E_vol (skin only)
    07_adding_volume_both/            # Development stage 7: + E_vol (muscle + skin)
    08_adding_volume_both_clean/      # Development stage 8: + visibility mask
    09_architectures/                 # Architecture ablations (Linear, MLP, UNet)
    10_priors/                        # Prior ablations (no_smooth_tan, no_physics, etc.)
  runs/                               # TensorBoard logs
  {S}_validation_filepaths.json       # Paths to held-out .npy frames for validation
```

---

## Archived Scripts (`99-Archived/`)

| File | Notes |
|------|-------|
| `old_01_end_to_end_training.py` | Previous training script; preserved as reference for the raw ablation history (multiple shadowing `LAMBDAS` dicts, verbose checkpoint names) |
| `old_end-to-end-training_vFinal_clean.py` | Older S1-only version; hardcoded to `preprocessed_vFinal/` (path no longer exists); uses abandoned `FullGenerativeModel` architecture |
| `old_validate_blendshapes_vFinal.py` | Older validation script; same broken data path and model mismatch |
| `old_validate_blendshapes_vFinal_clean.py` | Same as above with debug tools re-enabled |
| `old_diagnostic.py` | Ad-hoc diagnostic snippet |
| `old_E_set_muscle_colors.py` | Superseded by `F_set_muscle_colors_new.py` (had hardcoded S1/shot_006 path) |
| `old_G_animate_smpl.py` | Superseded by `H_animate_HIT.py` (H adds `MAX_FRAME` limit) |
