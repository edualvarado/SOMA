# 04-Blender

Fourth stage of the SOMA pipeline. Contains all Blender Python scripts that
run inside Blender locally. This folder is the bridge between the raw capture
data and the training inputs consumed by `05-Training`.

All scripts under `scripts/` are run inside Blender's built-in Python environment
(they import `bpy`), except for `15_scale_residuals.py` and
`build_canonical_model.py` which are standalone Python scripts.

---

## Role in the Pipeline

```
02-Canonical-Model ──→ build_canonical_model.py
                            ↓ output.json (3D canonical markers)

03-Registration  ←── 07_export_lbs_weights_skin.py
                            ↓ {S}_skin_lbs_weights_exported.json

03-Registration  ←── A_export_new_canonical_data.py
                            ↓ {S}_canonical_data_tpose.json

                     [per-shot pipeline]
01_export_GT_markers.py
        ↓ {S}_triangulated_sequence_{shot}_transformed.json
04_export_LBS_markers.py
        ↓ {S}_canonical_markers_lbs_{shot}_exported_tpose.json
13_estimate_residuals_world.py
        ↓ {S}_residuals_{shot}_world_lbs_tpose.json
15_scale_residuals.py
        ↓ {S}_residuals_{shot}_world_lbs_scaled_tpose.json

                     [precomputed once per subject]
map_markers_to_barycentric_coords.py  → mappings/marker_barycentric_map.json
map_markers_to_muscles.py             → mappings/marker_to_muscle_map.json
precompute_muscle_laplacians.py       → reconstruction/muscle_laplacians.npz
estimate_layer_interpolation_weights.py → reconstruction/*_layer_interpolation_weights.json

All outputs consumed by 05-Training.
```

---

## Blender Scene Setup

Before running per-shot scripts, the Blender scene for a given shot must be
set up manually following these steps:

### A) Import BVH Studio Capture

1. Create a collection `StudioBVH_{shot}` inside a parent collection `Shot_{shot}`.
2. Import the BVH file with scale `0.001`.
3. Set Viewport Display → **Stick**, enable **In Front**.
4. Rename armature to `unknown_{shot}` (and bones inside accordingly).
5. In **Edit Mode**, select `LeftFoot`, `LeftToeBase`, `RightFoot`, `RightToeBase`.
6. **Shift + S → Cursor to Selected**, then set Origin to 3D Cursor.
7. Copy the armature offset; set Location to `0, 0, 0`.
8. In **Graph Editor**, correct hip offsets:
   - `hips Z`: apply positive Y offset (`+0.11079`)
   - `hips Y`: apply negative X offset (`-0.86654`)

9. Run `01_export_GT_markers.py` — creates collection `Observed_GT_Markers_{shot}`.

### B) InsideHumans Model Setup

1. Copy/Paste collection `InsideHumans_000` into `Shot_{shot}`.
2. Run `04_tools/define_bone_constraints.py` — links the InsideHumans armature
   to the BVH armature (`unknown_{shot}`).
3. Run `01_markers/03_visualize_LBS_markers.py` — visualize LBS canonical markers.

### C) LBS Verification

1. Run `01_markers/04_export_LBS_markers.py` — exports LBS marker positions.
2. Run `01_markers/05_visualize_exported_LBS_markers.py` — verify point cloud
   matches the live `LBS_Canonical_Markers_{shot}` collection.

### D) Residuals

1. Run `03_residuals/13_estimate_residuals_world.py` — compute world-space
   residuals (LBS-predicted vs. observed marker positions).
2. Run `03_residuals/15_scale_residuals.py` — scale residuals by `0.01` for
   unit consistency with training.
3. *(Optional)* Run `03_residuals/14_visualize_residuals_simple.py` for visual
   inspection.

### E) Laplacian (optional — direct Laplacian optimization path)

1. Run `laplacian/estimate_dense_deformation.py`
2. Run `laplacian/estimate_dense_muscle_constrained_deformation.py`
3. Run `laplacian/estimate_refined_two_pass_deformation.py`
4. Run `laplacian/visualize_dense_deformation.py`

### F) Reconstruction Visualization

1. Use the `canonical_layers_000` collection in the scene.
   `reconstruction/estimate_layer_interpolation_weights.py` is run once per
   subject to precompute skin/muscle interpolation weights.
2. Run `reconstruction/visualize_layer_dense_deformation.py` to inspect layer
   deformations.

---

## Script Reference

### `scripts/01_markers/` — Marker Export and Visualization

| Script | Status | Purpose | Output |
|--------|--------|---------|--------|
| `01_export_GT_markers.py` | **Active** | Transforms and streams triangulated sequences from the cluster JSON into world-space Blender coordinates | `data/registration/{S}/{shot}/{S}_triangulated_sequence_{shot}_transformed.json` |
| `02_visualize_GT_markers.py` | Utility | Visualizes observed GT markers as a point cloud in Blender | — |
| `03_visualize_LBS_markers.py` | **Active** | Visualizes LBS canonical markers driven by the armature (used in step B and for 03-Registration verification) | — |
| `03_visualize_LBS_markers_new_skeleton.py` | Utility | Same as above but for the new skeleton variant | — |
| `03_visualize_LBS_markers_new_skeleton_tpose.py` | Utility | T-pose variant of the above | — |
| `04_export_LBS_markers.py` | **Active** | Exports frame-by-frame LBS-deformed marker positions (streaming, multi-subject) | `data/registration/{S}/{shot}/{S}_canonical_markers_lbs_{shot}_exported_tpose.json` |
| `05_visualize_exported_LBS_markers.py` | Utility | Visualizes the exported LBS markers for verification | — |
| `06_visualize_differences_markers_filter.py` | Utility | Overlays GT vs. LBS marker positions and highlights differences | — |
| `A_export_new_canonical_data.py` | **Active** | Exports T-pose canonical marker positions (frame 0) preserving original A-pose IDs | `03-Registration/registration/{S}/canonical_model/{S}_canonical_data_tpose.json` |
| `B_plot_new_canonical_data.py` | Utility | Plots the exported canonical data for visual inspection | — |
| `C_Test.py` | Development | Ad-hoc test script | — |
| `visualize_markers.py` | Utility | Visualizes 3D marker positions from a canonical JSON using spheres | — |

### `scripts/02_canonical_model/` — Canonical Model and LBS Weights

| Script | Status | Purpose | Output |
|--------|--------|---------|--------|
| `build_canonical_model.py` | **Active** | Projects UV 2D marker detections onto the 3D mesh (barycentric interpolation) to produce the canonical 3D point cloud | `02-Canonical-Model/{S}/source/canonical_model/output.json` + `missed.json` |

#### `lbs_skin/` — Skin LBS Weight Export

| Script | Status | Purpose | Output |
|--------|--------|---------|--------|
| `07_export_lbs_weights_skin.py` | **Active** | Exports per-vertex LBS weights from the rigged skin mesh, excluding finger bones | `02-Canonical-Model/{S}/weights/canonical_model/lbs_skin/{S}_skin_lbs_weights_exported.json` → copied to `03-Registration` |
| `07_export_lbs_weights_musc.py` | **Active** | Same for the muscle mesh | `data/weights/canonical_model/lbs_musc/` |
| `C_verify_lbs_weights_skin.py` | Utility | Sanity-checks exported skin weights | — |
| `D_verify_lbs_weights_skin_deformation.py` | Utility | Verifies that skin deformation under LBS looks correct | — |
| `E_visualize_lbs_viser.py` | Utility | Visualizes LBS weights using the Viser library | — |

#### `lbs_markers/` — Marker Barycentric Map

| Script | Status | Purpose | Output |
|--------|--------|---------|--------|
| `map_markers_to_barycentric_coords.py` | **Active** | Precomputes barycentric coordinates for each marker relative to the mesh (used by 05-Training to map marker residuals to mesh vertices) | `data/mappings/marker_barycentric_map.json` |
| `08_export_lbs_weights_markers.py` | Utility | Exports LBS weights for individual marker points | `data/weights/canonical_model/lbs_markers/` |
| `09_verify_lbs_weights_markers.py` | Utility | Verifies marker LBS weights | — |

#### `muscles/` — Muscle Laplacians

| Script | Status | Purpose | Output |
|--------|--------|---------|--------|
| `precompute_muscle_laplacians.py` | **Active** | Precomputes the mesh Laplacian matrix for the muscle layer (used by the smoothness loss in 05-Training) | `data/reconstruction/muscle_laplacians.npz` |
| `separate_muscles.py` | Utility | Separates the unified muscle mesh into individual muscle objects | — |

### `scripts/03_residuals/` — Residual Computation

| Script | Status | Purpose | Output |
|--------|--------|---------|--------|
| `13_estimate_residuals_world.py` | **Active** | Computes world-space residuals per frame (observed − LBS-predicted), streaming | `data/registration/{S}/{shot}/{S}_residuals_{shot}_world_lbs_tpose.json` |
| `15_scale_residuals.py` | **Active** | Scales residuals by 0.01 for unit consistency; standalone Python (no Blender required) | `data/registration/{S}/{shot}/{S}_residuals_{shot}_world_lbs_scaled_tpose.json` |
| `10_estimate_residuals_mask.py` | Utility | Estimates residuals with per-frame visibility masking | `{S}_masked_residuals_{shot}_world_tpose.json` |
| `11_estimate_residuals_separated_world.py` | Utility | Residuals for observed-only markers (partial visibility) | — |
| `12_visualize_residuals_separated_simple.py` | Utility | Visualizes separated residuals in Blender | — |
| `14_visualize_residuals_simple.py` | Utility | Visualizes world-space residuals in Blender | — |
| `check_residuals.py` | Debug | Ad-hoc residual sanity check | — |
| `estimate_residuals_separated.py` | Superseded | Early non-streaming version | — |
| `visualize_residuals.py` | Superseded | Early residual visualization | — |
| `visualize_residuals_separated.py` | Superseded | Early separated residual visualization | — |

### `scripts/04_tools/` — Scene Setup and Utilities

| Script | Purpose |
|--------|---------|
| `define_bone_constraints.py` | Sets up bone constraints linking the InsideHumans armature to the BVH armature — **run once per shot** |
| `check_bone_constraints.py` | Verifies that bone constraints are correctly configured |
| `color_muscles.py` | Assigns per-muscle colors to the mesh for visualization |
| `count_verts.py` | Counts vertices per mesh layer |
| `create_deformation_heatmap.py` | Generates a deformation heatmap texture |
| `create_muscle_ids.py` | Assigns integer IDs to muscle regions |
| `cross-section.py` | Creates a cross-section view of the mesh |
| `debug_connectivity.py` | Debugs vertex/edge connectivity issues |
| `define_individual_muscle_skin_connections.py` | Sets up per-muscle skin attachment points |
| `fix_bone.py` | Corrects bone orientations or rest poses |
| `generate_muscle_colors.py` | Generates a color palette for muscle visualization |
| `regenerate_bary_map.py` | Re-runs barycentric map generation with updated mesh |
| `separate_muscles.py` | Duplicates the `muscles/separate_muscles.py` utility |
| `transfer_baked_mesh.py` | Transfers a baked mesh deformation to another object |
| `transfer_bary_map.py` | Transfers a barycentric map between mesh versions |
| `visualize_alignment.py` | Checks alignment between skeleton and mesh |

### `scripts/laplacian/` — Direct Laplacian Optimization (optional)

An alternative deformation path using Laplacian interpolation from sparse
marker residuals to dense mesh deformation. Not used in the primary neural
network training pipeline but useful for analysis.

| Script | Purpose |
|--------|---------|
| `map_markers_to_muscles.py` | Maps each marker to its nearest muscle region → `data/mappings/marker_to_muscle_map.json` |
| `estimate_dense_deformation.py` | Propagates sparse marker residuals to all mesh vertices via Laplacian |
| `estimate_dense_muscle_constrained_deformation.py` | Same but with muscle boundary constraints |
| `estimate_refined_two_pass_deformation.py` | Two-pass refinement for higher accuracy |
| `visualize_dense_deformation.py` | Visualizes the three Laplacian deformation variants |

### `scripts/reconstruction/` — Layer Interpolation Weights

| Script | Status | Purpose | Output |
|--------|--------|---------|--------|
| `estimate_layer_interpolation_weights.py` | **Active** | Precomputes per-vertex interpolation weights from markers to skin/muscle layers (KD-tree, inverse-distance) | `data/reconstruction/muscle_layer_interpolation_weights.json` + `skin_layer_interpolation_weights.json` |
| `visualize_layer_dense_deformation.py` | Utility | Visualizes the interpolated layer deformation in Blender | — |
| `test/` | Development | Test scripts for muscle interpolation | — |

### `scripts/05_results/` — Final Results Visualization

| Script | Purpose |
|--------|---------|
| `display_animation.py` | Plays back a predicted deformation animation in Blender |
| `display_animation_muscles_joined.py` | Same with muscle and skin layers joined |
| `compare_two_obj.py` | Overlays two OBJ meshes for comparison |
| `new_colors_islands.py` | Assigns new colors to mesh islands |
| `transfer_meshes.py` | Transfers meshes between Blender scenes |

### `scripts/analysis/` — Deformation Analysis

| Script | Purpose |
|--------|---------|
| `analyze_baked_mesh.py` | Analyzes baked mesh deformation statistics |
| `avg_displacement_markers.py` | Computes average marker displacement across frames |
| `bake_deformation_to_sequence.py` | Bakes animated deformation into per-frame OBJ sequence |

### Other

| Script | Purpose |
|--------|---------|
| `scripts/import_all_bvh.py` | Batch imports all BVH files into the scene |
| `scripts/uv/plot_3D_uv.py` | Plots 3D UV mapping for inspection |
| `scripts/uv/plot_uv_editor.py` | Visualizes UV layout in the UV editor |
| `scripts/test/` | Experimental deformation scripts (not part of main pipeline) |

---

## Data Folder Structure

```
data/
  registration/
    canonical_model/                                  # S1 canonical data (older structure, no subject prefix in path)
      canonical_data.json                             # A-pose canonical markers (from build_canonical_model.py)
      canonical_data_tpose.json                       # T-pose canonical markers (from A_export_new_canonical_data.py)
      output.json                                     # Raw Blender output (same as canonical_data.json, pre-convert)
      missed.json                                     # Markers that could not be placed on the mesh

    S{N}/                                             # Per-subject folder (S2–S5)
      canonical_model/
        {S}_canonical_data.json                       # A-pose canonical markers
        {S}_canonical_data_tpose.json                 # T-pose canonical markers

      {shot}/                                         # Per-shot data
        {S}_triangulated_sequence_{shot}_transformed.json
                                                      # Output of 01_export_GT_markers.py
                                                      # → 05-Training input (observed marker positions)
        {S}_canonical_markers_lbs_{shot}_exported_tpose.json
                                                      # Output of 04_export_LBS_markers.py
                                                      # → used by 13_estimate_residuals_world.py
        {S}_residuals_{shot}_world_lbs_tpose.json     # Output of 13_estimate_residuals_world.py
        {S}_residuals_{shot}_world_lbs_scaled_tpose.json
                                                      # Output of 15_scale_residuals.py
                                                      # → 05-Training input (marker residuals)
        {S}_masked_residuals_{shot}_world_tpose.json  # Output of 10_estimate_residuals_mask.py (optional)
        {S}.bvh                                       # BVH motion file for this subject/shot

  weights/
    canonical_model/
      lbs_skin/
        skin_lbs_weights_exported.json                # Generic (non-subject-specific) copy of skin LBS weights.
                                                      # Per-subject outputs are written to
                                                      # 02-Canonical-Model/{S}/weights/canonical_model/lbs_skin/
                                                      # by 07_export_lbs_weights_skin.py → then copied to 03-Registration
      lbs_musc/
        musc_meshes_lbs_weights_exported_*.json       # Muscle mesh LBS weights
      lbs_markers/
        markers_lbs_weights_exported*.json            # Marker point LBS weights

  mappings/
    marker_barycentric_map.json                       # Output of map_markers_to_barycentric_coords.py
                                                      # → 05-Training input (marker → mesh vertex mapping)
    marker_to_muscle_map.json                         # Output of map_markers_to_muscles.py
                                                      # → used by Laplacian scripts

  reconstruction/
    muscle_laplacians.npz                             # Output of precompute_muscle_laplacians.py
                                                      # → 05-Training (smoothness loss)
    muscle_layer_interpolation_weights.json           # Output of estimate_layer_interpolation_weights.py
    skin_layer_interpolation_weights.json             # → 05-Training (dense deformation interpolation)

  analysis/                                           # Analysis outputs (plots, cached metrics)
  layers/                                             # Mesh layer OBJ exports
  scenes/                                             # Blender scene files
```
