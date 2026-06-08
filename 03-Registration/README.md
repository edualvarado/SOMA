# 03-Registration

Third stage of the SOMA pipeline. Computes Linear Blend Skinning (LBS) weights
for every marker corner point on the suit, by transferring weights from the
nearest vertices of the character mesh via barycentric interpolation.

The pipeline mixes one **Blender step** (run locally) with two Python scripts
run on the cluster.

---

## Pipeline Overview

```
[02-Canonical-Model] S{N}_canonical_data.json
        |
        | (copy to local)
        ↓
  [Blender] 04-Blender/scripts/02_canonical_model/lbs_skin/07_export_lbs_weights_skin.py
        |          Exports per-vertex skin LBS weights from the rigged mesh
        |          → {subject}_skin_lbs_weights_exported.json
        |
        | (copy to cluster)
        ↓
create_marker_LBS_weights_final.py       (Step 1 — cluster)
        |   Transfers mesh vertex weights to marker corners via
        |   barycentric interpolation
        |   → {subject}_marker_lbs_weights_exported.json
        ↓
verify_marker_LBS_weights.py             (Step 2 — cluster, optional)
        |   Sanity-checks the output (normalization, no negatives, no empties)
        ↓
  [Blender] 04-Blender/scripts/01_markers/03_visualize_LBS_markers.py
                   Visualizes the marker weights on the mesh (optional)
```

---

## Step-by-Step Instructions

### Step 1 — Export skin weights from Blender (local, manual)

Open Blender and run `04-Blender/scripts/02_canonical_model/lbs_skin/07_export_lbs_weights_skin.py`.

This script exports per-vertex LBS weights from the rigged skin mesh into:
```
registration/{subject}/canonical_model/{subject}_skin_lbs_weights_exported.json
```

Copy this file to the cluster before running Step 2.

Also ensure the following are in place in `registration/{subject}/canonical_model/`:
- `{subject}_canonical_data.json` — copied from `02-Canonical-Model` output
- `skin_layer-{subject}-APose.obj` — the skin mesh exported from Blender in A-pose

### Step 2 — Compute marker LBS weights (`create_marker_LBS_weights_final.py`)

Transfers LBS weights from the mesh vertices to each marker corner using
barycentric interpolation on the closest mesh triangle.

```bash
python create_marker_LBS_weights_final.py --subject S1
```

Inputs (all under `registration/{subject}/canonical_model/` by default):

| File | Description |
|------|-------------|
| `{subject}_canonical_data.json` | 3D marker corner positions (from 02-Canonical-Model) |
| `skin_layer-{subject}-APose.obj` | Suit skin mesh in A-pose |
| `{subject}_skin_lbs_weights_exported.json` | Per-vertex weights exported from Blender |

Output:
```
registration/{subject}/canonical_model/{subject}_marker_lbs_weights_exported.json
```

> **Note on coordinate alignment:** The .obj mesh is in Y-up (Blender default),
> while the JSON data is in Z-up. The script automatically applies a +90° X-axis
> rotation to the mesh before processing. The exported JSON coordinates are
> already in Z-up space and are used as-is.

### Step 3 — Verify weights (`verify_marker_LBS_weights.py`) — optional

Checks that all marker weights are normalized (sum ≈ 1.0), non-negative, and
that every marker has at least one bone influence.

```bash
python verify_marker_LBS_weights.py \
    --input registration/S1/canonical_model/S1_marker_lbs_weights_exported.json
```

### Step 4 — Visualize in Blender (local, optional)

Run `04-Blender/scripts/01_markers/03_visualize_LBS_markers.py` locally in
Blender to inspect the marker weights on the mesh.

---

## Data Folder Structure (per subject)

```
registration/
  S{N}/
    canonical_model/
      {subject}_canonical_data.json               # Copied from 02-Canonical-Model registration output.
                                                  # Marker 3D positions in A-pose, produced by
                                                  # 04-Blender: build_canonical_model.py

      {subject}_canonical_data_tpose.json         # Marker 3D positions in T-pose world coordinates.
                                                  # Exported from Blender via
                                                  # 04-Blender: 01_markers/A_export_new_canonical_data.py

      skin_layer-{subject}-APose.obj              # Skin mesh in A-pose.
                                                  # Exported directly from Blender.

      {subject}_skin_lbs_weights_exported.json    # Per-vertex LBS weights for the skin mesh.
                                                  # Exported from Blender via
                                                  # 04-Blender: 02_canonical_model/lbs_skin/07_export_lbs_weights_skin.py

      {subject}_marker_lbs_weights_exported.json  # Per-marker LBS weights (bone_indices, weights, bone_names).
                                                  # Computed by create_marker_LBS_weights_final.py (this folder).
```

---

## Archived Scripts (`99-Archived/`)

| File | Notes |
|------|-------|
| `old_create_marker_LBS_weights.py` | Original version — no `main()`, hardcoded paths, leftover debug blocks and unused rotation matrices |
