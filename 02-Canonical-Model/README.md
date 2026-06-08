# 02-Canonical-Model

Second stage of the SOMA pipeline. Produces a canonical 3D point cloud of the
marker positions on the body mesh, and converts the 3D tracking sequences into
the format required for registration.

The pipeline mixes Python scripts run on the cluster with one manual step that
runs inside **Blender** on a local machine.

---

## Pipeline Overview

```
detect_2D_uv.py                   (Step 1)  auto-detect markers on UV map
        ↓
  [manual fix of 4 wrong IDs in markers-skin.json → markers-skin-fixed.json]
        ↓
manual_marker_annotator.py        (Step 2)  manually annotate missing 4-pt markers
        ↓
merge_marker_annotations.py       (Step 3)  merge fixed + manual → markers-skin-final.json
        ↓
manual_marker_annotator_1point.py (Step 4)  annotate 1-pt edge markers
        ↓
fix_manual_annotations.py         (Step 5)  reformat 1-pt annotation output
        ↓
merge_marker_annotations.py       (Step 6)  merge final + 1-pt → markers-skin-final-corrected.json
        ↓
  [Blender: 04-Blender/scripts/02_canonical_model/build_canonical_model.py]
        ↓                         (Step 7 — local, manual)
convert_output.py                 (Step 8)  convert Blender + triangulation outputs
```

`draw-uv-markers.py` can be used at any stage to generate debug visualizations
of the current annotation state.

---

## Step-by-Step Instructions

### Step 1 — Auto-detect markers on the UV map (`detect_2D_uv.py`)

Runs ChArUco detection on the flat UV texture image of the suit skin.

```bash
python detect_2D_uv.py \
    --folder S1/ \
    --board  configs/suits/charuco-suit.json \
    --debug
```

- Input:  `S{N}/skin.jpg`
- Output: `S{N}/uv_detections_charuco-suit/markers-skin.json`
- Debug:  `S{N}/debug/skin_charuco-suit.jpg`

> **Known issue:** The auto-detection is incomplete and produces 4 incorrect IDs
> (925→256, 731→141, 995→940, 262→598). These must be corrected manually in the
> JSON before proceeding. Save the corrected file as `markers-skin-fixed.json`.

### Step 2 — Manually annotate missing 4-point markers (`manual_marker_annotator.py`)

Interactive OpenCV tool. Click the 4 corners of each missed marker in order;
press `s` to save and `q` to quit.

```bash
python manual_marker_annotator.py
```

- Input:  `S{N}/debug/skin_charuco-suit.jpg`
- Output: `S{N}/uv_detections_charuco-suit/markers-skin-manual.json`

### Step 3 — Merge automatic (fixed) + manual annotations (`merge_marker_annotations.py`)

```bash
python merge_marker_annotations.py \
    --json1  S1/uv_detections_charuco-suit/markers-skin-fixed.json \
    --json2  S1/uv_detections_charuco-suit/markers-skin-manual.json \
    --output S1/uv_detections_charuco-suit/markers-skin-final.json
```

### Step 4 — Manually annotate 1-point edge markers (`manual_marker_annotator_1point.py`)

Same interactive tool as Step 2 but each marker is annotated with a single click
(for markers on the suit edges that only have 1 visible corner).

```bash
python manual_marker_annotator_1point.py
```

- Input:  `S{N}/debug/skin_charuco-suit_final.jpg`
- Output: `S{N}/uv_detections_charuco-suit/markers-skin-manual-1-point.json`

### Step 5 — Fix 1-point annotation format (`fix_manual_annotations.py`)

The 1-point annotator saves each corner as a separate entry. This script groups
them back into the standard 4-corners-per-marker format.

```bash
python fix_manual_annotations.py \
    --input  S1/uv_detections_charuco-suit/markers-skin-manual-1-point.json \
    --output S1/uv_detections_charuco-suit/markers-skin-manual-1-point-corrected.json
```

- Input:  `S{N}/uv_detections_charuco-suit/markers-skin-manual-1-point.json`
- Output: `S{N}/uv_detections_charuco-suit/markers-skin-manual-1-point-corrected.json`

### Step 6 — Merge final + 1-point annotations (`merge_marker_annotations.py`)

```bash
python merge_marker_annotations.py \
    --json1  S1/uv_detections_charuco-suit/markers-skin-final.json \
    --json2  S1/uv_detections_charuco-suit/markers-skin-manual-1-point-corrected.json \
    --output S1/uv_detections_charuco-suit/markers-skin-final-corrected.json
```

`markers-skin-final-corrected.json` is the **final annotation file** used in
the next step.

### Step 7 — Build canonical model in Blender (manual, local)

Open Blender locally and run:

```
04-Blender/scripts/02_canonical_model/build_canonical_model.py
```

This script reads `markers-skin-final-corrected.json`, projects the UV
annotations onto the 3D mesh, and writes two output files into
`S{N}/uv_detections_charuco-suit/source/canonical_model/`:

| File | Description |
|------|-------------|
| `output.json` | 3D positions of all canonical marker corners on the mesh |
| `missed.json` | Markers that could not be placed on the mesh |

Copy `shot_XXX/triangulation_markers_processed.json` from the cluster into
`S{N}/uv_detections_charuco-suit/source/shot_XXX/` before Step 8.

### Step 8 — Convert outputs to registration format (`convert_output.py`)

Converts both the Blender canonical model and the triangulation sequence into
the unified key format (`marker_{id}_{instance}_{corner}`) required by the
registration stage.

```bash
# Convert canonical model + one shot (most common):
python convert_output.py --subject S1 --shot shot_001

# First time for a new subject (canonical model only):
python convert_output.py --subject S1

# Additional shots for an already-processed subject:
python convert_output.py --subject S1 --shot shot_002 --skip_canonical
```

Outputs written to `S{N}/uv_detections_charuco-suit/registration/`:

| File | Description |
|------|-------------|
| `canonical_model/canonical_data.json` | Static canonical 3D point cloud |
| `shot_XXX/S{N}_triangulated_sequence_shot_XXX.json` | Per-frame 3D tracking |

Copy these two files back to the cluster for use in `03-Registration`.

---

## Data Folder Structure (per subject)

```
S{N}/
  skin.jpg                                      # UV texture image of the suit
  debug/                                        # Visualization images at each stage
    skin_charuco-suit.jpg                       # Step 1 auto-detection
    skin_charuco-suit_fixed.jpg                 # After manual ID correction
    skin_charuco-suit_manual.jpg                # Step 2 manual annotations
    skin_charuco-suit_final.jpg                 # After Step 3 merge
    skin_charuco-suit_manual_1_point.jpg        # Step 4 edge annotations
    skin_charuco-suit_final_corrected.jpg       # Final annotation state
  uv_detections_charuco-suit/
    markers-skin.json                           # Step 1 output (raw auto-detection)
    markers-skin-fixed.json                     # Step 1 with 4 IDs corrected manually
    markers-skin-manual.json                    # Step 2 output
    markers-skin-final.json                     # Step 3 output
    markers-skin-manual-1-point.json            # Step 4 output
    markers-skin-manual-1-point-corrected.json  # Step 5 output
    markers-skin-final-corrected.json           # FINAL annotation file (→ Blender)
    source/
      canonical_model/
        output.json                             # Blender output (3D marker positions)
        missed.json                             # Markers Blender could not place
      shot_XXX/
        triangulation_markers_processed.json    # Copied from 01-Suit-Processing
    registration/
      canonical_model/
        canonical_data.json                     # Step 8 output (→ 03-Registration)
      shot_XXX/
        S{N}_triangulated_sequence_shot_XXX.json  # Step 8 output (→ 03-Registration)
  weights/canonical_model/lbs_skin/             # LBS weights for the skin mesh
```

---

## Utility Scripts

- **`draw-uv-markers.py`** — Draws marker annotations on the UV image for
  visual inspection. Edit the file to point to the JSON and image you want
  to visualize, then run `python draw-uv-markers.py`. Produces an overlay
  with auto-detected markers in red and manual markers in green.

---

## Configuration Files

| File | Description |
|------|-------------|
| `configs/suits/charuco-suit.json` | ChArUco suit layout (used by detect_2D_uv.py) |
| `configs/boards/charuco.json` | Generic ChArUco board config |
| `configs/boards/aruco.json` | ArUco-only board config |
| `configs/boards/color.json` | Color-based board config |
| `configs/boards/quest.json` | Quest headset board config |
| `configs/intrinsics/` | Camera intrinsics for stereo ego/exo cameras |

---

## Archived Scripts (`99-Archived/`)

| File | Notes |
|------|-------|
| `old_check_duplicated_ids.py` | Ad-hoc debug snippet to find duplicate marker IDs — no `main()`, not runnable |
| `old_side_by_side_visualization.py` | Hardcoded to a path (`final_learning_data/`) that no longer exists |
