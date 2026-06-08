# 01-Suit-Processing

First stage of the SOMA pipeline. Processes multi-camera video of a subject wearing a ChArUco marker suit to produce a dense, clean 3D point cloud of marker positions over time.

---

## Pipeline Overview

The full pipeline runs 5 steps in sequence. The entry point is `suit-processing-job_final.sh` (SLURM batch job).

```
detect_2D_final.py
        ↓
interpolation_2D_final.py
        ↓
tracking_3D_v2_final.py
        ↓
interpolation_3D_final.py
        ↓
post_process_triangulation_v3_final.py
```

### Step 1 — 2D Detection (`detect_2D_final.py`)

Detects ChArUco marker corners in each camera stream independently. Writes results to:

```
<shot>/detections_charuco-suit/
```

Key arguments: `--folder`, `--board`, `--setup`, `--modulus`, `--parallel`, `--max_frames`

### Step 2 — 2D Interpolation (`interpolation_2D_final.py`)

Fills short gaps in the per-camera 2D detections using linear interpolation. Output goes to:

```
<shot>/detections_charuco-suit/2D-interpolated-N{frames_2D_int}/
```

Key argument: `--frames_2D_int` (max gap length to interpolate, default 10)

### Step 3 — 3D Triangulation (`tracking_3D_v2_final.py`)

Triangulates each marker across all camera pairs using the calibrated projection matrices. Per-frame results are aggregated and optionally refined via `least_squares` reprojection minimization. Outliers are rejected via median + z-score filtering before refinement. Output:

```
<shot>/tracking_charuco-suit/triangulation/triangulation_markers.json
```

Key arguments: `--folder`, `--frames_2D_int`, `--max_frames`

### Step 4 — 3D Interpolation (`interpolation_3D_final.py`)

Fills short gaps in the 3D trajectories. Output:

```
<shot>/tracking_charuco-suit/triangulation/3D-interpolated-N{frames_3D_int}/triangulation_markers.json
```

Key argument: `--frames_3D_int` (default 10)

### Step 5 — Post-processing (`post_process_triangulation_v3_final.py`)

4-stage cleaning pipeline applied to the interpolated 3D tracks:

| Stage | Description |
|-------|-------------|
| 1 | Remove globally unstable markers (longest continuous track < `window_size_global`) |
| 2 | Temporal median filter to smooth per-marker trajectories |
| 3 | Motion consensus filter using KDTree — removes markers whose motion disagrees with neighbours |
| 4 | Remove isolated track islands shorter than `window_size_local` |

Output: `triangulation_markers_processed.json` (same directory as input)

Key arguments: `--folder`, `--frames_3D_int`, `--window_size`

---

## Running the Pipeline

```bash
sbatch suit-processing-job_final.sh \
  --folder /CT/SOMA/static00/data/<Subject-DD-MM-YY>/ \
  --board /CT/SOMA/work/01-Suit-Processing/configs/suits/charuco-suit.json
```

Before submitting, edit `suit-processing-job_final.sh` and set `SHOTS_TO_PROCESS` to the correct subject array:

```bash
# Options: SHOTS_VALENTIN, SHOTS_TIMO, SHOTS_MONA, SHOTS_SARAH
SHOTS_TO_PROCESS=("${SHOTS_SARAH[@]}")
```

The script expects `--folder` to point to the **parent directory** containing `shot_001/`, `shot_002/`, etc.

---

## Input / Output Structure

```
<subject-folder>/
  shot_001/
    cameras.calib                            # camera calibration file
    stream*.mp4                              # raw video streams
    detections_charuco-suit/                 # Step 1 output
      2D-interpolated-N10/                   # Step 2 output
    tracking_charuco-suit/
      triangulation/
        triangulation_markers.json           # Step 3 output
        3D-interpolated-N10/
          triangulation_markers.json         # Step 4 output
          triangulation_markers_processed.json  # Step 5 output
```

---

## Configuration Files

| File | Description |
|------|-------------|
| `configs/suits/charuco-suit.json` | ChArUco suit layout (active) |
| `configs/boards/charuco.json` | Generic ChArUco board config |
| `configs/boards/aruco.json` | ArUco-only board config |
| `configs/boards/color.json` | Color-based board config |
| `configs/boards/quest.json` | Quest headset board config |

---

## Utility Scripts

- `visualize_3D.py` — Static 3D visualization of triangulated markers
- `visualize_3D_video.py` — Renders the 3D marker trajectories as a video

---

## Toolkit (`toolkit/`)

Shared library used by all pipeline scripts:

| Module | Description |
|--------|-------------|
| `loading` | Load calibrations, detections, resolutions |
| `geometry` | 3D geometry utilities |
| `filtering` | Signal filtering helpers |
| `sync` | Multi-camera synchronization |
| `visual` | Visualization helpers |
| `bvh` | BVH motion capture utilities |
| `qrcode` | QR/ArUco code utilities |

---

## Archived Scripts (`99-Archived/`)

Outdated versions kept for reference only. Do not use.

| File | Notes |
|------|-------|
| `old_tracking_3D.py` | Original sequential (single-core) triangulation |
| `old_post_process_triangulation.py` | v1 post-processing — 2 stages only |
| `old_post_process_triangulation_v2.py` | v2 post-processing — 4 stages but contains a bug (undefined variable `frame`) |
| `old_suit-processing.sh` | Single-shot processing script, superseded by the SLURM job |
| `old_sub-process.sh` | Loop wrapper for `old_suit-processing.sh`, references undefined `$BASE_FOLDER` |
