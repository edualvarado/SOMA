"""
finalize_dataset.py -- scan the generated SKIM residual dataset and write metadata.json + README.md.

Run after batch_residuals.py finishes:
  /CT/SOMA/static00/miniforge3/envs/soma/bin/python finalize_dataset.py --out /CT/SOMA/static00/SKIM_dataset
"""
import os, glob, json, argparse
import numpy as np

# Anonymized capture codes (subject id + capture date) for the public dataset -- the
# original capture folder names (which contain subject first names) live only in the
# generation pipeline, not in the released metadata.
SUBJ_NAME = {"S1": "S1-21-11-24", "S2": "S2-12-12-25", "S3": "S3-17-12-25",
             "S4": "S4-22-12-25", "S5": "S5-19-12-25"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/CT/SOMA/static00/SKIM_dataset")
    args = ap.parse_args()

    subjects = {}
    for S in sorted(d for d in os.listdir(args.out) if os.path.isdir(f"{args.out}/{d}") and d.startswith("S")):
        shots = sorted(glob.glob(f"{args.out}/{S}/shot_*.npz"))
        frames = 0; resid = []; M = None; relabeled = os.path.exists(f"relabel_{S}.json")
        for p in shots:
            d = np.load(p, allow_pickle=True)
            frames += int(d["residual_m"].shape[0]); M = int(d["residual_m"].shape[1])
            m = d["mask"].astype(bool)
            if m.any():
                resid.append(float(np.linalg.norm(d["residual_m"][m], axis=1).mean()))
        subjects[S] = {"name": SUBJ_NAME.get(S, S), "shots": len(shots), "frames": frames,
                       "markers": M, "relabeled": relabeled,
                       "mean_residual_cm": round(float(np.mean(resid)) * 100, 2) if resid else None}

    meta = {
        "name": "SKIM marker residuals (pure-Python, no Blender)",
        "subjects": subjects,
        "units": "metres (residual_m); residual_scaled = residual_m * scale_factor",
        "scale_factor": 0.1,
        "coordinate_frame": "world, Z-up (triangulation frame)",
        "frame_alignment": "output frame i == BVH frame i+1 == triangulation frame i",
        "per_shot_npz_keys": {
            "residual_m": "(F,M,3) float16, observed - LBS in metres (0 where unobserved)",
            "residual_scaled": "(F,M,3) float16, residual_m * scale_factor",
            "mask": "(F,M) uint8, 1 where marker observed that frame",
            "pose6d": "(F,J,6) float32, per-joint 6D rotation (first two columns of the rotation matrix)",
            "root_position": "(F,3) float32, BVH root world position (metres)",
            "j_rest": "(J,3) float32, rest joint positions (re-aligned to canonical; metres)",
            "parents": "(J,) int32 joint parents (root=-1)",
            "frame_index": "(F,) int32, source BVH frame",
            "marker_ids": "(M,) marker id strings (canonical order)",
            "global_R": "(3,3) float32 per-shot rigid alignment LBS->observed",
            "global_t": "(3,) float32 per-shot translation",
            "scale_factor": "0.1",
        },
        "canonical_npz_keys": {
            "marker_ids": "(M,) canonical marker ids",
            "p_canon": "(M,3) canonical markers in T-pose (Z-up)",
            "p_bind": "(M,3) canonical markers rotated -90deg X into the BVH frame",
            "W": "(M,J) marker LBS weight matrix on the 24-joint skeleton",
            "joint_names": "(J,) BVH joint names",
        },
        "preprocessed_npz_keys": {
            "_path": "<S>/preprocessed_vFinal_clean/<shot>.npz  (training form; read via skim_loader.py)",
            "pose": "(F,J*6) float32, training 6D pose (root rotated to Z-up; column order [col0|col1])",
            "residuals": "(F,M,3) float16, scaled residual (residual_m * 0.1)",
            "masks": "(F,M) uint8, 1 where the marker was observed that frame",
        },
        "reconstruction": (
            "observed ~= global_R @ LBS(p_bind, pose) + global_t + residual_m, where LBS uses W, "
            "j_rest, parents, per-frame pose6d and root_position; rotate the result +90deg X back to Z-up."),
    }
    with open(f"{args.out}/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    tot_sh = sum(s["shots"] for s in subjects.values())
    tot_fr = sum(s["frames"] for s in subjects.values())
    rows = "\n".join(f"| {S} | {v['shots']} | {v['frames']} | {v['markers']} | {v['mean_residual_cm']} |"
                     for S, v in subjects.items())
    readme = f"""# SKIM marker residuals

Part of **SOMA: From Surface Observations to Muscle Anatomy** (ECCV 2026).
Project page: https://vcai.mpi-inf.mpg.de/projects/SOMA/ · Hosted at: https://gvv-assets.mpi-inf.mpg.de/soma

Per-subject marker residuals (soft-tissue deviation from Linear Blend Skinning), regenerated in
pure Python from BVH motion + canonical markers + triangulated mocap. No Blender.

| Subject | Shots | Frames | Markers | Mean residual (cm) |
|---|---|---|---|---|
{rows}

Total: {tot_sh} shots, {tot_fr} frames.

## Layout
- `<S>/canonical.npz` — per-subject canonical markers, bind markers, LBS weights, joint names.
- `<S>/shot_XXX.npz` — per-shot residual, visibility mask, pose (6D), and the per-shot alignment.
- `<S>/preprocessed_vFinal_clean/<shot>.npz` — packed training frames (`pose` (F,144),
  `residuals` (F,M,3), `masks` (F,M)); read with `skim_loader.py`.
- `<S>/layers/tpose`, `<S>/layers/apose` — canonical skin/muscle/skeleton meshes (.obj) per subject.
- `metadata.json` — full key descriptions, conventions, and the reconstruction recipe.
- `LICENSE` — MIT.

## Visualization
Two self-contained Viser viewers ship with the dataset: `visualize_residuals.py` (marker residual
cloud) and `visualize_skin_heatmap.py` (skin surface heatmap). See **VISUALIZE.md**;
`pip install -r requirements-viz.txt` then `python visualize_residuals.py`.

## Two representations of the residuals (you only need one)
The residuals are shipped in two forms for two different uses:
- `<S>/shot_XXX.npz` — the **research** form: full-precision `residual_m` (metres) + a scaled copy,
  the visibility `mask`, per-joint `pose6d`, `root_position`, `j_rest`, `parents`, and the per-shot
  `global_R`/`global_t`. Use this to reconstruct observed markers or work in world space (below).
- `<S>/preprocessed_vFinal_clean/<shot>.npz` — the **training** form used by SOMA: `pose` flattened
  to `(F, J*6)` in the network's 6D convention, the scaled `residuals`, and the `masks`. It is
  derivable from the research form and ships ready-to-train via `skim_loader.py`.

The two `pose` encodings differ — the research `pose6d` keeps the raw BVH joint rotations, while the
training `pose` rotates the root to Z-up and uses a different column order — so don't mix them.

## Training data / loader
`skim_loader.py` reads the packed per-shot training frames (numpy helper `load_shot`, plus a torch
`PackedMotionDataset` over all frames of a subject). Each frame is `(pose, residuals, masks)`;
residuals are the scaled GT (`residual_m * 0.1`).

## Conventions
- World frame: Z-up (the triangulation frame). Residuals are `observed - LBS` in **metres**;
  `residual_scaled = residual_m * 0.1` is a pre-scaled copy provided for convenience.
- Frame `i` in a shot corresponds to BVH frame `i+1`.
- `residual_m` is 0 and `mask` is 0 where a marker was not observed.

## Reconstruct observed markers
`observed ~= global_R @ LBS(p_bind, pose) + global_t + residual_m` (then rotate +90deg about X to Z-up).
See `metadata.json["reconstruction"]`.

## Citation
If you use SKIM, please cite:
```bibtex
@inproceedings{{2026:alvarado:soma,
  author    = {{Alvarado, Eduardo and Kim, Emily and Nolte, Gerrit and Runte, Friedemann and Botsch, Mario and Habermann, Marc and Theobalt, Christian}},
  title     = {{SOMA: From Surface Observations to Muscle Anatomy}},
  booktitle = {{European Conference on Computer Vision (ECCV)}},
  year      = {{2026}},
}}
```
"""
    with open(f"{args.out}/README.md", "w") as f:
        f.write(readme)
    print(f"wrote {args.out}/metadata.json and README.md")
    print(json.dumps(subjects, indent=2))


if __name__ == "__main__":
    main()
