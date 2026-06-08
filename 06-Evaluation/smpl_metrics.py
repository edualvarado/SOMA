import os
import json
import numpy as np
import trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION & CALIBRATION
# ==========================================
# Use the exact offsets you found in the alignment step
HEIGHT_OFFSET = -0.25 
DEPTH_OFFSET = -0.02
SUBJECT = "S1"

# Paths
HIT_BASE_DIR = Path(rf"/CT/SOMA/static00/outputs_hit_smpl/")
RESIDUALS_DIR = Path(rf"/CT/SOMA/static00/S1/preprocessed_vFinal_clean/residuals/")
GT_BIND_MARKERS_PATH = Path(rf"/CT/SOMA/static00/{SUBJECT}/canonical_model/{SUBJECT}_canonical_data_tpose.json")
BARY_MAP_PATH = Path("soma_to_smpl_barycentric_map.json")
VALIDATION_SET_PATH = Path(rf"/CT/SOMA/work/05-Training/{SUBJECT}_validation_filepaths.json")

def _rot_x(points, deg=0.0):
    R = Rotation.from_euler('x', deg, degrees=True).as_matrix()
    return (points @ R.T).astype(np.float32)

def run_evaluation():
    print("--- Starting HIT Baseline Evaluation ---")
    
    # 1. Load the Barycentric Map (2306 markers)
    with open(BARY_MAP_PATH, 'r') as f:
        bary_map = json.load(f)
    
    # These keys are the 2306 valid markers in sorted order
    eval_marker_ids = sorted(bary_map.keys())
    bary_verts = np.array([bary_map[mid]['vertex_indices'] for mid in eval_marker_ids])
    bary_weights = np.array([bary_map[mid]['bary_coords'][0] for mid in eval_marker_ids])

    # 2. Load the exact Validation Set used in 03_evaluate_metrics.py
    if not VALIDATION_SET_PATH.exists():
        print(f"❌ ERROR: Validation set file not found at {VALIDATION_SET_PATH}")
        return
        
    with open(VALIDATION_SET_PATH, 'r') as f:
        val_pose_paths = json.load(f)
    
    # Filter for shot_001 only (to match your current residuals directory)
    val_files = [Path(p).name for p in val_pose_paths if "shot_001" in p]
    print(f"Loaded {len(val_files)} unseen validation frames for Shot 001.")

    # 3. Establish Centroid Calibration (from Frame 0)
    with open(GT_BIND_MARKERS_PATH, 'r') as f:
        full_json = json.load(f)
    canonical_data = full_json.get("0", full_json)
    
    # Get the 2306 p_bind markers corresponding to the map
    p_bind = _rot_x(np.array([canonical_data[mid][0] for mid in eval_marker_ids], dtype=np.float32), deg=-90.0)
    
    # T-pose mesh to establish shift
    hit_tpose = trimesh.load(HIT_BASE_DIR.parent / "frame_000000_tpose.obj", process=False)
    soma_centroid = p_bind.mean(axis=0)
    hit_centroid = hit_tpose.vertices.mean(axis=0)
    
    # 4. Evaluation Loop
    all_errors = []
    dre_15, dre_20, dre_25 = [], [], []

    # Get p_gt_start for DRE (Frame 0 must be calculated)
    # Reconstruct Frame 0 for DRE baseline
    res0_path = RESIDUALS_DIR / "shot_001_frame_0000.npy"
    res0 = np.load(res0_path) # Already 2306
    p_gt_start = (p_bind + res0) - soma_centroid + hit_centroid
    p_gt_start[:, 1] += HEIGHT_OFFSET
    p_gt_start[:, 2] += DEPTH_OFFSET

    for filename in tqdm(val_files, desc="Evaluating Sequence"):
        # A. Load GT Residuals (Already 2306, matching eval_marker_ids order)
        res_path = RESIDUALS_DIR / filename
        if not res_path.exists(): continue
        
        residuals = np.load(res_path) # Shape: (2306, 3)
        p_gt = p_bind + residuals
        
        # B. Load HIT Mesh
        # Convert 'shot_001_frame_XXXX.npy' -> 'frame_00XXXX'
        frame_num = int(filename.split('_')[-1].split('.')[0])
        mesh_path = HIT_BASE_DIR / f"frame_{frame_num:06d}" / "hit_male_best" / "smpl_mesh.obj"
        if not mesh_path.exists(): continue
        hit_mesh = trimesh.load(mesh_path, process=False)
        
        # C. Predict Markers on SMPL Skin
        v = hit_mesh.vertices
        p_pred = (bary_weights[:, 0:1] * v[bary_verts[:, 0]] +
                  bary_weights[:, 1:2] * v[bary_verts[:, 1]] +
                  bary_weights[:, 2:3] * v[bary_verts[:, 2]])
        
        # D. Apply Your Calibrated Manual Alignment
        p_gt_aligned = p_gt - soma_centroid + hit_centroid
        p_gt_aligned[:, 1] += HEIGHT_OFFSET
        p_gt_aligned[:, 2] += DEPTH_OFFSET
        
        # E. Calculate Error (in mm)
        dist = np.linalg.norm(p_pred - p_gt_aligned, axis=1) * 1000.0 
        all_errors.extend(dist)
        
        # F. DRE Calculation (Tau movement relative to Frame 0)
        movement = np.linalg.norm(p_gt_aligned - p_gt_start, axis=1) * 1000.0
        dre_15.extend(dist[movement > 25.0])
        dre_20.extend(dist[movement > 20.0])
        dre_25.extend(dist[movement > 15.0])

    # 5. Print Results
    print("\n" + "="*45)
    print(f" HIT BASELINE METRICS ({len(all_errors)//2306} validation frames)")
    print("="*45)
    print(f"MPME   : {np.mean(all_errors):.2f} mm")
    print(f"MedPME : {np.median(all_errors):.2f} mm")
    print(f"P90    : {np.percentile(all_errors, 90):.2f} mm")
    print("-" * 45)
    print(f"DRE > 15mm : {np.mean(dre_15):.2f} mm")
    print(f"DRE > 20mm : {np.mean(dre_20):.2f} mm")
    print(f"DRE > 25mm : {np.mean(dre_25):.2f} mm")
    print("="*45)

if __name__ == "__main__":
    run_evaluation()