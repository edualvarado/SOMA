"""
00_preprocess_data.py

Pre-processes raw motion capture data for SOMA training.

This script performs the following:
1.  Loads a canonical marker set (Bind Pose).
2.  Iterates through all shot folders in the raw data directory.
3.  For each shot:
    a.  Loads BVH motion data and converts it to a continuous 6D rotation representation.
    b.  Loads Ground Truth (GT) residuals and visibility masks.
    c.  Loads Linear Blend Skinning (LBS) based canonical marker positions (actually not used during final training).
    d.  Validates frame counts between motion, residuals, and LBS data.
    e.  Saves processed data (Pose, Residuals, Masks, LBS) as per-frame .npy files.

Output Structure:
    processed_dir/
    ├── pose_rotations/   # (J*6,) flattened 6D rotation vectors
    ├── residuals/        # (M, 3) marker residual vectors
    ├── masks/            # (M,) boolean visibility masks
    └── canonical_lbs/    # (M, 3) LBS-deformed canonical markers

Usage:
    python 00_preprocess_data.py --subject S4
    python 00_preprocess_data.py --subject S1 --base_dir /custom/path/S1
"""

import json
import argparse
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from pymotion.io.bvh import BVH
import pymotion.rotations.quat as quat
import pymotion.rotations.ortho6d as sixd

# Global rotation correction applied to canonical markers to align with BVH coordinate system
DEGREE_BVH_X = -90.0

def _rot_x(points, deg=0.0):
    """Rotates points around the X-axis by deg degrees."""
    R = Rotation.from_euler('x', deg, degrees=True).as_matrix()
    return points @ R.T

def get_rest_joint_locations_zero_offset(bvh_obj, scale=1.0):
    """
    Calculates global rest pose joint positions from a BVH object, 
    setting the root joint's offset to zero.
    """
    _, _, parents, offsets, _, _ = bvh_obj.get_data()
    parents[0] = -1

    offsets = offsets.copy()
    if offsets.shape[0] > 0:
        offsets[0] = np.zeros(3, dtype=offsets.dtype)

    offsets *= scale
    j_rest = np.zeros_like(offsets)
    
    # Forward Kinematics for Rest Pose
    for i in range(len(parents)):
        if parents[i] == -1:
            j_rest[i] = offsets[i]
        else:
            j_rest[i] = j_rest[parents[i]] + offsets[i]

    return j_rest, parents, offsets

def load_canonical_model(canonical_path, barycentric_path):
    """
    Loads the canonical marker definition and barycentric mapping.
    Returns the bind pose positions and a mapping from marker ID to index.
    """
    logger.info(f"Loading canonical model from {canonical_path}...")
    
    with open(canonical_path, 'r') as f:
        canonical_data = json.load(f).get("0", {})

    with open(barycentric_path, 'r') as f:
        barycentric_map = json.load(f)

    # Filter/Sort markers based on the barycentric map keys (the subset we care about)
    bary_marker_ids = sorted(barycentric_map.keys())
    marker_id_to_index = {mid: i for i, mid in enumerate(bary_marker_ids)}
    num_markers = len(bary_marker_ids)

    logger.info(f"Found {num_markers} canonical markers in barycentric map.")

    # Extract positions
    p_bind = np.zeros((num_markers, 3), dtype=np.float32)
    for i, mid in enumerate(bary_marker_ids):
        # Position is usually [[x,y,z]] in json, take index 0
        p_bind[i] = np.array(canonical_data[mid][0])

    # Apply global rotation fix to align with BVH
    p_bind = _rot_x(p_bind, deg=DEGREE_BVH_X)
    
    return p_bind, marker_id_to_index

def process_shot(shot_path, processed_dir, p_bind, marker_id_to_index, subject):
    """
    Processes a single shot folder: loads BVH, residuals, masks, LBS data, validates, and saves.
    Returns True if successful, False if skipped/failed.
    """
    shot_name = shot_path.name
    shot_number = shot_name.split('_')[1]

    # File definitions
    files = {
        "bvh": shot_path / f'{subject}_shot_{shot_number}.bvh',
        "residuals": shot_path / f'{subject}_residuals_shot_{shot_number}_world_lbs_scaled_tpose.json',
        "mask": shot_path / f'{subject}_masked_residuals_shot_{shot_number}_world_tpose.json',
        "canonical_lbs": shot_path / f'{subject}_canonical_markers_lbs_shot_{shot_number}_exported_tpose.json'
    }

    # Validation: Check if all required files exist
    if not all(p.exists() for p in files.values()):
        missing = [k for k, v in files.items() if not v.exists()]
        logger.warning(f"Skipping {shot_name}. Missing files: {missing}")
        return False

    try:
        # --- 1. Load & Process BVH (Motion) ---
        bvh = BVH()
        bvh.load(str(files["bvh"]))
        
        scale = 1.0 if subject == "S1" else 0.001
        
        # Extract Raw Data
        local_rotations, local_positions, _, _, _, _ = bvh.get_data()
        
        # A. Center Root Position to (0,0,0)
        local_positions[:, 0, :] = np.zeros_like(local_positions[:, 0, :])
        
        # B. Fix Root Rotation (Z-up vs Y-up conversion)
        angles = np.array([0, np.pi/2, 0])[..., np.newaxis]
        axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        q_fix = quat.from_angle_axis(angles, axes)
        rotmats_fix = quat.to_matrix(q_fix)
        R_total = rotmats_fix[2] @ rotmats_fix[1] @ rotmats_fix[0] # Z @ Y @ X

        # Get current root rotations (scalar-first w,x,y,z in pymotion)
        q_root = local_rotations[:, 0, :] 
        
        # Convert to scipy (scalar-last x,y,z,w)
        q_root_scipy = np.roll(q_root, -1, axis=1) 
        rotmats_root = Rotation.from_quat(q_root_scipy).as_matrix()
        
        # Apply rotation fix
        R_new = R_total @ rotmats_root
        q_new_scipy = Rotation.from_matrix(R_new).as_quat()
        q_new_pymotion = np.roll(q_new_scipy, 1, axis=1) # Back to (w,x,y,z)
        local_rotations[:, 0, :] = q_new_pymotion
        
        # C. Scale Position
        local_positions[:, 0, :] *= scale
        
        # D. Convert to 6D Continuous Representation
        # local_rotations: (F, J, 4) -> continuous_6d: (F, J, 3, 2)
        continuous_6d = sixd.from_quat(local_rotations) 
        
        # --- 2. Load Residuals & Masks ---
        with open(files["residuals"], 'r') as f:
            res_data = json.load(f)
        with open(files["mask"], 'r') as f:
            mask_data = json.load(f)
            
        frames_sorted = sorted(res_data.keys(), key=int)
        num_target_frames = len(frames_sorted)
        num_markers = p_bind.shape[0]

        # Optimize: Pre-allocate arrays
        res_arr = np.zeros((num_target_frames, num_markers, 3), dtype=np.float32)
        mask_arr = np.zeros((num_target_frames, num_markers), dtype=np.uint8)

        # Fill arrays
        for i, frame_key in enumerate(frames_sorted):
            # Residuals
            frame_res = res_data.get(frame_key, {})
            for mid, val in frame_res.items():
                if mid in marker_id_to_index:
                    v = val[0] if (isinstance(val, list) and len(val)==1 and isinstance(val[0], list)) else val
                    res_arr[i, marker_id_to_index[mid]] = v
            
            # Masks
            frame_mask = mask_data.get(frame_key, {})
            for mid, val in frame_mask.items():
                if mid in marker_id_to_index:
                    v = val[0] if (isinstance(val, list) and val) else val
                    mask_arr[i, marker_id_to_index[mid]] = v

        # --- 3. Load Canonical LBS ---
        with open(files["canonical_lbs"], 'r') as f:
            lbs_data = json.load(f)
            
        lbs_arr = np.zeros((len(lbs_data), num_markers, 3), dtype=np.float32)
        for i, frame_key in enumerate(frames_sorted):
             if frame_key in lbs_data:
                 frame_lbs = lbs_data[frame_key]
                 for mid, val in frame_lbs.items():
                     if mid in marker_id_to_index:
                         v = val[0] if (isinstance(val, list) and len(val)==1 and isinstance(val[0], list)) else val
                         lbs_arr[i, marker_id_to_index[mid]] = v

        # --- 4. Frame Count Sync & Padding ---
        n_motion = continuous_6d.shape[0]
        n_gt = res_arr.shape[0]
        
        if n_motion == n_gt - 1:
            logger.info(f"Padding Motion ({n_motion} -> {n_gt}) for {shot_name}")
            continuous_6d = np.concatenate([continuous_6d[:1], continuous_6d], axis=0)
        
        if lbs_arr.shape[0] == n_gt - 1:
            logger.info(f"Padding LBS ({lbs_arr.shape[0]} -> {n_gt}) for {shot_name}")
            lbs_arr = np.concatenate([lbs_arr[:1], lbs_arr], axis=0)
            
        if not (continuous_6d.shape[0] == res_arr.shape[0] == mask_arr.shape[0] == lbs_arr.shape[0]):
            logger.error(f"Frame mismatch in {shot_name}: Mot={continuous_6d.shape[0]}, Res={res_arr.shape[0]}, Mask={mask_arr.shape[0]}, LBS={lbs_arr.shape[0]}")
            return False

        # --- 5. Save Data (FIXED VECTOR LAYOUT & TYPES) ---
        
        # Explicitly concatenate columns to match old script: [col1, col2]
        rot_ordered = np.concatenate([continuous_6d[..., 0], continuous_6d[..., 1]], axis=-1)
        
        # Flatten and Cast to Float32 (Size Fix)
        rot_flat = rot_ordered.reshape(rot_ordered.shape[0], -1).astype(np.float32)

        save_dirs = {
            "pose": processed_dir / "pose_rotations",
            "residual": processed_dir / "residuals",
            "mask": processed_dir / "masks",
            "lbs": processed_dir / "canonical_lbs"
        }
        
        for i in range(continuous_6d.shape[0]):
            fname = f"shot_{shot_number}_frame_{i:04d}.npy"
            np.save(save_dirs["pose"] / fname, rot_flat[i])
            np.save(save_dirs["residual"] / fname, res_arr[i].astype(np.float32))
            np.save(save_dirs["mask"] / fname, mask_arr[i].astype(np.uint8))
            np.save(save_dirs["lbs"] / fname, lbs_arr[i].astype(np.float32))
            
        return True

    except Exception as e:
        logger.exception(f"Error processing {shot_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Pre-process raw motion capture data for SOMA training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--subject", type=str, required=True,
        help="Subject identifier (e.g. S4). Used to build default input/output paths."
    )
    parser.add_argument(
        "--base_dir", type=Path, default=None,
        help="Override base directory. Defaults to '/CT/SOMA/static00/{subject}'."
    )
    args = parser.parse_args()

    subject  = args.subject
    base_dir = args.base_dir or Path(f"/CT/SOMA/static00/{subject}")

    paths = {
        "raw_dir":       base_dir / "raw",
        "processed_dir": base_dir / "preprocessed_vFinal_clean",
        "canonical_data": base_dir / "canonical_model" / f"{subject}_canonical_data_tpose.json",
        "barycentric_map": base_dir / "canonical_model" / "generated_marker_barycentric_map.json",
    }

    logger.info(f"Starting Preprocessing for Subject: {subject}")
    logger.info(f"Base dir:        {base_dir}")
    logger.info(f"Raw dir:         {paths['raw_dir']}")
    logger.info(f"Output dir:      {paths['processed_dir']}")
    logger.info(f"Canonical data:  {paths['canonical_data']}")
    logger.info(f"Barycentric map: {paths['barycentric_map']}")

    if not paths["raw_dir"].exists():
        logger.error(f"Raw data dir not found: {paths['raw_dir']}")
        return

    # Create Output Dirs
    for sub in ["pose_rotations", "residuals", "masks", "canonical_lbs"]:
        (paths["processed_dir"] / sub).mkdir(parents=True, exist_ok=True)

    # Load Model
    p_bind, marker_map = load_canonical_model(paths["canonical_data"], paths["barycentric_map"])

    # Find Shots
    shot_folders = sorted(list(paths["raw_dir"].glob("shot_*_captury")))
    logger.info(f"Found {len(shot_folders)} shots.")

    # Process Shots
    success_count = 0
    for shot in tqdm(shot_folders, desc="Shots"):
        if process_shot(shot, paths["processed_dir"], p_bind, marker_map, subject):
            success_count += 1

    logger.success(f"Preprocessing complete. Processed {success_count}/{len(shot_folders)} shots.")

if __name__ == '__main__':
    main()