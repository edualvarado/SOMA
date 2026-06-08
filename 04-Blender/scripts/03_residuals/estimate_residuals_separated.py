"""

STATUS: CHECK PROBLEM WITH SCALE, WE NEED TO SOLVE IT

Script: estimate_residuals_separated.py
Goal: For each frame, calculate the local, non-rigid displacement for all
      OBSERVED markers. For UNOBSERVED markers, assign a zero displacement vector.
      Saves the results into two separate files.
"""

import bpy
import json
import numpy as np
from mathutils import Vector, Matrix
import os

# --- User Configuration ---

# ---
shot = "shot_001"  # Change this to your shot name
# ---

CANONICAL_MARKERS_JSON_PATH = "C:/Users/ealvarad/00-Local/02-Python/Blender/data/registration/canonical_model/canonical_data.json"
MARKER_LBS_WEIGHTS_JSON_PATH = "C:/Users/ealvarad/00-Local/02-Python/Blender/data/weights/canonical_model/lbs_markers/exported_marker_lbs_weights_with_names.json"

# MOTION_DATA_JSON_PATH = f"C:/Users/ealvarad/00-Local/02-Python/Blender/data/registration/{shot}/triangulated_sequence_{shot}.json"  # Your observed data
MOTION_DATA_JSON_PATH = f"C:/Users/ealvarad/00-Local/02-Python/Blender/data/registration/{shot}/triangulated_sequence_{shot}_transformed_filtered.json"  # Your observed data

ARMATURE_OBJECT_NAME = f"root_{shot[-3:]}"

# Paths for the TWO final output files
OUTPUT_OBSERVED_DISPLACEMENTS_JSON_PATH = f"C:/Users/ealvarad/00-Local/02-Python/Blender/data/displacements/{shot}/residuals/observed_residuals_only_{shot}_corrected.json"
OUTPUT_UNOBSERVED_DISPLACEMENTS_JSON_PATH = f"C:/Users/ealvarad/00-Local/02-Python/Blender/data/displacements/{shot}/residuals/unobserved_residuals_only_{shot}_corrected.json"

PARENT_COLLECTION_NAME = shot.capitalize()

# ----------------------------------------------------


def calculate_and_separate_displacements():
    """Main function to calculate and separate observed/unobserved displacements."""

    # --- 1. Load All Necessary Data Files ---
    print("--- Step 1: Loading Input Data ---")
    try:
        with open(CANONICAL_MARKERS_JSON_PATH, 'r') as f:
            canonical_points_raw = json.load(f).get("0", {})
        with open(MARKER_LBS_WEIGHTS_JSON_PATH, 'r') as f:
            marker_lbs_weights = json.load(f)
        with open(MOTION_DATA_JSON_PATH, 'r') as f:
            motion_data_by_frame = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load one or more JSON files. Check paths. Error: {e}"); return

    canonical_points = {key: Vector(val[0]) for key, val in canonical_points_raw.items()}
    print(f"Loaded {len(canonical_points)} canonical marker positions.")
    print(f"Loaded LBS weights for {len(marker_lbs_weights)} marker points.")
    print(f"Loaded observed motion data for {len(motion_data_by_frame)} frames.")

    # --- 2. Get Armature and Prepare Bind Pose Data ---
    armature_obj = bpy.data.objects.get(ARMATURE_OBJECT_NAME)
    if not armature_obj or armature_obj.type != 'ARMATURE':
        print(f"ERROR: Armature object '{ARMATURE_OBJECT_NAME}' not found."); return
    print(f"Successfully found armature object: '{armature_obj.name}'")

    bind_pose_matrices = {b.name: armature_obj.matrix_world @ b.matrix_local for b in armature_obj.data.bones}
    inverse_bind_matrices = {name: mat.inverted() for name, mat in bind_pose_matrices.items()}
    bone_idx_to_name_map = {idx: bone.name for idx, bone in enumerate(armature_obj.data.bones)}
    print("Prepared bind pose matrices.")

    # --- 3. Set up and Run Animation Loop ---
    all_frames_observed_displacements = {}
    all_frames_unobserved_displacements = {}

    if not motion_data_by_frame:
        print("Motion data file is empty. Nothing to process."); return

    sorted_frame_keys = sorted(motion_data_by_frame.keys(), key=int)
    print(f"Processing {len(sorted_frame_keys)} frames...")

    for frame_str in sorted_frame_keys:
        frame = int(frame_str)
        if frame % 25 == 0:
            print(f"  Processing frame {frame}...")

        bpy.context.scene.frame_set(frame)
        posed_bone_matrices = {bone.name: bone.matrix for bone in armature_obj.pose.bones}
        observed_markers_t = motion_data_by_frame[frame_str]

        observed_displacements_t = {}
        unobserved_displacements_t = {}

        # --- CORE LOGIC: Loop through ALL canonical markers ---
        for marker_key, p_unposed in canonical_points.items():

            # Check if this canonical marker was observed in the current frame
            if marker_key in observed_markers_t:
                # --- THIS IS AN OBSERVED MARKER ---
                weights_info = marker_lbs_weights.get(marker_key)
                if not weights_info or not weights_info.get("bone_indices"): continue # Skip if no weights

                # Calculate its Rigid Reference Position
                primary_bone_idx = weights_info["bone_indices"][np.argmax(weights_info["weights"])]
                primary_bone_name = bone_idx_to_name_map[primary_bone_idx]
                rigid_transform_matrix = posed_bone_matrices[primary_bone_name] @ inverse_bind_matrices[primary_bone_name]
                p_rigid_ref = (rigid_transform_matrix @ p_unposed.to_4d()).to_3d()

                # Calculate World-Space Residual against the observation
                p_motion = Vector(observed_markers_t[marker_key][0])
                d_world_residual = p_motion - p_rigid_ref

                # Transform Residual to Local Bone Space
                R_primary_bone_pose_inv = posed_bone_matrices[primary_bone_name].to_3x3().transposed()
                d_local = R_primary_bone_pose_inv @ d_world_residual

                observed_displacements_t[marker_key] = d_local[:]
            else:
                # --- THIS IS AN UNOBSERVED MARKER ---
                # Its non-rigid displacement is assumed to be zero.
                unobserved_displacements_t[marker_key] = [0.0, 0.0, 0.0]

        all_frames_observed_displacements[frame_str] = observed_displacements_t
        all_frames_unobserved_displacements[frame_str] = unobserved_displacements_t

    print("\nFinished processing all animation frames.")

    # --- 4. Save the Final Results to TWO JSON Files ---
    print(f"Saving OBSERVED displacement data to: {OUTPUT_OBSERVED_DISPLACEMENTS_JSON_PATH}")

    # Ensure the directory exists
    output_dir = os.path.dirname(OUTPUT_OBSERVED_DISPLACEMENTS_JSON_PATH)
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(OUTPUT_OBSERVED_DISPLACEMENTS_JSON_PATH, 'w') as f:
            json.dump(all_frames_observed_displacements, f, indent=2)
        print("Save complete.")
    except Exception as e:
        print(f"ERROR writing observed displacements JSON: {e}")

    print(f"Saving UNOBSERVED displacement data to: {OUTPUT_UNOBSERVED_DISPLACEMENTS_JSON_PATH}")

    # Ensure the directory exists
    output_dir = os.path.dirname(OUTPUT_UNOBSERVED_DISPLACEMENTS_JSON_PATH)
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(OUTPUT_UNOBSERVED_DISPLACEMENTS_JSON_PATH, 'w') as f:
            json.dump(all_frames_unobserved_displacements, f, indent=2)
        print("Save complete.")
    except Exception as e:
        print(f"ERROR writing unobserved displacements JSON: {e}")

# --- Run the main function ---
if __name__ == "__main__":
    calculate_and_separate_displacements()