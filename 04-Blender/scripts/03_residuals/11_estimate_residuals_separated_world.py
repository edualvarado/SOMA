"""
STATUS: Complete - Create world-space version of separated residuals script. Scale issue seems resolved.
Script: estimate_residuals_separated_world.py
Goal: For each frame, calculate the world-space displacement for all
      OBSERVED markers. For UNOBSERVED markers, assign a zero displacement vector.
      Saves the results into two separate files.
"""

import bpy
import json
import numpy as np
from mathutils import Vector, Matrix
import os

# --- User Configuration ---
shots = ["shot_001", "shot_002", "shot_003", "shot_004", "shot_005", "shot_006", "shot_007", "shot_008", "shot_009", "shot_010", "shot_011", "shot_012", "shot_013", "shot_014", "shot_015", "shot_016", "shot_017", "shot_018", "shot_019", "shot_020"]  # Add your shot names here
# shots = ["shot_001"]

CANONICAL_MARKERS_JSON_PATH = "C:/Users/ealvarad/00-Local/02-Python/Blender/data/registration/canonical_model/canonical_data.json"
MARKER_LBS_WEIGHTS_JSON_PATH = "C:/Users/ealvarad/00-Local/02-Python/Blender/data/weights/canonical_model/lbs_markers/markers_lbs_weights_exported.json"


# ----------------------------------------------------

def calculate_and_separate_displacements_world(shot):
    """Main function to calculate and separate observed/unobserved displacements in world space for a single shot."""

    # Update paths for the current shot
    global CANONICAL_MARKERS_JSON_PATH, MARKER_LBS_WEIGHTS_JSON_PATH, MOTION_DATA_JSON_PATH
    global OUTPUT_OBSERVED_DISPLACEMENTS_JSON_PATH, OUTPUT_UNOBSERVED_DISPLACEMENTS_JSON_PATH, ARMATURE_OBJECT_NAME

    EXPORTED_LBS_MARKERS_JSON_PATH = f"C:/Users/ealvarad/00-Local/02-Python/Blender/data/displacements/{shot}/canonical/canonical_markers_lbs_{shot}_exported.json"
    MOTION_DATA_JSON_PATH = f"C:/Users/ealvarad/00-Local/02-Python/Blender/data/registration/{shot}/triangulated_sequence_{shot}_transformed_filtered.json"

    # Output paths for the marker data computing LBS using weights
    OUTPUT_OBSERVED_DISPLACEMENTS_JSON_PATH = f"C:/Users/ealvarad/00-Local/02-Python/Blender/data/displacements/{shot}/residuals/observed_residuals_only_{shot}_world.json"
    OUTPUT_UNOBSERVED_DISPLACEMENTS_JSON_PATH = f"C:/Users/ealvarad/00-Local/02-Python/Blender/data/displacements/{shot}/residuals/unobserved_residuals_only_{shot}_world.json"

    # Output paths for the precomputed LBS marker data method
    OUTPUT_OBSERVED_DISPLACEMENTS_LBS_JSON_PATH = f"C:/Users/ealvarad/00-Local/02-Python/Blender/data/displacements/{shot}/residuals/observed_residuals_only_{shot}_world_lbs.json"
    OUTPUT_UNOBSERVED_DISPLACEMENTS_LBS_JSON_PATH = f"C:/Users/ealvarad/00-Local/02-Python/Blender/data/displacements/{shot}/residuals/unobserved_residuals_only_{shot}_world_lbs.json"

    ARMATURE_OBJECT_NAME = f"root_{shot[-3:]}"

    # --- 1. Load All Necessary Data Files ---
    print("--- Step 1: Loading Input Data ---")
    try:
        with open(CANONICAL_MARKERS_JSON_PATH, 'r') as f:
            canonical_points_raw = json.load(f).get("0", {})
        with open(MARKER_LBS_WEIGHTS_JSON_PATH, 'r') as f:
            marker_lbs_weights = json.load(f)
        with open(MOTION_DATA_JSON_PATH, 'r') as f:
            motion_data_by_frame = json.load(f)
        with open(EXPORTED_LBS_MARKERS_JSON_PATH, 'r') as f:
            exported_lbs_data = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load one or more JSON files. Check paths. Error: {e}"); return

    canonical_points = {key: Vector(val[0]) for key, val in canonical_points_raw.items()}
    print(f"Loaded {len(canonical_points)} canonical marker positions.")
    print(f"Loaded LBS weights for {len(marker_lbs_weights)} marker points.")
    print(f"Loaded observed motion data for {len(motion_data_by_frame)} frames.")
    print(f"Loaded precomputed LBS marker data for {len(exported_lbs_data)} frames.")

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
    all_frames_observed_displacements_lbs = {}
    all_frames_unobserved_displacements_lbs = {}


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
        observed_displacements_lbs_t = {}
        unobserved_displacements_lbs_t = {}

        # --- CORE LOGIC: Loop through ALL canonical markers ---
        for marker_key, p_unposed in canonical_points.items():

            """ 
            # Method 1: Using canonical data and LBS weights
            if marker_key in observed_markers_t:
                weights_info = marker_lbs_weights.get(marker_key)
                if not weights_info or not weights_info.get("bone_indices"): continue # Skip if no weights

                # Calculate the LBS Posed Position: P_n^LBS(t)
                p_unposed_homogeneous = p_unposed.to_4d()
                blended_transform = Matrix.Identity(4)
                blended_transform.zero()
                for i, bone_idx in enumerate(weights_info["bone_indices"]):
                    weight = weights_info["weights"][i]
                    bone_name = bone_idx_to_name_map.get(bone_idx)
                    if bone_name:
                        skinning_matrix = posed_bone_matrices[bone_name] @ inverse_bind_matrices[bone_name]
                        blended_transform += weight * skinning_matrix

                p_lbs_posed = (blended_transform @ p_unposed_homogeneous).to_3d()
                
                p_motion = Vector(observed_markers_t[marker_key][0])
                d_world_residual = p_motion - p_lbs_posed
                observed_displacements_t[marker_key] = d_world_residual[:]
            else:
                # --- THIS IS AN UNOBSERVED MARKER ---
                # Its non-rigid displacement is assumed to be zero.
                unobserved_displacements_t[marker_key] = [0.0, 0.0, 0.0]
            """

            # ----

            """
            # Fixed Method 1
            if marker_key in observed_markers_t:
                weights_info = marker_lbs_weights.get(marker_key)
                if not weights_info or not weights_info.get("bone_indices"): continue  # Skip if no weights

                # Calculate the LBS Posed Position: P_n^LBS(t)
                p_unposed_homogeneous = p_unposed.to_4d()
                blended_transform = Matrix.Identity(4)
                blended_transform.zero()
                for i, bone_idx in enumerate(weights_info["bone_indices"]):
                    weight = weights_info["weights"][i]
                    bone_name = bone_idx_to_name_map.get(bone_idx)
                    if bone_name:
                        skinning_matrix = posed_bone_matrices[bone_name] @ inverse_bind_matrices[bone_name]
                        blended_transform += weight * skinning_matrix

                # Compute the LBS-deformed position in local space
                p_lbs_posed = (blended_transform @ p_unposed_homogeneous).to_3d()

                # Transform the LBS-deformed position into world space
                p_lbs_posed_world = armature_obj.matrix_world @ p_lbs_posed

                # Compute the residual in world space
                p_motion = Vector(observed_markers_t[marker_key][0])
                d_world_residual = p_motion - p_lbs_posed_world
                observed_displacements_t[marker_key] = d_world_residual[:]
            else:
                # --- THIS IS AN UNOBSERVED MARKER ---
                # Its non-rigid displacement is assumed to be zero.
                unobserved_displacements_t[marker_key] = [0.0, 0.0, 0.0]
            """

            # ----

            # Method 2: Using precomputed LBS marker data
            if marker_key in observed_markers_t:
                p_lbs_precomputed = Vector(exported_lbs_data[frame_str][marker_key][0])
                p_motion = Vector(observed_markers_t[marker_key][0])
                d_world_residual_lbs = p_motion - p_lbs_precomputed
                observed_displacements_lbs_t[marker_key] = d_world_residual_lbs[:]
            else:
                unobserved_displacements_lbs_t[marker_key] = [0.0, 0.0, 0.0]

        # all_frames_observed_displacements[frame_str] = observed_displacements_t
        # all_frames_unobserved_displacements[frame_str] = unobserved_displacements_t
        all_frames_observed_displacements_lbs[frame_str] = observed_displacements_lbs_t
        all_frames_unobserved_displacements_lbs[frame_str] = unobserved_displacements_lbs_t

    print("\nFinished processing all animation frames.")

    # --- 4. Save the Final Results to JSON Files ---
    def save_results(output_path, data):
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        try:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Save complete: {output_path}")
        except Exception as e:
            print(f"ERROR writing JSON file: {e}")

    # save_results(OUTPUT_OBSERVED_DISPLACEMENTS_JSON_PATH, all_frames_observed_displacements)
    # save_results(OUTPUT_UNOBSERVED_DISPLACEMENTS_JSON_PATH, all_frames_unobserved_displacements)
    save_results(OUTPUT_OBSERVED_DISPLACEMENTS_LBS_JSON_PATH, all_frames_observed_displacements_lbs)
    save_results(OUTPUT_UNOBSERVED_DISPLACEMENTS_LBS_JSON_PATH, all_frames_unobserved_displacements_lbs)

# --- Run the main function ---
if __name__ == "__main__":
    for shot in shots:
        print(f"Processing shot: {shot}")
        calculate_and_separate_displacements_world(shot)
