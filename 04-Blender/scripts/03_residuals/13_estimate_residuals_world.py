"""
STATUS: Optimized (Streaming) - Calculates world-space residuals.
Script: estimate_residuals_world_stream.py
Goal: Writes residuals to disk frame-by-frame to prevent MemoryErrors.
"""

import bpy
import json
import numpy as np
from mathutils import Vector
import os
import gc

# --- User Configuration ---
shots = ["shot_001"]



CANONICAL_MARKERS_JSON_PATH = "S:/work/03-MUSK/04-Blender/data/registration/S5/canonical_model/S5_canonical_data_tpose.json"
MARKER_LBS_WEIGHTS_JSON_PATH = "S:/work/03-MUSK/04-Registration/registration/S5/canonical_model/S5_marker_lbs_weights_exported.json"

def calculate_difference_vectors_streaming(shot):
    """
    Calculates residuals and streams them directly to the JSON file.
    """
    print(f"\n--- Processing Shot: {shot} (Streaming Mode) ---")

    # Paths
    EXPORTED_LBS_MARKERS_JSON_PATH = f"S:/work/03-MUSK/04-Blender/data/registration/S5/{shot}/S5_canonical_markers_lbs_{shot}_exported_tpose.json"
    MOTION_DATA_JSON_PATH = f"S:/work/03-MUSK/04-Blender/data/registration/S5/{shot}/S5_triangulated_sequence_{shot}_transformed.json"
    OUTPUT_DIFFERENCE_LBS_JSON_PATH = f"S:/work/03-MUSK/04-Blender/data/registration/S5/{shot}/S5_residuals_{shot}_world_lbs_tpose.json"

    # --- 1. Load Input Data ---
    print("Loading Input Data...")
    try:
        with open(CANONICAL_MARKERS_JSON_PATH, 'r') as f:
            canonical_points_raw = json.load(f).get("0", {})
        
        # NOTE: If these files are massive (2GB+), loading them might still be risky.
        # But streaming the output usually frees up enough RAM to make it work.
        with open(MOTION_DATA_JSON_PATH, 'r') as f:
            motion_data_by_frame = json.load(f)
        with open(EXPORTED_LBS_MARKERS_JSON_PATH, 'r') as f:
            exported_lbs_data = json.load(f)
            
    except Exception as e:
        print(f"ERROR: Failed to load inputs. {e}"); return

    canonical_keys = list(canonical_points_raw.keys())
    sorted_frame_keys = sorted([k for k in motion_data_by_frame.keys() if k.isdigit()], key=int)
    total_frames = len(sorted_frame_keys)
    
    print(f"Loaded {len(canonical_keys)} markers and {total_frames} frames.")

    # --- 2. Stream Process & Write ---
    print(f"Streaming results to: {OUTPUT_DIFFERENCE_LBS_JSON_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_DIFFERENCE_LBS_JSON_PATH), exist_ok=True)

    try:
        with open(OUTPUT_DIFFERENCE_LBS_JSON_PATH, 'w') as f_out:
            f_out.write('{\n') # Start JSON

            for i, frame_str in enumerate(sorted_frame_keys):
                if i % 100 == 0:
                    print(f"  Processing frame {frame_str}...", end='\r')

                observed_markers_t = motion_data_by_frame.get(frame_str, {})
                lbs_markers_t = exported_lbs_data.get(frame_str, {})
                
                differences_lbs_for_this_frame = {}

                # Calculate Residuals for this frame
                for marker_key in canonical_keys:
                    # We need BOTH the observed motion AND the LBS reference to calculate a residual
                    if (marker_key in observed_markers_t) and (marker_key in lbs_markers_t):
                        
                        # Get Observed Position
                        p_observed = Vector(observed_markers_t[marker_key][0])
                        
                        # Get LBS Position
                        p_lbs = Vector(lbs_markers_t[marker_key][0])
                        
                        # Calculate Difference (Observed - LBS)
                        diff_vector = p_observed - p_lbs
                        
                        differences_lbs_for_this_frame[marker_key] = [diff_vector[:]]
                    else:
                        # If unobserved or missing LBS data, residual is 0
                        differences_lbs_for_this_frame[marker_key] = [[0.0, 0.0, 0.0]]

                # Write frame immediately
                json_str = json.dumps(differences_lbs_for_this_frame)
                f_out.write(f'  "{frame_str}": {json_str}')

                # Comma handling
                if i < total_frames - 1:
                    f_out.write(',\n')
                else:
                    f_out.write('\n')
                    
            f_out.write('}') # End JSON

        print(f"\nSUCCESS: Saved residuals for {shot}")

    except Exception as e:
        print(f"\nCRITICAL ERROR during write: {e}")

    # Clean up
    del motion_data_by_frame
    del exported_lbs_data
    gc.collect()

if __name__ == "__main__":
    for shot in shots:
        calculate_difference_vectors_streaming(shot)

# --

# """
# STATUS: Complete - World-space residuals calculation script for multiple shots.
# Script: estimate_residuals_world.py
# Goal: For each frame, calculate the world-space difference vector between the
#       observed marker positions and their LBS-posed counterparts for multiple shots.
# """

# import bpy
# import json
# import numpy as np
# from mathutils import Vector, Matrix
# import os

# # --- User Configuration ---

# # List of shots to process
# # shots = [
# #     "shot_002", "shot_003", "shot_004", "shot_005",
# #     "shot_006", "shot_007", "shot_008", "shot_009", "shot_010",
# #     "shot_011", "shot_012", "shot_013", "shot_014", "shot_015",
# #     "shot_016", "shot_017", "shot_018", "shot_019", "shot_020"
# # ]

# shots = ["shot_001"]  # For testing, process only this shot

# # CANONICAL_MARKERS_JSON_PATH = "C:/Users/ealvarad/00-Local/02-Python/Blender/data/registration/canonical_model/canonical_data_tpose.json"
# CANONICAL_MARKERS_JSON_PATH = "S:/work/03-MUSK/04-Blender/data/registration/S5/canonical_model/S5_canonical_data_tpose.json"

# # MARKER_LBS_WEIGHTS_JSON_PATH = "C:/Users/ealvarad/00-Local/02-Python/Blender/data/weights/canonical_model/lbs_markers/markers_lbs_weights_exported.json"
# MARKER_LBS_WEIGHTS_JSON_PATH = "S:/work/03-MUSK/04-Registration/registration/S5/canonical_model/S5_marker_lbs_weights_exported.json"

# # ----------------------------------------------------


# def calculate_difference_vectors(shot):
#     """Main function to calculate difference between observed and LBS points for a single shot."""

#     print(f"\n--- Processing Shot: {shot} ---")

#     # EXPORTED_LBS_MARKERS_JSON_PATH = f"C:/Users/ealvarad/00-Local/02-Python/Blender/data/displacements/{shot}/canonical/canonical_markers_lbs_{shot}_exported_tpose.json"
#     EXPORTED_LBS_MARKERS_JSON_PATH = f"S:/work/03-MUSK/04-Blender/data/registration/S5/{shot}/canonical_markers_lbs_shot_001_exported_tpose.json"

#     # MOTION_DATA_JSON_PATH = f"C:/Users/ealvarad/00-Local/02-Python/Blender/data/registration/{shot}/triangulated_sequence_{shot}_transformed_filtered.json"
#     MOTION_DATA_JSON_PATH = f"S:/work/03-MUSK/04-Blender/data/registration/S5/{shot}/triangulated_sequence_{shot}_transformed.json"

#     # ARMATURE_OBJECT_NAME = f"root_{shot[-3:]}"
#     ARMATURE_OBJECT_NAME = "root_001_S5_TPose"

#     # OUTPUT_DIFFERENCE_JSON_PATH = f"C:/Users/ealvarad/00-Local/02-Python/Blender/data/displacements/{shot}/residuals/residuals_{shot}_world.json"
#     # OUTPUT_DIFFERENCE_LBS_JSON_PATH = f"C:/Users/ealvarad/00-Local/02-Python/Blender/data/displacements/{shot}/residuals/residuals_{shot}_world_lbs_tpose.json"
#     OUTPUT_DIFFERENCE_LBS_JSON_PATH = f"S:/work/03-MUSK/04-Blender/data/registration/S5/{shot}/residuals_{shot}_world_lbs_tpose.json"

#     # --- 1. Load All Necessary Data Files ---
#     print("--- Step 1: Loading Input Data ---")
#     try:
#         with open(CANONICAL_MARKERS_JSON_PATH, 'r') as f:
#             canonical_points_raw = json.load(f).get("0", {})
#         with open(MARKER_LBS_WEIGHTS_JSON_PATH, 'r') as f:
#             marker_lbs_weights = json.load(f)
#         with open(MOTION_DATA_JSON_PATH, 'r') as f:
#             motion_data_by_frame = json.load(f)
#         with open(EXPORTED_LBS_MARKERS_JSON_PATH, 'r') as f:
#             exported_lbs_data = json.load(f)
#     except Exception as e:
#         print(f"ERROR: Failed to load one or more JSON files for {shot}. Check paths. Error: {e}");
#         return

#     canonical_points = {key: Vector(val[0]) for key, val in canonical_points_raw.items()}
#     print(f"Loaded {len(canonical_points)} canonical marker positions.")
#     print(f"Loaded LBS weights for {len(marker_lbs_weights)} marker points.")
#     print(f"Loaded observed motion data for {len(motion_data_by_frame)} frames.")
#     print(f"Loaded precomputed LBS marker data for {len(exported_lbs_data)} frames.")

#     # --- 2. Get Armature and Prepare Bind Pose Data ---
#     armature_obj = bpy.data.objects.get(ARMATURE_OBJECT_NAME)
#     if not armature_obj or armature_obj.type != 'ARMATURE':
#         print(f"ERROR: Armature object '{ARMATURE_OBJECT_NAME}' not found or is not an Armature.");
#         return
#     print(f"Successfully found armature object: '{armature_obj.name}'")

#     bind_pose_matrices = {b.name: armature_obj.matrix_world @ b.matrix_local for b in armature_obj.data.bones}
#     inverse_bind_matrices = {name: mat.inverted() for name, mat in bind_pose_matrices.items()}
#     bone_idx_to_name_map = {idx: bone.name for idx, bone in enumerate(armature_obj.data.bones)}
#     print(f"Calculated bind pose matrices for {len(bind_pose_matrices)} bones.")

#     # --- 3. Set up and Run Animation Loop ---
#     all_frames_differences = {}
#     all_frames_differences_lbs = {}

#     if not motion_data_by_frame:
#         print("Motion data file is empty. Nothing to process.");
#         return

#     sorted_frame_keys = sorted(motion_data_by_frame.keys(), key=int)
#     print(f"Processing {len(sorted_frame_keys)} frames found in motion data...")

#     for frame_str in sorted_frame_keys:
#         frame = int(frame_str)
#         if frame % 25 == 0:
#             print(f"  Processing frame {frame}...")

#         bpy.context.scene.frame_set(frame)
#         posed_bone_matrices = {bone.name: bone.matrix for bone in armature_obj.pose.bones}
#         observed_markers_t = motion_data_by_frame[frame_str]

#         differences_for_this_frame = {}
#         differences_lbs_for_this_frame = {}

#         # --- CORE LOGIC: Loop through ALL markers (observed and unobserved) ---
#         for marker_key in canonical_points.keys():
#             p_unposed = canonical_points.get(marker_key)
#             weights_info = marker_lbs_weights.get(marker_key)

#             # Check if the marker is observed in the current frame
#             if marker_key in observed_markers_t:
#                 p_motion_list = observed_markers_t[marker_key]

#                 """
#                 # --- Fixed Method 1: Using Weights ---
#                 if p_unposed and weights_info and weights_info.get("bone_indices"):
#                     # Calculate the LBS Posed Position: P_n^LBS(t)
#                     p_unposed_homogeneous = p_unposed.to_4d()
#                     blended_transform = Matrix.Identity(4)
#                     blended_transform.zero()
#                     for i, bone_idx in enumerate(weights_info["bone_indices"]):
#                         weight = weights_info["weights"][i]
#                         bone_name = bone_idx_to_name_map.get(bone_idx)
#                         if bone_name:
#                             skinning_matrix = posed_bone_matrices[bone_name] @ inverse_bind_matrices[bone_name]
#                             blended_transform += weight * skinning_matrix

#                     p_lbs_posed = (blended_transform @ p_unposed_homogeneous).to_3d()

#                     # Transform the LBS-deformed position into world space
#                     p_lbs_posed_world = armature_obj.matrix_world @ p_lbs_posed

#                     # Calculate the Difference Vector
#                     p_observed_motion = Vector(p_motion_list[0])
#                     difference_vector = p_observed_motion - p_lbs_posed_world

#                     # Store the final difference vector [dx, dy, dz]
#                     differences_for_this_frame[marker_key] = difference_vector[:]
#                 """

#                 # --- Method 2: Using Precomputed LBS Markers ---
#                 if marker_key in exported_lbs_data[frame_str]:
#                     p_lbs_precomputed = Vector(exported_lbs_data[frame_str][marker_key][0])
#                     p_observed_motion = Vector(p_motion_list[0])
#                     difference_vector_lbs = p_observed_motion - p_lbs_precomputed

#                     # Store the final difference vector [dx, dy, dz]
#                     differences_lbs_for_this_frame[marker_key] = difference_vector_lbs[:]
#             else:
#                 # --- Marker is UNOBSERVED ---
#                 # Assign a zero residual for unobserved markers
#                 # differences_for_this_frame[marker_key] = [0.0, 0.0, 0.0]
#                 differences_lbs_for_this_frame[marker_key] = [0.0, 0.0, 0.0]

#         # all_frames_differences[frame_str] = differences_for_this_frame
#         all_frames_differences_lbs[frame_str] = differences_lbs_for_this_frame

#     print("\nFinished processing all animation frames.")

#     # --- 4. Save the Final Results to JSON ---
#     def save_results(output_path, data):
#         output_dir = os.path.dirname(output_path)
#         os.makedirs(output_dir, exist_ok=True)
#         try:
#             with open(output_path, 'w') as f:
#                 json.dump(data, f, indent=2)
#             print(f"Save complete: {output_path}")
#         except Exception as e:
#             print(f"ERROR writing final JSON file: {e}")

#     # save_results(OUTPUT_DIFFERENCE_JSON_PATH, all_frames_differences)
#     save_results(OUTPUT_DIFFERENCE_LBS_JSON_PATH, all_frames_differences_lbs)


# # --- Run the main function for all shots ---
# if __name__ == "__main__":
#     for shot in shots:
#         calculate_difference_vectors(shot)