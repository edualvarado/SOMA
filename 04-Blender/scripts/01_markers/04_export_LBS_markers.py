"""
STATUS: Optimized (Streaming) - Exports LBS-deformed marker positions frame-by-frame.
Script: export_LBS_markers_streaming_multi_subject.py
Goal: Writes JSON incrementally to prevent MemoryError, running across multiple subjects.
"""

import bpy
import json
import os
import gc

# --- User Configuration ---
shots = [
    "shot_001"
]

# List of subjects to process
subjects = ["S3", "S4", "S5"]

# ----------------------------------------------------

def export_deformed_positions_streaming(subject, shot):
    """
    Exports the world-space positions of a deformed mesh's vertices, streaming directly to disk.
    """
    print(f"--- Starting Export for Subject: {subject} | Shot: {shot} (Streaming Mode) ---")

    # --- 1. Set up subject/shot-specific variables ---
    
    # Path to the canonical data for this subject
    # Replaces S2 with the current subject
    CANONICAL_MARKERS_JSON_PATH = f"S:/work/03-MUSK/04-Blender/data/registration/{subject}/canonical_model/{subject}_canonical_data_tpose.json"
    
    # Blender Object Names (Dynamically using subject and shot)
    # Assumes naming convention: LBS_Canonical_Markers_SX_shot_XXX_TPose
    MARKER_CLOUD_OBJ_NAME = f"LBS_Canonical_Markers_{subject}_{shot}_TPose"
    
    # Assumes naming convention: unknown_001_SX
    # Note: If '001' is tied to the shot number, you might need to parse {shot} (e.g. shot.split('_')[1])
    # For now, keeping "001" as requested, or assuming it matches the shot. 
    # If your animation object name changes with the shot (e.g. unknown_002_S3), use: f"unknown_{shot.split('_')[-1]}_{subject}"
    animation_obj_name = f"unknown_001_{subject}" 

    # Output Path
    OUTPUT_LBS_MOTION_JSON_PATH = f"S:/work/03-MUSK/04-Blender/data/registration/{subject}/{shot}/{subject}_canonical_markers_lbs_{shot}_exported_tpose.json"

    # --- 2. Get the Deformed Object ---
    obj = bpy.data.objects.get(MARKER_CLOUD_OBJ_NAME)
    if not obj:
        print(f"ERROR: Object '{MARKER_CLOUD_OBJ_NAME}' not found. Skipping {subject}/{shot}.")
        return

    # Determine Frame Range from Animation Object or Scene
    animation_obj = bpy.data.objects.get(animation_obj_name)
    if animation_obj and animation_obj.animation_data and animation_obj.animation_data.action:
        end_frame = int(animation_obj.animation_data.action.frame_range[1])
    else:
        # Fallback to scene end frame if specific animation object isn't found
        print(f"Warning: Animation object '{animation_obj_name}' not found. Using scene frame end.")
        end_frame = bpy.context.scene.frame_end
    
    start_frame = bpy.context.scene.frame_start
    frame_range = range(start_frame, end_frame + 1)
    total_frames = len(frame_range)

    # --- 3. Pre-load Canonical Mapping (Do this ONCE per subject) ---
    print(f"Loading canonical mapping from: {CANONICAL_MARKERS_JSON_PATH}")
    try:
        with open(CANONICAL_MARKERS_JSON_PATH, 'r') as f:
            static_pose_data = json.load(f).get("0", {})
        ordered_marker_keys = list(static_pose_data.keys())
    except Exception as e:
        print(f"Warning: Could not load canonical data ({e}). Saving by index.")
        ordered_marker_keys = None # Will fallback to index inside loop

    # --- 4. Stream Process & Write ---
    print(f"Streaming results to: {OUTPUT_LBS_MOTION_JSON_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_LBS_MOTION_JSON_PATH), exist_ok=True)
    
    depsgraph = bpy.context.evaluated_depsgraph_get()

    try:
        with open(OUTPUT_LBS_MOTION_JSON_PATH, 'w') as f_out:
            f_out.write('{\n') # Start JSON

            for i, frame in enumerate(frame_range):
                if i % 50 == 0:
                    print(f"  Exporting frame {frame}/{end_frame}...", end='\r')

                # Update Scene
                bpy.context.scene.frame_set(frame)
                
                # Evaluate Mesh
                obj_eval = obj.evaluated_get(depsgraph)
                mesh_eval = obj_eval.to_mesh()
                
                # Extract World Vertices
                matrix_world = obj_eval.matrix_world
                world_vertices = [matrix_world @ v.co for v in mesh_eval.vertices]
                num_verts = len(world_vertices)

                # Fallback mapping if load failed
                if ordered_marker_keys is None:
                    current_keys = [f"vertex_{x}" for x in range(num_verts)]
                else:
                    current_keys = ordered_marker_keys

                # Build Frame Dictionary
                positions_for_this_frame = {}
                for v_idx in range(min(num_verts, len(current_keys))):
                    marker_key = current_keys[v_idx]
                    # Store as list of lists to match format: "key": [[x,y,z]]
                    positions_for_this_frame[marker_key] = [world_vertices[v_idx][:]]

                # Write to File
                json_str = json.dumps(positions_for_this_frame)
                f_out.write(f'  "{frame}": {json_str}')

                # Handle Comma
                if i < total_frames - 1:
                    f_out.write(',\n')
                else:
                    f_out.write('\n')

                # Cleanup
                obj_eval.to_mesh_clear()
                
            f_out.write('}') # End JSON

        print(f"\nSUCCESS: Saved LBS markers for {subject}/{shot}")

    except Exception as e:
        print(f"\nCRITICAL ERROR during export for {subject}/{shot}: {e}")

    # Final Cleanup
    gc.collect()

if __name__ == "__main__":
    # Loop over all subjects and shots
    for subject in subjects:
        for shot in shots:
            export_deformed_positions_streaming(subject, shot)

# --

# """
# STATUS: Optimized (Streaming) - Exports LBS-deformed marker positions frame-by-frame.
# Script: export_LBS_markers_streaming.py
# Goal: Writes JSON incrementally to prevent MemoryError on large sequences.
# """

# import bpy
# import json
# import os
# import gc

# # --- User Configuration ---
# shots = [
#     "shot_001"
# ]

# CANONICAL_MARKERS_JSON_PATH = "S:/work/03-MUSK/04-Blender/data/registration/S2/canonical_model/S2_canonical_data_tpose.json"

# # ----------------------------------------------------

# def export_deformed_positions_streaming(shot):
#     """
#     Exports the world-space positions of a deformed mesh's vertices, streaming directly to disk.
#     """
#     print(f"--- Starting Export for Shot: {shot} (Streaming Mode) ---")

#     # --- 1. Set up shot-specific variables ---
#     MARKER_CLOUD_OBJ_NAME = f"LBS_Canonical_Markers_S2_shot_001_TPose"
#     OUTPUT_LBS_MOTION_JSON_PATH = f"S:/work/03-MUSK/04-Blender/data/registration/S2/{shot}/S2_canonical_markers_lbs_{shot}_exported_tpose.json"
#     animation_obj_name = f"unknown_001_S2"

#     # --- 2. Get the Deformed Object ---
#     obj = bpy.data.objects.get(MARKER_CLOUD_OBJ_NAME)
#     if not obj:
#         print(f"ERROR: Object '{MARKER_CLOUD_OBJ_NAME}' not found.")
#         return

#     # Determine Frame Range
#     animation_obj = bpy.data.objects.get(animation_obj_name)
#     if animation_obj and animation_obj.animation_data and animation_obj.animation_data.action:
#         end_frame = int(animation_obj.animation_data.action.frame_range[1])
#     else:
#         end_frame = bpy.context.scene.frame_end
    
#     start_frame = bpy.context.scene.frame_start
#     frame_range = range(start_frame, end_frame + 1)
#     total_frames = len(frame_range)

#     # --- 3. Pre-load Canonical Mapping (Do this ONCE) ---
#     print("Loading canonical mapping...")
#     try:
#         with open(CANONICAL_MARKERS_JSON_PATH, 'r') as f:
#             static_pose_data = json.load(f).get("0", {})
#         ordered_marker_keys = list(static_pose_data.keys())
#     except Exception as e:
#         print(f"Warning: Could not load canonical data ({e}). Saving by index.")
#         ordered_marker_keys = None # Will fallback to index inside loop

#     # --- 4. Stream Process & Write ---
#     print(f"Streaming results to: {OUTPUT_LBS_MOTION_JSON_PATH}")
#     os.makedirs(os.path.dirname(OUTPUT_LBS_MOTION_JSON_PATH), exist_ok=True)
    
#     depsgraph = bpy.context.evaluated_depsgraph_get()

#     try:
#         with open(OUTPUT_LBS_MOTION_JSON_PATH, 'w') as f_out:
#             f_out.write('{\n') # Start JSON

#             for i, frame in enumerate(frame_range):
#                 if i % 50 == 0:
#                     print(f"  Exporting frame {frame}/{end_frame}...", end='\r')

#                 # Update Scene
#                 bpy.context.scene.frame_set(frame)
                
#                 # Evaluate Mesh
#                 obj_eval = obj.evaluated_get(depsgraph)
#                 mesh_eval = obj_eval.to_mesh()
                
#                 # Extract World Vertices
#                 # (Matrix mult is faster with list comp than numpy for small counts inside Blender python)
#                 matrix_world = obj_eval.matrix_world
#                 world_vertices = [matrix_world @ v.co for v in mesh_eval.vertices]
#                 num_verts = len(world_vertices)

#                 # Fallback mapping if load failed
#                 if ordered_marker_keys is None:
#                     current_keys = [f"vertex_{x}" for x in range(num_verts)]
#                 else:
#                     current_keys = ordered_marker_keys

#                 # Build Frame Dictionary
#                 positions_for_this_frame = {}
#                 for v_idx in range(min(num_verts, len(current_keys))):
#                     marker_key = current_keys[v_idx]
#                     # Store as list of lists to match format: "key": [[x,y,z]]
#                     positions_for_this_frame[marker_key] = [world_vertices[v_idx][:]]

#                 # Write to File
#                 json_str = json.dumps(positions_for_this_frame)
#                 f_out.write(f'  "{frame}": {json_str}')

#                 # Handle Comma
#                 if i < total_frames - 1:
#                     f_out.write(',\n')
#                 else:
#                     f_out.write('\n')

#                 # Cleanup
#                 obj_eval.to_mesh_clear()
                
#             f_out.write('}') # End JSON

#         print(f"\nSUCCESS: Saved LBS markers for {shot}")

#     except Exception as e:
#         print(f"\nCRITICAL ERROR during export: {e}")

#     # Final Cleanup
#     gc.collect()

# if __name__ == "__main__":
#     for shot in shots:
#         export_deformed_positions_streaming(shot)

# --

# """
# STATUS: Completed - Export LBS-deformed marker positions to JSON after animation for multiple shots.
# Script: export_LBS_markers.py
# Goal: Iterate through an array of shots and export the final world-space coordinates
#       of the LBS-deformed marker points for each frame to JSON files.
# """

# import bpy
# import json
# from mathutils import Vector
# import os

# # --- User Configuration ---
# # shots = [
# #     "shot_002", "shot_003", "shot_004", "shot_005",
# #     "shot_006", "shot_007", "shot_008", "shot_009", "shot_010",
# #     "shot_011", "shot_012", "shot_013", "shot_014", "shot_015",
# #     "shot_016", "shot_017", "shot_018", "shot_019", "shot_020"
# # ]

# shots = [
#     "shot_001"
# ]

# # CANONICAL_MARKERS_JSON_PATH = "C:/Users/ealvarad/00-Local/02-Python/Blender/data/registration/canonical_model/canonical_data_tpose.json"
# CANONICAL_MARKERS_JSON_PATH = "S:/work/03-MUSK/04-Blender/data/registration/S5/canonical_model/S5_canonical_data_tpose.json"

# # ----------------------------------------------------

# def export_deformed_positions_for_shot(shot):
#     """
#     Exports the world-space positions of a deformed mesh's vertices for each frame for a given shot.
#     """
#     print(f"--- Starting Export for Shot: {shot} ---")

#     # --- 1. Set up shot-specific variables ---
#     # MARKER_CLOUD_OBJ_NAME = f"LBS_Canonical_Markers_{shot[-3:]}"
#     MARKER_CLOUD_OBJ_NAME = f"LBS_Canonical_Markers_S5_shot_001_TPose"

#     # OUTPUT_LBS_MOTION_JSON_PATH = f"C:/Users/ealvarad/00-Local/02-Python/Blender/data/displacements/{shot}/canonical/canonical_markers_lbs_{shot}_exported_tpose.json"
#     OUTPUT_LBS_MOTION_JSON_PATH = f"S:/work/03-MUSK/04-Blender/data/registration/S5/{shot}/canonical_markers_lbs_{shot}_exported_tpose.json"

#     # animation_obj_name = f"unknown_{shot[-3:]}"  # Adjust based on shot
#     animation_obj_name = f"unknown_001_S5"  # Adjust based on shot

#     # Dynamically find the last frame of the animation for the corresponding object
#     animation_obj = bpy.data.objects.get(animation_obj_name)
#     if animation_obj and animation_obj.animation_data and animation_obj.animation_data.action:
#         end_frame = int(animation_obj.animation_data.action.frame_range[1])
#     else:
#         print(f"WARNING: Could not find animation for object '{animation_obj_name}'. Using scene's frame_end.")
#         end_frame = bpy.context.scene.frame_end

#     # --- 2. Get the Deformed Object ---
#     obj = bpy.data.objects.get(MARKER_CLOUD_OBJ_NAME)
#     if not obj:
#         print(f"ERROR: Object '{MARKER_CLOUD_OBJ_NAME}' not found.")
#         print("Please run the 'visualize_LBS_markers.py' script first to create it.")
#         return

#     # --- 3. Get the Evaluated, Deformed Mesh Data ---
#     depsgraph = bpy.context.evaluated_depsgraph_get()
#     all_frames_posed_data = {}

#     start_frame = bpy.context.scene.frame_start
#     print(f"Will export animation from frame {start_frame} to {end_frame}.")

#     for frame in range(start_frame, end_frame + 1):
#         if frame % 25 == 0:
#             print(f"  Exporting frame {frame}...")

#         bpy.context.scene.frame_set(frame)
#         obj_eval = obj.evaluated_get(depsgraph)
#         mesh_eval = obj_eval.to_mesh()

#         num_verts = len(mesh_eval.vertices)
#         world_vertices = [obj_eval.matrix_world @ v.co for v in mesh_eval.vertices]

#         try:
#             with open(CANONICAL_MARKERS_JSON_PATH, 'r') as f:
#                 static_pose_data = json.load(f).get("0", {})
#             ordered_marker_keys = list(static_pose_data.keys())
#         except:
#             print("Could not load canonical data to get key order. Saving by index.")
#             ordered_marker_keys = [f"vertex_{i}" for i in range(num_verts)]

#         positions_for_this_frame = {}
#         for v_idx in range(num_verts):
#             if v_idx < len(ordered_marker_keys):
#                 marker_key = ordered_marker_keys[v_idx]
#                 positions_for_this_frame[marker_key] = [world_vertices[v_idx][:]]

#         all_frames_posed_data[str(frame)] = positions_for_this_frame
#         obj_eval.to_mesh_clear()

#     print(f"Finished processing all animation frames for shot: {shot}.")

#     # --- 4. Save the Final Results to JSON ---
#     print(f"Saving LBS-posed positions to: {OUTPUT_LBS_MOTION_JSON_PATH}")
#     output_dir = os.path.dirname(OUTPUT_LBS_MOTION_JSON_PATH)
#     os.makedirs(output_dir, exist_ok=True)

#     try:
#         with open(OUTPUT_LBS_MOTION_JSON_PATH, 'w') as f:
#             json.dump(all_frames_posed_data, f, indent=2)
#         print(f"Save complete for shot: {shot}.")
#     except Exception as e:
#         print(f"ERROR writing final JSON file for shot {shot}: {e}")


# # --- Main Loop for All Shots ---
# if __name__ == "__main__":
#     for shot in shots:
#         export_deformed_positions_for_shot(shot)