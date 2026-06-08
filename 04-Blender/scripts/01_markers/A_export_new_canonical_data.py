"""
STATUS: Fixed - Exports T-Pose data preserving original marker IDs.
Script: export_canonical_data.py
Goal: Exports marker positions from the deformed point cloud, ensuring the 
      JSON keys (IDs) match the original A-Pose file so weights can be mapped.
"""

import bpy
import json
from mathutils import Vector
from pathlib import Path

# --- User Configuration ---
shot = "shot_001"

FRAME_TO_EXPORT = 0
MARKER_CLOUD_OBJ_NAME = f"LBS_Canonical_Markers_S5_{shot}_APose"

# 1. Path to the NEW file we want to save (T-Pose)
OUTPUT_JSON_PATH = "S:/work/03-MUSK/03-Registration/registration/S5/canonical_model/S5_canonical_data_tpose.json"

# 2. Path to the ORIGINAL file (A-Pose) to steal the correct IDs/Keys
ORIGINAL_JSON_PATH = "S:/work/03-MUSK/03-Registration/registration/S5/canonical_model/S5_canonical_data.json"

# ----------------------------------------------------

def export_canonical_data():
    print(f"--- Exporting Canonical Data for Frame {FRAME_TO_EXPORT} ---")

    # --- 1. Get the Marker Cloud Object ---
    marker_cloud_obj = bpy.data.objects.get(MARKER_CLOUD_OBJ_NAME)
    if not marker_cloud_obj:
        print(f"ERROR: Marker cloud object '{MARKER_CLOUD_OBJ_NAME}' not found.")
        return

    # --- 2. Set Frame & Evaluate Mesh ---
    bpy.context.scene.frame_set(FRAME_TO_EXPORT)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = marker_cloud_obj.evaluated_get(depsgraph)
    mesh_eval = obj_eval.to_mesh()

    if not mesh_eval:
        print(f"ERROR: Could not evaluate mesh for '{MARKER_CLOUD_OBJ_NAME}'.")
        return

    # --- 3. Get World Positions of T-Pose Markers ---
    # These vertices are ordered 0..N based on how they were created
    world_vertices = [obj_eval.matrix_world @ v.co for v in mesh_eval.vertices]
    print(f"Detected {len(world_vertices)} vertices in the point cloud.")

    # --- 4. Load Original Keys to maintain IDs ---
    try:
        print(f"Loading original keys from: {ORIGINAL_JSON_PATH}")
        with open(ORIGINAL_JSON_PATH, 'r') as f:
            # We assume frame "0" exists and keys are in insertion order (Python 3.7+)
            original_data = json.load(f).get("0", {})
            
        # This list MUST match the order used when creating the point cloud object
        ordered_marker_keys = list(original_data.keys())
        
        if len(ordered_marker_keys) != len(world_vertices):
            print(f"WARNING: ID Count ({len(ordered_marker_keys)}) != Vertex Count ({len(world_vertices)}). Mismatch possible!")
            
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load original keys. Aborting to prevent bad export.\nError: {e}")
        return

    # --- 5. Map T-Pose Positions to Original IDs ---
    tpose_marker_data = {}
    
    for v_idx, world_pos in enumerate(world_vertices):
        if v_idx < len(ordered_marker_keys):
            marker_id = ordered_marker_keys[v_idx]
            # Save format: [[x, y, z]]
            tpose_marker_data[marker_id] = [[world_pos.x, world_pos.y, world_pos.z]]

    # --- 6. Save New JSON ---
    output_path = Path(OUTPUT_JSON_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, 'w') as f:
            json.dump({"0": tpose_marker_data}, f, indent=4)
        print(f"SUCCESS: Exported {len(tpose_marker_data)} markers to:")
        print(f"{output_path}")
    except Exception as e:
        print(f"ERROR: Failed to save JSON. {e}")

    # Cleanup
    obj_eval.to_mesh_clear()

if __name__ == "__main__":
    export_canonical_data()

















# -------------

# """
# STATUS: Updated - Exports the canonical data from the marker cloud object in the pose position.
# Script: export_canonical_data.py
# Goal: Exports the marker positions from the MARKER_CLOUD_OBJ_NAME object in a particular frame
#       as a JSON file in the same format as canonical_data.json, following the armature's pose position.
# """

# import bpy
# import json
# from mathutils import Vector
# from pathlib import Path

# # --- User Configuration ---
# # shot = "FBX-NewSkeleton-TPose"  # Change this to your shot name
# shot = "shot_001"  # Change this to your shot name

# FRAME_TO_EXPORT = 0  # Frame to export the canonical data from
# MARKER_CLOUD_OBJ_NAME = f"LBS_Canonical_Markers_S2_{shot}"  # Name of the marker cloud object

# # OUTPUT_JSON_PATH = f"S:/work/03-MUSK/04-Blender/data/registration/canonical_model/canonical_data_tpose_new.json"
# OUTPUT_JSON_PATH = "S:/work/03-MUSK/03-Registration/registration/S2/canonical_model/S2_canonical_data_tpose.json"

# # ----------------------------------------------------

# def export_canonical_data():
#     """
#     Exports the marker positions from the MARKER_CLOUD_OBJ_NAME object as a JSON file,
#     transformed by the armature's pose position for a single frame.
#     """
#     print(f"--- Exporting Canonical Data for Frame {FRAME_TO_EXPORT} ---")

#     # --- 1. Get the Marker Cloud Object ---
#     marker_cloud_obj = bpy.data.objects.get(MARKER_CLOUD_OBJ_NAME)
#     if not marker_cloud_obj:
#         print(f"ERROR: Marker cloud object '{MARKER_CLOUD_OBJ_NAME}' not found in the scene.")
#         return

#     # --- 2. Set the Frame ---
#     bpy.context.scene.frame_set(FRAME_TO_EXPORT)

#     # --- 3. Get the Evaluated, Deformed Mesh Data ---
#     depsgraph = bpy.context.evaluated_depsgraph_get()
#     obj_eval = marker_cloud_obj.evaluated_get(depsgraph)
#     mesh_eval = obj_eval.to_mesh()

#     if not mesh_eval:
#         print(f"ERROR: Could not evaluate the mesh for object '{MARKER_CLOUD_OBJ_NAME}'.")
#         return

#     # --- 4. Extract Vertex Positions in World Space ---
#     marker_positions = {}
#     world_vertices = [obj_eval.matrix_world @ v.co for v in mesh_eval.vertices]

#     # Load canonical marker keys to maintain consistent ordering
#     try:
#         with open(OUTPUT_JSON_PATH, 'r') as f:
#             static_pose_data = json.load(f).get("0", {})
#         ordered_marker_keys = list(static_pose_data.keys())
#     except:
#         print("Could not load canonical data to get key order. Saving by index.")
#         ordered_marker_keys = [f"marker_{i}" for i in range(len(world_vertices))]

#     # Map vertex positions to marker keys
#     for v_idx, world_position in enumerate(world_vertices):
#         if v_idx < len(ordered_marker_keys):
#             marker_key = ordered_marker_keys[v_idx]
#             marker_positions[marker_key] = [[world_position.x, world_position.y, world_position.z]]

#     print(f"Extracted {len(marker_positions)} marker positions in pose position.")

#     # Clear the evaluated mesh
#     obj_eval.to_mesh_clear()

#     # --- 5. Save to JSON ---
#     output_path = Path(OUTPUT_JSON_PATH)
#     output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists
#     try:
#         with open(output_path, 'w') as f:
#             json.dump({"0": marker_positions}, f, indent=2)
#         print(f"Canonical data successfully exported to: {output_path}")
#     except Exception as e:
#         print(f"ERROR: Failed to save canonical data to {output_path}. Error: {e}")

# # --- Run the Export Function ---
# if __name__ == "__main__":
#     export_canonical_data()