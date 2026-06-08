"""
Script: bake_deformation_to_sequence.py
Goal: Instead of a real-time visualization, this script calculates the final
      deformation for each frame and saves each deformed mesh as a separate .obj file.
"""

import bpy
import json
import numpy as np
from mathutils import Vector, Matrix
# Ensure you have scipy installed in Blender's Python environment
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import os

# --- User Configuration ---

shot = "shot_014"  # Change this to your shot name

# --- The mesh you want to deform (e.g., the combined muscle mesh)
SOURCE_MESH_OBJECT_NAME = f"canonical_muscle_000"
ARMATURE_OBJECT_NAME = f"root_{shot[-3:]}"

# --- Input File Paths ---
CANONICAL_MARKERS_JSON_PATH = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/registration/canonical_model/canonical_data.json"
MARKER_LBS_WEIGHTS_JSON_PATH = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/weights/canonical_model/marker_lbs_weights.json"
DENSE_DISPLACEMENTS_JSON_PATH = f"C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/displacements/{shot}/reconstruction/refined_two_pass_displacements_{shot}.json"
SKIN_INTERPOLATION_WEIGHTS_JSON = f"C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/weights/canonical_model/muscle_layer_interpolation_weights.json"

# --- Output Configuration ---
# The FOLDER where the .obj sequence will be saved
OUTPUT_DIRECTORY = f"C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/output/{shot}/"
# The base name for the output files
OUTPUT_FILENAME_BASE = f"deformed_mesh_{shot}"

# Make sure output directory exists
if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)

# --- Parameters ---
DATA_SCALE_FACTOR = 0.001

# ----------------------------------------------------

def bake_deformation_sequence():
    """
    A one-time process to calculate and save the entire animation as a mesh sequence.
    """
    print("--- Starting Deformation Bake Process ---")

    # --- 1. Load All Data and Prepare Cache ---
    # This section remains unchanged
    print("Loading data files...")
    try:
        with open(CANONICAL_MARKERS_JSON_PATH, 'r') as f:
            canonical_points_raw = json.load(f).get("0", {})
        with open(MARKER_LBS_WEIGHTS_JSON_PATH, 'r') as f:
            marker_lbs_weights = json.load(f)
        with open(DENSE_DISPLACEMENTS_JSON_PATH, 'r') as f:
            dense_displacements_by_frame = json.load(f)
        with open(SKIN_INTERPOLATION_WEIGHTS_JSON, 'r') as f:
            interpolation_weights = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load JSON files: {e}"); return

    source_mesh_obj = bpy.data.objects.get(SOURCE_MESH_OBJECT_NAME)
    if not source_mesh_obj: print(f"ERROR: Source mesh '{SOURCE_MESH_OBJECT_NAME}' not found."); return
    armature_obj = bpy.data.objects.get(ARMATURE_OBJECT_NAME)
    if not armature_obj: print(f"ERROR: Armature object '{ARMATURE_OBJECT_NAME}' not found."); return

    bind_pose_rotations = {idx: (armature_obj.matrix_world @ b.matrix_local).to_3x3() for idx, b in
                           enumerate(armature_obj.data.bones)}
    primary_bone_map = {key: wd["bone_indices"][np.argmax(wd["weights"])] for key, wd in marker_lbs_weights.items() if
                        wd and wd.get("bone_indices")}
    ordered_marker_keys = sorted(list(canonical_points_raw.keys()))
    marker_key_to_idx_map = {key: i for i, key in enumerate(ordered_marker_keys)}
    num_markers = len(ordered_marker_keys)

    all_frames_marker_offsets = {}
    for frame_str, marker_disps_local in dense_displacements_by_frame.items():
        marker_displacements_unposed_t = np.zeros((num_markers, 3), dtype=np.float32)
        for marker_key, d_local_list in marker_disps_local.items():
            marker_idx = marker_key_to_idx_map.get(marker_key)
            primary_bone_idx = primary_bone_map.get(marker_key)
            if marker_idx is not None and primary_bone_idx is not None:
                R_bind = bind_pose_rotations[primary_bone_idx]
                d_unposed_offset = R_bind @ Vector(d_local_list)
                marker_displacements_unposed_t[marker_idx] = d_unposed_offset[:]
        all_frames_marker_offsets[frame_str] = marker_displacements_unposed_t

    num_skin_verts = len(source_mesh_obj.data.vertices)
    rows, cols, data = [], [], []
    for skin_v_idx_str, interp_data in interpolation_weights.items():
        skin_v_idx = int(skin_v_idx_str)
        if skin_v_idx < num_skin_verts:
            for i, marker_key in enumerate(interp_data["influencing_markers"]):
                marker_idx = marker_key_to_idx_map.get(marker_key)
                if marker_idx is not None:
                    rows.append(skin_v_idx);
                    cols.append(marker_idx);
                    data.append(interp_data["interpolation_weights"][i])
    W_interp = csc_matrix((data, (rows, cols)), shape=(num_skin_verts, num_markers))

    p_unposed_skin_vertices = np.array([source_mesh_obj.matrix_world @ v.co for v in source_mesh_obj.data.vertices],
                                       dtype=np.float32)
    print("All data prepared for baking.")

    # --- 2. Create a single duplicate object to modify and export ---
    if "Bake_Object" in bpy.data.objects: bpy.data.objects.remove(bpy.data.objects["Bake_Object"], do_unlink=True)
    bake_obj = source_mesh_obj.copy()
    bake_obj.data = source_mesh_obj.data.copy()
    bake_obj.name = "Bake_Object"
    bpy.context.collection.objects.link(bake_obj)

    # --- 3. Loop Through Frames, Deform Mesh, and Export ---
    print(f"\nBaking animation to OBJ sequence in: {OUTPUT_DIRECTORY}")
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    sorted_frame_keys = sorted(dense_displacements_by_frame.keys(), key=int)
    for frame_str in sorted_frame_keys:
        frame = int(frame_str)
        print(f"  Baking frame {frame}...")

        D_marker_t = all_frames_marker_offsets.get(frame_str, np.zeros((num_markers, 3), dtype=np.float32))
        D_skin_t_original_units = W_interp @ D_marker_t
        P_skin_final_t = p_unposed_skin_vertices + (D_skin_t_original_units * DATA_SCALE_FACTOR)

        bake_obj.data.vertices.foreach_set("co", P_skin_final_t.ravel())
        bake_obj.data.update()

        bpy.ops.object.select_all(action='DESELECT')
        bake_obj.select_set(True)
        bpy.context.view_layer.objects.active = bake_obj

        output_filepath = os.path.join(OUTPUT_DIRECTORY, f"{OUTPUT_FILENAME_BASE}.{frame:04d}.obj")

        # --- THIS IS THE CORRECTED EXPORT CALL for Blender 4.0+ ---
        bpy.ops.wm.obj_export(
            filepath=output_filepath,
            export_selected_objects=True,  # This keyword replaced 'use_selection'
            # The axis keywords may have also changed slightly
            # forward_axis='-Z', # Use if needed
            # up_axis='Y'      # Use if needed
        )

    # --- 4. Clean up the bake object ---
    bpy.data.objects.remove(bake_obj, do_unlink=True)
    print("\n--- Bake Complete ---")


if __name__ == "__main__":
    # Ensure numpy is available if running directly in Blender's Text Editor
    try:
        import numpy as np
        from scipy.sparse import csc_matrix
        from scipy.sparse.linalg import spsolve
    except ImportError:
        print("ERROR: numpy or scipy not found. Please install them in Blender's Python environment.")
        # This function will fail if libs are missing, so exit early
        # Or raise an exception. For now, just printing.

    bake_deformation_sequence()