"""
Script: visualize_layer_dense_deformation.py
Goal: Deforms a dense skin mesh smoothly and efficiently using the pre-calculated
      DENSE displacement data from the previous step.
"""

import bpy
import json
import numpy as np
from mathutils import Vector, Matrix
# Ensure you have scipy installed in Blender's Python environment
from scipy.sparse import csc_matrix

# --- User Configuration ---

# ---
shot = "shot_014"  # Change this to your shot name
# ---

# INPUT
# SKIN_MESH_OBJECT_NAME = f"canonical_skin_{shot[-3:]}"  # The original mesh to be deformed
SKIN_MESH_OBJECT_NAME = f"canonical_muscle_{shot[-3:]}"  # The original mesh to be deformed

ARMATURE_OBJECT_NAME = f"root_{shot[-3:]}"

CANONICAL_MARKERS_JSON_PATH = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/registration/canonical_model/canonical_data.json"
MARKER_LBS_WEIGHTS_JSON_PATH = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/weights/canonical_model/marker_lbs_weights.json"

# DENSE_DISPLACEMENTS_JSON_PATH = f"C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/displacements/{shot}/reconstruction/dense_local_displacements_{shot}.json"
DENSE_DISPLACEMENTS_JSON_PATH = f"C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/displacements/{shot}/reconstruction/refined_two_pass_displacements_{shot}.json"

# SKIN_INTERPOLATION_WEIGHTS_JSON = f"C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/weights/canonical_model/reconstruction/skin_layer_interpolation_weights.json"
SKIN_INTERPOLATION_WEIGHTS_JSON = f"C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/weights/canonical_model/reconstruction/muscle_layer_interpolation_weights.json"

# --- Visualization Settings ---
DEFORMED_SKIN_OBJ_NAME = f"Deformed_Layer_Result_{shot[-3:]}"
COLLECTION_NAME = f"Deformed_Layer_Result_{shot[-3:]}"
PARENT_COLLECTION_NAME = shot.capitalize()
EXAGGERATION_SCALE_DEFAULT = 0.001
# ----------------------------------------------------

# --- Global dictionary to hold all pre-computed data ---
DEFORMATION_CACHE = {}


def on_frame_change_final_skin(scene):
    """
    This handler deforms the skin mesh using fast, pre-computed data.
    """
    deformed_obj = scene.objects.get(DEFORMED_SKIN_OBJ_NAME)
    if not deformed_obj or "marker_displacements_unposed" not in DEFORMATION_CACHE:
        return

    current_frame_str = str(scene.frame_current)

    # 1. Get pre-computed marker displacements for this frame
    D_marker_t = DEFORMATION_CACHE["marker_displacements_unposed"].get(
        current_frame_str, DEFORMATION_CACHE["zero_marker_displacements"]
    )

    scale_factor = scene.get("skin_deform_exaggeration_scale", 1.0)

    # 2. Interpolate skin displacements with a single sparse matrix multiplication
    W_interp = DEFORMATION_CACHE["interpolation_matrix"]
    D_skin_t = W_interp @ D_marker_t

    # 3. Calculate final vertex positions
    P_skin_unposed = DEFORMATION_CACHE["p_unposed_skin_vertices"]
    P_skin_final_t = P_skin_unposed + (D_skin_t * scale_factor)

    # 4. Update mesh efficiently
    deformed_obj.data.vertices.foreach_set("co", P_skin_final_t.ravel())
    deformed_obj.data.update()


def setup_final_skin_visualization():
    """
    Performs a one-time setup to pre-compute all necessary data
    for real-time performance.
    """
    global DEFORMATION_CACHE
    print("--- Setting up Final Skin Deformation Visualization ---")

    # --- 1. Load Data ---
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

    # --- 2. Get Scene Objects and Basic Data ---

    # CHANGE TO SKIN/MUSCLE OBJECT
    # canonical_obj = bpy.data.objects.get("canonical_skin_000")
    canonical_obj = bpy.data.objects.get("canonical_muscle_000")
    if not canonical_obj:
        print(f"ERROR: Mesh {canonical_obj} not found in the scene.")
        return

    canonical_obj.hide_set(True)

    # Make a copy of the mesh
    source_skin_obj = canonical_obj.copy()
    source_skin_obj.data = canonical_obj.data.copy()
    source_skin_obj.name = SKIN_MESH_OBJECT_NAME  # Assign the new name

    source_skin_obj = bpy.data.objects.get(SKIN_MESH_OBJECT_NAME)
    if not source_skin_obj: print(f"ERROR: Source skin mesh '{SKIN_MESH_OBJECT_NAME}' not found."); return
    armature_obj = bpy.data.objects.get(ARMATURE_OBJECT_NAME)
    if not armature_obj: print(f"ERROR: Armature object '{ARMATURE_OBJECT_NAME}' not found."); return

    # --- 3. Pre-compute Bind Data and Mappings ---
    bind_pose_rotations = {idx: (armature_obj.matrix_world @ b.matrix_local).to_3x3() for idx, b in
                           enumerate(armature_obj.data.bones)}
    primary_bone_map = {key: wd["bone_indices"][np.argmax(wd["weights"])] for key, wd in marker_lbs_weights.items() if
                        wd and wd.get("bone_indices")}

    ordered_marker_keys = sorted(list(canonical_points_raw.keys()))
    marker_key_to_idx_map = {key: i for i, key in enumerate(ordered_marker_keys)}
    num_markers = len(ordered_marker_keys)
    DEFORMATION_CACHE["zero_marker_displacements"] = np.zeros((num_markers, 3), dtype=np.float32)

    # --- 4. Pre-compute Marker A-pose Displacements for ALL Frames ---
    print("Pre-computing all marker displacements for entire animation...")
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

    DEFORMATION_CACHE["marker_displacements_unposed"] = all_frames_marker_offsets
    print("Finished pre-computing marker displacements.")

    # --- 5. Build the Sparse Interpolation Matrix ---
    print("Building sparse interpolation matrix...")
    num_skin_verts = len(source_skin_obj.data.vertices)
    rows, cols, data = [], [], []
    for skin_v_idx_str, interp_data in interpolation_weights.items():
        skin_v_idx = int(skin_v_idx_str)
        if skin_v_idx < num_skin_verts:
            for i, marker_key in enumerate(interp_data["influencing_markers"]):
                marker_idx = marker_key_to_idx_map.get(marker_key)
                if marker_idx is not None:
                    rows.append(skin_v_idx)
                    cols.append(marker_idx)
                    data.append(interp_data["interpolation_weights"][i])

    W_interp = csc_matrix((data, (rows, cols)), shape=(num_skin_verts, num_markers))
    DEFORMATION_CACHE["interpolation_matrix"] = W_interp
    print("Sparse interpolation matrix created.")

    # --- 6. Create Visualization Object ---
    if DEFORMED_SKIN_OBJ_NAME in bpy.data.objects: bpy.data.objects.remove(bpy.data.objects[DEFORMED_SKIN_OBJ_NAME],
                                                                           do_unlink=True)
    deformed_obj = source_skin_obj.copy()
    deformed_obj.data = source_skin_obj.data.copy()
    deformed_obj.name = DEFORMED_SKIN_OBJ_NAME

    p_unposed_skin_verts = np.empty(num_skin_verts * 3, dtype=np.float32)
    deformed_obj.data.vertices.foreach_get("co", p_unposed_skin_verts)
    DEFORMATION_CACHE["p_unposed_skin_vertices"] = p_unposed_skin_verts.reshape(num_skin_verts, 3)

    # if COLLECTION_NAME not in bpy.data.collections: bpy.context.scene.collection.children.link(
    #     bpy.data.collections.new(COLLECTION_NAME))
    # bpy.data.collections[COLLECTION_NAME].objects.link(deformed_obj)

    if COLLECTION_NAME not in bpy.data.collections:
        vis_collection = bpy.data.collections.new(COLLECTION_NAME)

        # Put this collection under the parent collection if it exists
        if PARENT_COLLECTION_NAME in bpy.data.collections:
            parent_collection = bpy.data.collections[PARENT_COLLECTION_NAME]
            parent_collection.children.link(vis_collection)
        else:
            print(f"Parent collection '{PARENT_COLLECTION_NAME}' not found. Creating it.")
            parent_collection = bpy.data.collections.new(PARENT_COLLECTION_NAME)
            bpy.context.scene.collection.children.link(parent_collection)
            parent_collection.children.link(vis_collection)
    else:
        vis_collection = bpy.data.collections[COLLECTION_NAME]
    vis_collection.objects.link(deformed_obj)

    # source_skin_obj.hide_set(True)

    # --- 7. Setup Slider and Register Handler ---
    bpy.context.scene["skin_deform_exaggeration_scale"] = EXAGGERATION_SCALE_DEFAULT
    bpy.context.scene.id_properties_ui("skin_deform_exaggeration_scale").update(min=0.0, max=50.0, step=0.1,
                                                                                description="Scale factor for non-rigid skin deformation")

    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(on_frame_change_final_skin)

    on_frame_change_final_skin(bpy.context.scene)

    print("\n--- OPTIMIZED SETUP COMPLETE ---")
    print("Animation handler registered. Scrub the timeline to see the final skin mesh deform.")


if __name__ == "__main__":
    setup_final_skin_visualization()