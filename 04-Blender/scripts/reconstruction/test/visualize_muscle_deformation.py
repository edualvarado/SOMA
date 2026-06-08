"""
Script: visualize_muscle_deformation.py
Goal: Deforms a collection of individual muscle meshes in real-time based on
      dense marker displacements and pre-computed, muscle-specific interpolation weights.

NOT READY YET
"""

import bpy
import json
import numpy as np
from mathutils import Vector, Matrix
# Ensure you have scipy installed in Blender's Python environment
from scipy.sparse import csc_matrix

# --- User Configuration ---

# ---
shot = "shot_002"  # Change this to your shot name
# ---

# INPUT
MUSCLE_COLLECTION_NAME = f"canonical_muscle_complex_{shot[-3:]}" # Collection containing ALL muscle objects
ARMATURE_OBJECT_NAME = f"root_{shot[-3:]}"

# --- JSON File Paths ---
CANONICAL_MARKERS_JSON_PATH = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/registration/canonical_model/canonical_data.json"
MARKER_LBS_WEIGHTS_JSON_PATH = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/weights/canonical_model/marker_lbs_weights.json"

DENSE_DISPLACEMENTS_JSON_PATH = f"C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/displacements/{shot}/refined_two_pass_displacements_{shot}.json"

MUSCLE_INTERPOLATION_WEIGHTS_JSON = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/weights/canonical_model/muscle_interpolation_weights.json"

# --- Visualization Settings ---
DEFORMED_COLLECTION_NAME = "Final_Deformed_Muscles"
EXAGGERATION_SCALE_DEFAULT = 1.0 # Start with 1.0 for true deformation
DATA_SCALE_FACTOR = 0.001 # To convert mm data to meter scene
# ----------------------------------------------------

# --- Global dictionary to hold all pre-computed data ---
DEFORMATION_CACHE = {}

def on_frame_change_muscle_deform(scene):
    """
    This handler deforms the collection of muscle meshes using fast, pre-computed data.
    """
    if "marker_displacements_unposed" not in DEFORMATION_CACHE: return

    current_frame_str = str(scene.frame_current)
    scale_factor = scene.get("muscle_deform_exaggeration_scale", 1.0)

    # 1. Get pre-computed marker displacements for this frame
    D_marker_t = DEFORMATION_CACHE["marker_displacements_unposed"].get(
        current_frame_str, DEFORMATION_CACHE["zero_marker_displacements"]
    )

    # Combine unit conversion and user exaggeration into one scale factor
    total_scale = DATA_SCALE_FACTOR * scale_factor
    D_marker_scaled_t = D_marker_t * total_scale

    # 2. Loop through each muscle and apply its unique deformation
    for muscle_name, deform_data in DEFORMATION_CACHE["deformable_muscles"].items():
        deformed_obj = deform_data["object"]
        W_interp = deform_data["interp_matrix"]
        P_unposed = deform_data["p_unposed_verts"]

        # 3. Interpolate displacements for this muscle with one sparse matrix multiplication
        D_muscle_t = W_interp @ D_marker_scaled_t

        # 4. Calculate final vertex positions
        P_muscle_final_t = P_unposed + D_muscle_t

        # 5. Update mesh efficiently
        deformed_obj.data.vertices.foreach_set("co", P_muscle_final_t.ravel())
        deformed_obj.data.update()

def setup_muscle_deformation_visualization():
    """
    Performs a one-time setup for all muscle meshes.
    """
    global DEFORMATION_CACHE
    print("--- Setting up Muscle Deformation Visualization ---")

    # --- 1. Load All Data Files ---
    try:
        with open(CANONICAL_MARKERS_JSON_PATH, 'r') as f: canonical_points_raw = json.load(f).get("0", {})
        with open(MARKER_LBS_WEIGHTS_JSON_PATH, 'r') as f: marker_lbs_weights = json.load(f)
        with open(DENSE_DISPLACEMENTS_JSON_PATH, 'r') as f: dense_displacements_by_frame = json.load(f)
        with open(MUSCLE_INTERPOLATION_WEIGHTS_JSON, 'r') as f: muscle_interpolation_weights = json.load(f)
    except Exception as e: print(f"ERROR: Failed to load JSON files: {e}"); return

    # --- 2. Get Scene Objects and Basic Data ---
    armature_obj = bpy.data.objects.get(ARMATURE_OBJECT_NAME)
    if not armature_obj: print(f"ERROR: Armature object '{ARMATURE_OBJECT_NAME}' not found."); return
    muscle_collection = bpy.data.collections.get(MUSCLE_COLLECTION_NAME)
    if not muscle_collection: print(f"ERROR: Muscle collection '{MUSCLE_COLLECTION_NAME}' not found."); return
    source_muscle_objects = [obj for obj in muscle_collection.objects if obj.type == 'MESH']
    if not source_muscle_objects: print(f"ERROR: No mesh objects in collection '{MUSCLE_COLLECTION_NAME}'."); return

    # --- 3. Pre-compute Bind Data and Mappings ---
    bind_pose_rotations = {idx: (armature_obj.matrix_world @ b.matrix_local).to_3x3() for idx,b in enumerate(armature_obj.data.bones)}
    primary_bone_map = {key: wd["bone_indices"][np.argmax(wd["weights"])] for key, wd in marker_lbs_weights.items() if wd and wd.get("bone_indices")}
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

    # --- 5. Create Duplicates and Build Interpolation Matrices for EACH Muscle ---
    print("Creating duplicate muscle objects and building interpolation matrices...")
    if DEFORMED_COLLECTION_NAME not in bpy.data.collections: bpy.context.scene.collection.children.link(bpy.data.collections.new(DEFORMED_COLLECTION_NAME))
    deformed_collection = bpy.data.collections[DEFORMED_COLLECTION_NAME]

    DEFORMATION_CACHE["deformable_muscles"] = {}

    for source_obj in source_muscle_objects:
        deformed_obj = source_obj.copy()
        deformed_obj.data = source_obj.data.copy()
        deformed_obj.name = source_obj.name + "_deformed"
        deformed_collection.objects.link(deformed_obj)

        num_muscle_verts = len(deformed_obj.data.vertices)

        # Build the sparse interpolation matrix W for this muscle
        rows, cols, data = [], [], []
        interp_weights_for_this_muscle = muscle_interpolation_weights.get(source_obj.name, {})
        for v_idx_str, interp_data in interp_weights_for_this_muscle.items():
            v_idx = int(v_idx_str)
            if v_idx < num_muscle_verts:
                for i, marker_key in enumerate(interp_data["influencing_markers"]):
                    marker_idx = marker_key_to_idx_map.get(marker_key)
                    if marker_idx is not None:
                        rows.append(v_idx)
                        cols.append(marker_idx)
                        data.append(interp_data["interpolation_weights"][i])

        W_interp = csc_matrix((data, (rows, cols)), shape=(num_muscle_verts, num_markers))

        # Cache all necessary data for this muscle
        p_unposed_verts = np.array([source_obj.matrix_world @ v.co for v in source_obj.data.vertices], dtype=np.float32)
        DEFORMATION_CACHE["deformable_muscles"][source_obj.name] = {
            "object": deformed_obj,
            "interp_matrix": W_interp,
            "p_unposed_verts": p_unposed_verts
        }

    muscle_collection.hide_viewport = True  # Hide original muscles
    print(f"Finished setup for {len(source_muscle_objects)} muscles.")

    # --- 6. Setup Slider and Register Handler ---
    bpy.context.scene["muscle_deform_exaggeration_scale"] = EXAGGERATION_SCALE_DEFAULT
    bpy.context.scene.id_properties_ui("muscle_deform_exaggeration_scale").update(min=0.0, max=200.0, step=0.5, description="Scale factor for non-rigid muscle deformation")

    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(on_frame_change_muscle_deform)

    on_frame_change_muscle_deform(bpy.context.scene)

    print("\n--- OPTIMIZED SETUP COMPLETE ---")
    print("Animation handler registered. Scrub the timeline to see the muscle meshes deform.")

if __name__ == "__main__":
    setup_muscle_deformation_visualization()