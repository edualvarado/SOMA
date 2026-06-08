"""
Script: visualize_dense_deformation.py
Goal: Visualizes the final, dense, and smooth non-rigid deformation by applying
      the pre-calculated dense local displacements to the canonical A-pose point cloud.
"""

import bpy
import json
import numpy as np
from mathutils import Vector, Matrix

# --- User Configuration ---

# ---
shot = "shot_004"  # Change this to your shot name
# ---

CANONICAL_MARKERS_JSON_PATH = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/registration/canonical_model/canonical_data.json"
MARKER_LBS_WEIGHTS_JSON_PATH = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/weights/canonical_model/marker_lbs_weights.json"

# Option 1: Single-mesh Displacements
# DENSE_DISPLACEMENTS_JSON_PATH = f"C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/displacements/{shot}/reconstruction/dense_local_displacements_{shot}.json"
# VIS_CLOUD_OBJ_NAME = f"Dense_Deformation_{shot[-3:]}"
# COLLECTION_NAME = f"Dense_Deformation_{shot[-3:]}"

# Option 2: Muscle-constrained Displacements
# DENSE_DISPLACEMENTS_JSON_PATH = f"C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/displacements/{shot}/reconstruction/dense_muscle_constrained_displacements_{shot}.json"
# VIS_CLOUD_OBJ_NAME = f"Dense_Deformation_Muscle_{shot[-3:]}"
# COLLECTION_NAME = f"Dense_Deformation_Muscle_{shot[-3:]}"

# Option 3: Single-mesh + Muscle-constrained Displacements
DENSE_DISPLACEMENTS_JSON_PATH = f"C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/displacements/{shot}/reconstruction/refined_two_pass_displacements_{shot}.json"
VIS_CLOUD_OBJ_NAME = f"Dense_Deformation_Two_Passes_{shot[-3:]}"
COLLECTION_NAME = f"Dense_Deformation_Two_Passes_{shot[-3:]}"

ARMATURE_OBJECT_NAME = f"root_{shot[-3:]}"

PARENT_COLLECTION_NAME = shot.capitalize()

# --- Visualization Settings ---
EXAGGERATION_SCALE_DEFAULT = 0.001  # Start with 1.0 for true deformation
SPHERE_COLOR = (0.8, 0.1, 0.1, 1.0)  # Red color for the final result

# ----------------------------------------------------

# --- Global dictionary to hold all loaded data ---
DEFORMATION_CACHE = {}


def on_frame_change_dense_vis(scene):
    """
    This handler updates the dense point cloud on each frame change.
    """
    # Use .get() for safe access to avoid errors if object is deleted
    obj = scene.objects.get(VIS_CLOUD_OBJ_NAME)
    # Check if our cache has been populated by the setup function
    if not obj or "dense_displacements" not in DEFORMATION_CACHE:
        return

    current_frame_str = str(scene.frame_current)
    displacements_for_frame = DEFORMATION_CACHE["dense_displacements"].get(current_frame_str)

    scale_factor = scene.get("dense_deform_exaggeration_scale", 1.0)
    posed_bone_rotations = {b.name: b.matrix.to_3x3() for b in bpy.data.objects[ARMATURE_OBJECT_NAME].pose.bones}

    num_verts = len(obj.data.vertices)
    new_coords_flat = np.empty(num_verts * 3, dtype=np.float32)

    # If there's no displacement data for this frame, reset to the base A-pose
    if displacements_for_frame is None:
        for v_idx, marker_key in enumerate(DEFORMATION_CACHE["ordered_keys"]):
            p_unposed = DEFORMATION_CACHE["canonical_points"][marker_key]
            new_coords_flat[v_idx * 3: v_idx * 3 + 3] = p_unposed[:]
        obj.data.vertices.foreach_set("co", new_coords_flat)
        obj.data.update()
        return

    # Loop through each vertex of our point cloud object
    for v_idx, marker_key in enumerate(DEFORMATION_CACHE["ordered_keys"]):
        p_unposed = DEFORMATION_CACHE["canonical_points"][marker_key]
        d_local = displacements_for_frame.get(marker_key)  # Use .get() for safety

        if d_local:
            primary_bone_idx = DEFORMATION_CACHE["primary_bone_map"].get(marker_key)
            if primary_bone_idx is not None:
                R_bind = DEFORMATION_CACHE["bind_pose_rotations"][primary_bone_idx]
                primary_bone_name = DEFORMATION_CACHE["bone_idx_to_name_map"][primary_bone_idx]
                R_pose = posed_bone_rotations.get(primary_bone_name)

                if R_pose:
                    d_unposed_offset = R_bind @ Vector(d_local)
                    p_visualized = p_unposed + (d_unposed_offset * scale_factor)
                    new_coords_flat[v_idx * 3: v_idx * 3 + 3] = p_visualized[:]
                else:  # Fallback if bone name not found
                    new_coords_flat[v_idx * 3: v_idx * 3 + 3] = p_unposed[:]
            else:  # Fallback if primary bone not found
                new_coords_flat[v_idx * 3: v_idx * 3 + 3] = p_unposed[:]
        else:  # If no displacement data for this specific marker in this frame
            new_coords_flat[v_idx * 3: v_idx * 3 + 3] = p_unposed[:]

    # Update all vertex positions at once
    obj.data.vertices.foreach_set("co", new_coords_flat)
    obj.data.update()


def setup_dense_visualization():
    """
    Sets up the scene to visualize the final dense deformation.
    """
    global DEFORMATION_CACHE  # Declare that we intend to modify the global dictionary
    print("--- Setting up Dense Deformation Visualization ---")

    # --- 1. Load All Data Files ---
    try:
        with open(CANONICAL_MARKERS_JSON_PATH, 'r') as f:
            canonical_points_raw = json.load(f).get("0", {})
        with open(MARKER_LBS_WEIGHTS_JSON_PATH, 'r') as f:
            marker_lbs_weights = json.load(f)
        with open(DENSE_DISPLACEMENTS_JSON_PATH, 'r') as f:
            DEFORMATION_CACHE["dense_displacements"] = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load JSON files: {e}"); return

    # --- 2. Prepare Data Cache ---
    armature_obj = bpy.data.objects.get(ARMATURE_OBJECT_NAME)
    if not armature_obj or armature_obj.type != 'ARMATURE': print(
        f"ERROR: Armature object '{ARMATURE_OBJECT_NAME}' not found."); return
    DEFORMATION_CACHE["bind_pose_rotations"] = {idx: (armature_obj.matrix_world @ b.matrix_local).to_3x3() for idx, b in
                                                enumerate(armature_obj.data.bones)}
    DEFORMATION_CACHE["bone_idx_to_name_map"] = {idx: b.name for idx, b in enumerate(armature_obj.data.bones)}
    DEFORMATION_CACHE["canonical_points"] = {key: Vector(val[0]) for key, val in canonical_points_raw.items()}
    # Ensure ordered_keys uses the canonical points so the index matches vertices
    DEFORMATION_CACHE["ordered_keys"] = sorted(list(DEFORMATION_CACHE["canonical_points"].keys()))
    DEFORMATION_CACHE["primary_bone_map"] = {key: wd["bone_indices"][np.argmax(wd["weights"])] for key, wd in
                                             marker_lbs_weights.items() if wd and wd.get("bone_indices")}
    print("All data loaded and prepared.")

    # --- 3. Create Visualization Object and Material ---
    if VIS_CLOUD_OBJ_NAME in bpy.data.objects: bpy.data.objects.remove(bpy.data.objects[VIS_CLOUD_OBJ_NAME],
                                                                       do_unlink=True)
    mesh_data = bpy.data.meshes.new(VIS_CLOUD_OBJ_NAME + "_Mesh")
    vis_obj = bpy.data.objects.new(VIS_CLOUD_OBJ_NAME, mesh_data)
    vertex_coords = [DEFORMATION_CACHE["canonical_points"][key][:] for key in DEFORMATION_CACHE["ordered_keys"]]
    mesh_data.from_pydata(vertex_coords, [], [])

    mat = bpy.data.materials.new(name="DenseDeformMat")
    mat.use_nodes = True
    mat.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value = SPHERE_COLOR
    mesh_data.materials.append(mat)
    mesh_data.update()

    # if COLLECTION_NAME not in bpy.data.collections:
    #     bpy.context.scene.collection.children.link(bpy.data.collections.new(COLLECTION_NAME))
    # bpy.data.collections[COLLECTION_NAME].objects.link(vis_obj)

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
    vis_collection.objects.link(vis_obj)

    vis_obj.matrix_world = Matrix.Identity(4)
    print(f"Created visualization object '{vis_obj.name}'.")

    # --- 4. Setup Slider and Register Handler ---
    bpy.context.scene["dense_deform_exaggeration_scale"] = EXAGGERATION_SCALE_DEFAULT
    bpy.context.scene.id_properties_ui("dense_deform_exaggeration_scale").update(min=0.0, max=50.0, step=0.1,
                                                                                 description="Scale factor for dense non-rigid displacements")

    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(on_frame_change_dense_vis)

    on_frame_change_dense_vis(bpy.context.scene)

    print("\n--- SETUP COMPLETE ---")
    print("Animation handler registered. Scrub the timeline to see the final, dense deformation.")
    print("To see colors, switch to 'Material Preview' viewport shading mode.")


# --- Run the main setup function ---
if __name__ == "__main__":
    setup_dense_visualization()