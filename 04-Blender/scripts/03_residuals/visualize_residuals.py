"""
STATUS: Complete - Visualizes the true non-rigid deformation captured from studio data.
Script: visualize_residuals.py
Goal: Visualizes the true non-rigid deformation captured from studio data.
      It applies the calculated difference vectors (Observed - LBS) as a
      local displacement to the canonical A-pose marker point cloud.
"""

import bpy
import json
import numpy as np
from mathutils import Vector, Matrix

# --- User Configuration ---

# ---
shot = "shot_001"  # Change this to your shot name
# ---

CANONICAL_MARKERS_JSON_PATH = "C:/Users/ealvarad/00-Local/02-Python/Blender/data/registration/canonical_model/canonical_data.json"
MARKER_LBS_WEIGHTS_JSON_PATH = "C:/Users/ealvarad/00-Local/02-Python/Blender/data/weights/canonical_model/lbs_markers/exported_marker_lbs_weights.json"

# This is the NEW input file you just created
OBSERVED_DIFFERENCE_JSON_PATH = f"C:/Users/ealvarad/00-Local/02-Python/Blender/data/displacements/{shot}/residuals/residuals_{shot}_world_scaled.json"

ARMATURE_OBJECT_NAME = f"root_{shot[-3:]}"

# Naming for the new visualization object
VIS_CLOUD_OBJ_NAME = f"All_Residuals_{shot[-3:]}"
COLLECTION_NAME = f"All_Residuals_{shot[-3:]}"

PARENT_COLLECTION_NAME = shot.capitalize()

# Exaggeration Factor for Visualization
EXAGGERATION_SCALE_DEFAULT = 0.001  # Increase this to make subtle deformations more visible
# ----------------------------------------------------

# --- Global dictionary to hold all loaded data ---
VIS_DATA_CACHE = {
    "world_differences": None,
    "canonical_points": None,
    "primary_bone_map": None,
    "bind_pose_rotations": None,
    "ordered_keys": None
}


def on_frame_change_observed_deform(scene):
    """
    This handler updates the visualization based on the observed difference vectors.
    """
    obj = scene.objects.get(VIS_CLOUD_OBJ_NAME)
    if not obj or not VIS_DATA_CACHE.get("world_differences"):
        return

    current_frame_str = str(scene.frame_current)
    diffs_for_frame = VIS_DATA_CACHE["world_differences"].get(current_frame_str, {})

    scale_factor = scene.get("deform_scale", 1.0)

    posed_bone_rotations = {b.name: b.matrix.to_3x3() for b in bpy.data.objects[ARMATURE_OBJECT_NAME].pose.bones}

    num_verts = len(obj.data.vertices)
    new_coords_flat = np.empty(num_verts * 3, dtype=np.float32)

    for v_idx in range(num_verts):
        marker_key = VIS_DATA_CACHE["ordered_keys"][v_idx]
        p_unposed = VIS_DATA_CACHE["canonical_points"][marker_key]

        # Get the pre-calculated WORLD-SPACE difference vector for this frame
        d_world = diffs_for_frame.get(marker_key)

        if d_world:
            primary_bone_idx = VIS_DATA_CACHE["primary_bone_map"].get(marker_key)
            if primary_bone_idx is not None:
                R_bind = VIS_DATA_CACHE["bind_pose_rotations"][primary_bone_idx]

                # Get the posed rotation for the primary bone
                primary_bone_name = VIS_DATA_CACHE["bone_idx_to_name_map"][primary_bone_idx]
                R_pose = posed_bone_rotations.get(primary_bone_name)

                if R_pose:
                    # Transform world-space residual to local-space
                    d_local = R_pose.transposed() @ Vector(d_world)
                    # Transform local-space displacement to canonical A-pose world space
                    d_unposed_offset = R_bind @ d_local

                    # The final visualized position with exaggeration
                    p_visualized = p_unposed + (d_unposed_offset * scale_factor)
                    new_coords_flat[v_idx * 3: v_idx * 3 + 3] = p_visualized[:]
                else:  # Fallback if bone name not found in pose bones
                    new_coords_flat[v_idx * 3: v_idx * 3 + 3] = p_unposed[:]
            else:  # Fallback if primary bone not found
                new_coords_flat[v_idx * 3: v_idx * 3 + 3] = p_unposed[:]
        else:
            # If no displacement data for this point in this frame, keep it at its A-pose position
            new_coords_flat[v_idx * 3: v_idx * 3 + 3] = p_unposed[:]

    obj.data.vertices.foreach_set("co", new_coords_flat)
    obj.data.update()


def setup_observed_deformation_visualization():
    """
    One-time setup function to load all data, create the point cloud object,
    and register the animation handler.
    """
    global VIS_DATA_CACHE
    print("--- Setting up Observed Deformation Visualization ---")

    # --- 1. Load All Data Files ---
    print("Loading data files...")
    try:
        with open(CANONICAL_MARKERS_JSON_PATH, 'r') as f:
            canonical_points_raw = json.load(f).get("0", {})
        with open(MARKER_LBS_WEIGHTS_JSON_PATH, 'r') as f:
            marker_lbs_weights = json.load(f)
        with open(OBSERVED_DIFFERENCE_JSON_PATH, 'r') as f:
            VIS_DATA_CACHE["world_differences"] = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load JSON files. Check paths. Error: {e}");
        return

    # --- 2. Get Armature and Prepare Data Cache ---
    armature_obj = bpy.data.objects.get(ARMATURE_OBJECT_NAME)
    if not armature_obj or armature_obj.type != 'ARMATURE':
        print(f"ERROR: Armature object '{ARMATURE_OBJECT_NAME}' not found.");
        return

    VIS_DATA_CACHE["bind_pose_rotations"] = {idx: (armature_obj.matrix_world @ b.matrix_local).to_3x3() for idx, b in
                                             enumerate(armature_obj.data.bones)}
    VIS_DATA_CACHE["bone_idx_to_name_map"] = {idx: b.name for idx, b in enumerate(armature_obj.data.bones)}
    VIS_DATA_CACHE["canonical_points"] = {key: Vector(val[0]) for key, val in canonical_points_raw.items()}
    VIS_DATA_CACHE["ordered_keys"] = list(VIS_DATA_CACHE["canonical_points"].keys())
    VIS_DATA_CACHE["primary_bone_map"] = {key: wd["bone_indices"][np.argmax(wd["weights"])] for key, wd in
                                          marker_lbs_weights.items() if wd and wd.get("bone_indices")}

    print("All data loaded and prepared in cache.")

    # --- 3. Create the Visualization Point Cloud Object ---
    if VIS_CLOUD_OBJ_NAME in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[VIS_CLOUD_OBJ_NAME], do_unlink=True)

    mesh_data = bpy.data.meshes.new(VIS_CLOUD_OBJ_NAME + "_Mesh")
    vis_obj = bpy.data.objects.new(VIS_CLOUD_OBJ_NAME, mesh_data)

    vertex_coords = [VIS_DATA_CACHE["canonical_points"][key][:] for key in VIS_DATA_CACHE["ordered_keys"]]
    mesh_data.from_pydata(vertex_coords, [], [])
    mesh_data.update()

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

    vis_obj.matrix_world = Matrix.Identity(4)  # Place at origin with no rotation
    print(f"Created visualization object '{vis_obj.name}'.")

    # --- 4. Setup Slider and Register Handler ---
    bpy.context.scene["deform_scale"] = EXAGGERATION_SCALE_DEFAULT
    bpy.context.scene.id_properties_ui("deform_scale").update(min=0.001, max=1000.0, step=1.0,
                                                                             description="Scale factor for observed non-rigid displacements")

    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(on_frame_change_observed_deform)

    on_frame_change_observed_deform(bpy.context.scene)  # Initial update

    print("\n--- SETUP COMPLETE ---")
    print("Animation handler registered. Scrub the timeline to see the observed deformation.")
    print("Find the 'deform_scale' slider in Scene Properties -> Custom Properties.")


# --- Run the main setup function ---
if __name__ == "__main__":
    setup_observed_deformation_visualization()