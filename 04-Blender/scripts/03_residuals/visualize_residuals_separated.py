"""
STATUS: Complete - Creates a single animated point cloud that ONLY shows the deforming
Script: visualize_residuals_separated.py
Goal: Creates a single animated point cloud that ONLY shows the deforming
      markers that were observed in the studio recording for each frame.
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
MARKER_LBS_WEIGHTS_JSON_PATH = "C:/Users/ealvarad/00-Local/02-Python/Blender/data/weights/canonical_model/lbs_markers/markers_lbs_weights_exported.json"

# This script uses the LOCAL displacements file you calculated for OBSERVED markers
OBSERVED_DISPLACEMENTS_JSON_PATH = f"C:/Users/ealvarad/00-Local/02-Python/Blender/data/displacements/{shot}/residuals/observed_residuals_only_{shot}_world_lbs.json"

ARMATURE_OBJECT_NAME = f"root_{shot[-3:]}"

PARENT_COLLECTION_NAME = shot.capitalize()

# --- Visualization Settings ---
VIS_CLOUD_OBJ_NAME = f"Observed_Residuals_{shot[-3:]}"
COLLECTION_NAME = f"Observed_Residuals_{shot[-3:]}"
SCALE_DEFAULT = 0.001
HIDE_LOCATION = (10000.0, 10000.0, 10000.0) # Move unobserved vertices here
SPHERE_COLOR = (0.1, 0.8, 0.2, 1.0)  # Bright Green for observed markers
# ----------------------------------------------------

# --- Global dictionary to hold all loaded data ---
VIS_DATA_CACHE = {}

def on_frame_change_observed_only(scene):
    """
    This handler updates one point cloud, showing only observed markers.
    """
    obj = scene.objects.get(VIS_CLOUD_OBJ_NAME)
    if not obj or "observed_displacements" not in VIS_DATA_CACHE:
        return

    current_frame_str = str(scene.frame_current)
    displacements_for_frame = VIS_DATA_CACHE["observed_displacements"].get(current_frame_str, {})

    scale_factor = scene.get("deform_scale_separated", 1.0)
    posed_bone_rotations = {b.name: b.matrix.to_3x3() for b in bpy.data.objects[ARMATURE_OBJECT_NAME].pose.bones}

    num_verts = len(obj.data.vertices)
    new_coords_flat = np.empty(num_verts * 3, dtype=np.float32)

    # Loop through all vertices in our object. Each vertex corresponds to a unique marker key.
    for v_idx, marker_key in enumerate(VIS_DATA_CACHE["ordered_keys"]):

        # Check if this marker has a displacement value in the current frame's data
        d_local = displacements_for_frame.get(marker_key)

        if d_local:
            # --- Point was OBSERVED ---
            # Calculate its deformed position and show it.
            p_unposed = VIS_DATA_CACHE["canonical_points"][marker_key]
            primary_bone_idx = VIS_DATA_CACHE["primary_bone_map"].get(marker_key)

            if primary_bone_idx is not None:
                R_bind = VIS_DATA_CACHE["bind_pose_rotations"][primary_bone_idx]
                primary_bone_name = VIS_DATA_CACHE["bone_idx_to_name_map"][primary_bone_idx]
                R_pose = posed_bone_rotations.get(primary_bone_name)

                if R_pose:
                    d_unposed_offset = R_bind @ Vector(d_local)
                    p_visualized = p_unposed + (Vector(d_local) * scale_factor)
                    new_coords_flat[v_idx * 3 : v_idx * 3 + 3] = p_visualized[:]
                else: # Fallback if bone name not found
                    new_coords_flat[v_idx * 3 : v_idx * 3 + 3] = HIDE_LOCATION
            else: # Fallback if primary bone not found
                new_coords_flat[v_idx * 3 : v_idx * 3 + 3] = HIDE_LOCATION
        else:
            # --- Point was UNOBSERVED ---
            # Move the vertex far away to hide it.
            new_coords_flat[v_idx * 3 : v_idx * 3 + 3] = HIDE_LOCATION

    # Update all vertex positions at once
    obj.data.vertices.foreach_set("co", new_coords_flat)
    obj.data.update()

def setup_observed_only_visualization():
    """
    Sets up the visualization for only the observed markers.
    """
    global VIS_DATA_CACHE
    print("--- Setting up Observed Marker Deformation Visualization ---")

    # --- 1. Load Data ---
    try:
        with open(CANONICAL_MARKERS_JSON_PATH, 'r') as f: canonical_points_raw = json.load(f).get("0", {})
        with open(MARKER_LBS_WEIGHTS_JSON_PATH, 'r') as f: marker_lbs_weights = json.load(f)
        with open(OBSERVED_DISPLACEMENTS_JSON_PATH, 'r') as f: VIS_DATA_CACHE["observed_displacements"] = json.load(f)
    except Exception as e: print(f"ERROR: Failed to load JSON files. Check paths. Error: {e}"); return

    # --- 2. Prepare Data Cache ---
    armature_obj = bpy.data.objects.get(ARMATURE_OBJECT_NAME)
    if not armature_obj or armature_obj.type != 'ARMATURE': print(f"ERROR: Armature object '{ARMATURE_OBJECT_NAME}' not found."); return
    VIS_DATA_CACHE["bind_pose_rotations"] = {idx: (armature_obj.matrix_world @ b.matrix_local).to_3x3() for idx,b in enumerate(armature_obj.data.bones)}
    VIS_DATA_CACHE["bone_idx_to_name_map"] = {idx: b.name for idx, b in enumerate(armature_obj.data.bones)}
    VIS_DATA_CACHE["canonical_points"] = {key: Vector(val[0]) for key, val in canonical_points_raw.items()}

    # The vertices in our object will correspond to ALL markers that are EVER observed
    all_observed_keys = sorted(list({key for frame_data in VIS_DATA_CACHE["observed_displacements"].values() for key in frame_data.keys()}))
    VIS_DATA_CACHE["ordered_keys"] = all_observed_keys

    VIS_DATA_CACHE["primary_bone_map"] = {key: wd["bone_indices"][np.argmax(wd["weights"])] for key, wd in marker_lbs_weights.items() if wd and wd.get("bone_indices")}
    print("All data loaded and prepared.")

    # --- 3. Create Visualization Object and Material ---
    if VIS_CLOUD_OBJ_NAME in bpy.data.objects: bpy.data.objects.remove(bpy.data.objects[VIS_CLOUD_OBJ_NAME], do_unlink=True)
    mesh_data = bpy.data.meshes.new(VIS_CLOUD_OBJ_NAME + "_Mesh")
    vis_obj = bpy.data.objects.new(VIS_CLOUD_OBJ_NAME, mesh_data)

    # Initial positions are their canonical A-pose positions
    initial_vertex_coords = [VIS_DATA_CACHE["canonical_points"][key][:] for key in VIS_DATA_CACHE["ordered_keys"]]
    mesh_data.from_pydata(initial_vertex_coords, [], [])

    # Create and assign a simple material
    mat = bpy.data.materials.new(name="ObservedMarkerMat")
    mat.use_nodes = True
    mat.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value = SPHERE_COLOR
    mesh_data.materials.append(mat)
    mesh_data.update()

    # if COLLECTION_NAME not in bpy.data.collections: bpy.context.scene.collection.children.link(bpy.data.collections.new(COLLECTION_NAME))
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
    bpy.context.scene["deform_scale_separated"] = SCALE_DEFAULT
    bpy.context.scene.id_properties_ui("deform_scale_separated").update(min=0.001, max=1000.0, step=1.0, description="Scale factor for observed displacements")

    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(on_frame_change_observed_only)

    on_frame_change_observed_only(bpy.context.scene)

    print("\n--- SETUP COMPLETE ---")
    print("Animation handler registered. Scrub the timeline to see ONLY the observed markers deform.")


# --- Run the main setup function ---
if __name__ == "__main__":
    # Add numpy import if it's not at the top level
    import numpy as np
    setup_observed_only_visualization()