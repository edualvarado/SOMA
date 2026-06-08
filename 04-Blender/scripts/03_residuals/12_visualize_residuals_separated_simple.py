"""
STATUS: Simplified - Visualizes residuals on the canonical shape.
Script: visualize_residuals_separated.py
Goal: Displays residuals on the canonical shape by adding observed displacements
      to the canonical marker positions in world space.
"""

import bpy
import json
import numpy as np
from mathutils import Vector, Matrix

# --- User Configuration ---
shot = "shot_001"  # Change this to your shot name

CANONICAL_MARKERS_JSON_PATH = "C:/Users/ealvarad/00-Local/02-Python/Blender/data/registration/canonical_model/canonical_data.json"
OBSERVED_DISPLACEMENTS_JSON_PATH = f"C:/Users/ealvarad/00-Local/02-Python/Blender/data/displacements/{shot}/residuals/observed_residuals_only_{shot}_world_lbs.json"

ARMATURE_OBJECT_NAME = f"root_{shot[-3:]}"
PARENT_COLLECTION_NAME = shot.capitalize()

# Visualization settings
VIS_CLOUD_OBJ_NAME = f"Observed_Residuals_World_{shot[-3:]}"
COLLECTION_NAME = f"Observed_Residuals_World_{shot[-3:]}"
SCALE_DEFAULT = 0.1
HIDE_LOCATION = (10000.0, 10000.0, 10000.0)  # Move unobserved vertices here
SPHERE_COLOR = (0.8, 0.1, 0.1, 1.0)  # Bright Red for residuals

# --- Global dictionary to hold all loaded data ---
VIS_DATA_CACHE = {}

def on_frame_change_residuals(scene):
    """
    This handler updates the point cloud to display residuals on the canonical shape.
    """
    obj = scene.objects.get(VIS_CLOUD_OBJ_NAME)
    if not obj or "observed_displacements" not in VIS_DATA_CACHE:
        return

    current_frame_str = str(scene.frame_current)
    displacements_for_frame = VIS_DATA_CACHE["observed_displacements"].get(current_frame_str, {})

    scale_factor = scene.get("residual_scale_separated", 1.0)
    num_verts = len(obj.data.vertices)
    new_coords_flat = np.empty(num_verts * 3, dtype=np.float32)

    # Loop through all vertices in our object. Each vertex corresponds to a unique marker key.
    for v_idx, marker_key in enumerate(VIS_DATA_CACHE["ordered_keys"]):
        # Get the canonical position in world space
        p_canonical_world = VIS_DATA_CACHE["canonical_points_world"][marker_key]

        # Check if this marker has a displacement value in the current frame's data
        d_world = displacements_for_frame.get(marker_key)

        if d_world:
            # --- Point was OBSERVED ---
            # Apply the residual displacement to the canonical position
            p_visualized = p_canonical_world + (Vector(d_world) * scale_factor)
            new_coords_flat[v_idx * 3 : v_idx * 3 + 3] = p_visualized[:]
        else:
            # --- Point was UNOBSERVED ---
            # Move the vertex far away to hide it.
            new_coords_flat[v_idx * 3 : v_idx * 3 + 3] = HIDE_LOCATION

    # Update all vertex positions at once
    obj.data.vertices.foreach_set("co", new_coords_flat)
    obj.data.update()

def setup_residuals_visualization():
    """
    Sets up the visualization for residuals on the canonical shape.
    """
    global VIS_DATA_CACHE
    print("--- Setting up Residuals Visualization ---")

    # --- 1. Load Data ---
    try:
        with open(CANONICAL_MARKERS_JSON_PATH, 'r') as f:
            canonical_points_raw = json.load(f).get("0", {})
        with open(OBSERVED_DISPLACEMENTS_JSON_PATH, 'r') as f:
            VIS_DATA_CACHE["observed_displacements"] = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load JSON files. Check paths. Error: {e}")
        return

    # --- 2. Prepare Data Cache ---
    armature_obj = bpy.data.objects.get(ARMATURE_OBJECT_NAME)
    if not armature_obj or armature_obj.type != 'ARMATURE':
        print(f"ERROR: Armature object '{ARMATURE_OBJECT_NAME}' not found.")
        return

    # Extract translation and scale from the armature's matrix_world
    armature_translation = armature_obj.matrix_world.to_translation()
    armature_scale = armature_obj.matrix_world.to_scale()

    # Transform canonical points into world space (ignoring rotation)
    VIS_DATA_CACHE["canonical_points_world"] = {
        key: Vector(val[0]) * armature_scale + armature_translation
        for key, val in canonical_points_raw.items()
    }

    # The vertices in our object will correspond to ALL markers that are EVER observed
    all_observed_keys = sorted(list({key for frame_data in VIS_DATA_CACHE["observed_displacements"].values() for key in frame_data.keys()}))
    VIS_DATA_CACHE["ordered_keys"] = all_observed_keys

    print("All data loaded and prepared.")

    # --- 3. Create Visualization Object and Material ---
    if VIS_CLOUD_OBJ_NAME in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[VIS_CLOUD_OBJ_NAME], do_unlink=True)
    mesh_data = bpy.data.meshes.new(VIS_CLOUD_OBJ_NAME + "_Mesh")
    vis_obj = bpy.data.objects.new(VIS_CLOUD_OBJ_NAME, mesh_data)

    # Initial positions are their canonical world-space positions
    initial_vertex_coords = [VIS_DATA_CACHE["canonical_points_world"][key][:] for key in VIS_DATA_CACHE["ordered_keys"]]
    mesh_data.from_pydata(initial_vertex_coords, [], [])

    # Create and assign a simple material
    mat = bpy.data.materials.new(name="ResidualsMarkerMat")
    mat.use_nodes = True
    mat.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value = SPHERE_COLOR
    mesh_data.materials.append(mat)
    mesh_data.update()

    # Add the object to the collection
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
    bpy.context.scene["residual_scale_separated"] = SCALE_DEFAULT
    bpy.context.scene.id_properties_ui("residual_scale_separated").update(min=0.001, max=1000.0, step=1.0, description="Scale factor for residuals")

    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(on_frame_change_residuals)

    on_frame_change_residuals(bpy.context.scene)

    print("\n--- SETUP COMPLETE ---")
    print("Animation handler registered. Scrub the timeline to see residuals on the canonical shape.")

# --- Run the main setup function ---
if __name__ == "__main__":
    setup_residuals_visualization()