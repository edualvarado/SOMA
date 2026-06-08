"""
STATUS: New - Plots the exported canonical data in Blender.
Script: plot_canonical_data.py
Goal: Loads the exported canonical data JSON file and visualizes the markers as a point cloud in Blender.
"""

import bpy
import json
from mathutils import Vector
from pathlib import Path

# --- User Configuration ---
shot = "FBX-NewSkeleton-TPose"  # Change this to your shot name

# CANONICAL_DATA_JSON_PATH = f"S:/work/03-MUSK/05-Training/debug/data/canonical_model/canonical_data.json"
CANONICAL_DATA_JSON_PATH = f"S:/work/03-MUSK/04-Blender/data/registration/canonical_model/canonical_data_tpose_new.json"

VIS_CLOUD_OBJ_NAME = f"Canonical_Data_Visualization_{shot}"
COLLECTION_NAME = f"Canonical_Data_Visualization_{shot}"
SPHERE_COLOR = (0.1, 0.5, 0.8, 1.0)  # Light Blue for visualization

# ----------------------------------------------------

def plot_canonical_data():
    """
    Plots the canonical data from the JSON file as a point cloud in Blender.
    """
    print(f"--- Plotting Canonical Data from: {CANONICAL_DATA_JSON_PATH} ---")

    # --- 1. Load Canonical Data ---
    canonical_data_path = Path(CANONICAL_DATA_JSON_PATH)
    if not canonical_data_path.exists():
        print(f"ERROR: Canonical data file not found at {CANONICAL_DATA_JSON_PATH}.")
        return

    try:
        with open(canonical_data_path, 'r') as f:
            canonical_data = json.load(f).get("0", {})
    except Exception as e:
        print(f"ERROR: Failed to load canonical data from {CANONICAL_DATA_JSON_PATH}. Error: {e}")
        return

    print(f"Loaded {len(canonical_data)} markers from the canonical data.")

    # --- 2. Prepare Marker Positions ---
    marker_positions = []
    for marker_key, coord_list in canonical_data.items():
        if len(coord_list) == 1 and len(coord_list[0]) == 3:
            marker_positions.append(Vector(coord_list[0]))
        else:
            print(f"WARNING: Marker {marker_key} has invalid coordinates: {coord_list}")

    print(f"Prepared {len(marker_positions)} valid marker positions for visualization.")

    # --- 3. Create Visualization Object ---
    if VIS_CLOUD_OBJ_NAME in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[VIS_CLOUD_OBJ_NAME], do_unlink=True)
    mesh_data = bpy.data.meshes.new(VIS_CLOUD_OBJ_NAME + "_Mesh")
    vis_obj = bpy.data.objects.new(VIS_CLOUD_OBJ_NAME, mesh_data)

    # Add the object to the collection
    if COLLECTION_NAME not in bpy.data.collections:
        vis_collection = bpy.data.collections.new(COLLECTION_NAME)
        bpy.context.scene.collection.children.link(vis_collection)
    else:
        vis_collection = bpy.data.collections[COLLECTION_NAME]
    vis_collection.objects.link(vis_obj)

    # Create the point cloud from the marker positions
    mesh_data.from_pydata(marker_positions, [], [])
    mesh_data.update()

    # --- 4. Create and Assign Material ---
    mat = bpy.data.materials.new(name="CanonicalDataMaterial")
    mat.use_nodes = True
    mat.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value = SPHERE_COLOR
    mesh_data.materials.append(mat)

    vis_obj.matrix_world = Matrix.Identity(4)
    print(f"Created visualization object '{vis_obj.name}' with {len(marker_positions)} markers.")

    print("\n--- Visualization Complete ---")
    print("Check the Blender viewport to verify the marker positions.")

# --- Run the Plot Function ---
if __name__ == "__main__":
    plot_canonical_data()