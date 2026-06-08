"""
Script: visualize_markers.py
Goal: Visualize 3D marker positions as spheres from JSON in Blender.
"""

import bpy
import json
import os  # Though os might not be strictly needed here

# --- User Configuration ---
# !!! IMPORTANT: SET THIS TO THE CORRECT PATH FOR YOUR JSON FILE !!!

# JSON_FILE_PATH = "C:/Users/Eduardo/00-Local/Blender/SOMA/S1/canonical_model/canonical_data_tpose.json"
JSON_FILE_PATH = "S:/work/03-MUSK/02-Canonical-Model/S2/uv_detections_charuco-suit/registration/canonical_model/canonical_data.json"

# Visual properties for the spheres
SPHERE_RADIUS = 0.005  # Adjust based on your model's scale (e.g., 0.5 if units are cm)
SPHERE_SEGMENTS = 12  # Number of segments for the sphere (lower for better performance)
SPHERE_RINGS = 6  # Number of rings for the sphere

# Naming for created objects

# COLLECTION_NAME = "Static_Canonical_Marker_Spheres_S2_TPose"
COLLECTION_NAME = "Static_Canonical_Marker_Spheres_S2"

BASE_SPHERE_MESH_NAME = "_BaseMarkerSphereMeshTPose"  # Mesh data will be shared
POINT_OBJECT_PREFIX = "MarkerVisTPose_"

# [NEW] Material properties for the spheres
MATERIAL_NAME = "MarkerSphereMaterial"
MARKER_COLOR = (1.0, 0.2, 0.0, 1.0)  # RGBA (Red, Green, Blue, Alpha) - Changed to Orange

# --------------------------

def create_spheres_from_json(json_path, radius, segments, rings, collection_name, material_name, marker_color):
    """
    Loads 3D points from a JSON file and creates sphere instances at those locations.
    """

    # --- 1. Load JSON Data ---
    try:
        with open(json_path, 'r') as f:
            all_frames_data = json.load(f)
    except FileNotFoundError:
        message = f"ERROR: JSON file not found at: {json_path}"
        print(message)
        # For a more visible error in Blender's UI, you can use an operator popup:
        # self.report({'ERROR'}, message) # If run from an operator
        # For a script run directly, print is the main feedback to system console
        return
    except json.JSONDecodeError:
        message = f"ERROR: Could not decode JSON from: {json_path}"
        print(message)
        return
    except Exception as e:
        message = f"ERROR: An unexpected error occurred loading JSON: {e}"
        print(message)
        return

    # Assuming data is under frame key "0" for the static pose
    static_pose_data = all_frames_data.get("0", {})
    if not static_pose_data:
        message = f"ERROR: No data found for frame '0' in {json_path}"
        print(message)
        return

    marker_points = []
    point_names = []
    for key, coord_list_of_list in static_pose_data.items():
        try:
            # The coordinate is wrapped in an extra list: [[X,Y,Z]]
            coord = coord_list_of_list[0]
            if len(coord) == 3:
                marker_points.append(tuple(coord))  # Ensure it's a tuple of floats
                point_names.append(key)
            else:
                print(f"Warning: Coordinate for '{key}' is not 3D: {coord}. Skipping.")
        except (TypeError, IndexError) as e:
            print(f"Warning: Malformed coordinate data for key '{key}': {coord_list_of_list}. Error: {e}. Skipping.")

    if not marker_points:
        message = "ERROR: No valid 3D points extracted from the JSON data."
        print(message)
        return

    print(f"Loaded {len(marker_points)} marker points from JSON.")

    # --- 2. Create or Get Target Collection ---
    if collection_name in bpy.data.collections:
        vis_collection = bpy.data.collections[collection_name]
    else:
        vis_collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(vis_collection)
    print(f"Using collection: '{vis_collection.name}'")

    # --- 3. Create a Single Base Sphere Mesh (for instancing) ---
    if BASE_SPHERE_MESH_NAME in bpy.data.meshes:
        # If re-running, remove old mesh data to avoid issues
        bpy.data.meshes.remove(bpy.data.meshes[BASE_SPHERE_MESH_NAME])

    base_sphere_mesh = bpy.data.meshes.new(BASE_SPHERE_MESH_NAME)

    import bmesh  # bmesh is generally preferred for mesh creation/editing
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=rings, radius=radius)
    bm.to_mesh(base_sphere_mesh)
    bm.free()
    base_sphere_mesh.update()

    # [NEW] Create or get material and assign it to the base mesh
    if material_name in bpy.data.materials:
        marker_material = bpy.data.materials[material_name]
    else:
        marker_material = bpy.data.materials.new(name=material_name)
    
    marker_material.diffuse_color = marker_color # Set the base color (RGBA)
    marker_material.use_nodes = False # Simple diffuse color, no complex nodes needed

    # Assign the material to the base mesh
    if base_sphere_mesh.materials:
        base_sphere_mesh.materials[0] = marker_material
    else:
        base_sphere_mesh.materials.append(marker_material)

    # --- 4. Create Sphere Objects (Linked Duplicates/Instances of the Base Mesh) ---
    created_count = 0
    # Deselect all objects first
    bpy.ops.object.select_all(action='DESELECT')

    for i, point_coord in enumerate(marker_points):
        point_name_from_json = point_names[i]
        obj_name = f"{POINT_OBJECT_PREFIX}{point_name_from_json}"

        # Create new object using the base sphere mesh data
        sphere_obj = bpy.data.objects.new(obj_name, base_sphere_mesh)
        sphere_obj.location = point_coord  # Set its 3D location

        # Link new object to the target collection
        vis_collection.objects.link(sphere_obj)
        sphere_obj.select_set(True)  # Select the new object
        created_count += 1

    if created_count > 0:
        bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]  # Make one active

    print(f"Created {created_count} sphere instances in collection '{collection_name}'.")

    print("\n--- Visualization Script Finished ---")
    print("Import your suit mesh (OBJ/FBX) into the scene if you haven't already.")
    print(f"Look for a collection named '{collection_name}' containing the spheres.")
    print("The spheres represent your canonical marker corner positions.")


# --- Run the main function ---
if __name__ == "__main__":
    # Optional: Clean up previously generated spheres from this script before running
    # This makes it easier to re-run without clutter.
    # Be careful if you have other objects with this prefix that you want to keep.
    # if bpy.data.collections.get(COLLECTION_NAME):
    #     coll_to_delete = bpy.data.collections[COLLECTION_NAME]
    #     for obj in list(coll_to_delete.objects): # Iterate over a copy
    #         bpy.data.objects.remove(obj, do_unlink=True)
    #     bpy.data.collections.remove(coll_to_delete)

    create_spheres_from_json(
        JSON_FILE_PATH,
        SPHERE_RADIUS,
        SPHERE_SEGMENTS,
        SPHERE_RINGS,
        COLLECTION_NAME,
        MATERIAL_NAME, # [NEW] Pass material name
        MARKER_COLOR   # [NEW] Pass marker color
    )