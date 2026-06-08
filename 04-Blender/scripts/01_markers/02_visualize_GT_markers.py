"""
STATUS: Completed - Visualize GT markers as a point cloud in Blender.
Script: visualize_GT_markers.py
Goal: Plot observed markers from JSON in Blender as a point cloud.
"""

import bpy
import json
from mathutils import Vector, Matrix
import math
import numpy as np
import os
import bmesh

# --- User Configuration ---

# ---
shot = "shot_001"  # Change this to your shot name
# ---

# Previous unfiltered version
# MOTION_DATA_JSON_PATH = f"S:/work/03-MUSK/04-Blender/data/registration/{shot}/triangulated_sequence_{shot}_transformed.json"

# Final filtered version
# MOTION_DATA_JSON_PATH = f"S:/work/03-MUSK/04-Blender/data/registration/{shot}/triangulated_sequence_{shot}_transformed_filtered.json"
MOTION_DATA_JSON_PATH = f"S:/work/03-MUSK/04-Blender/data/registration/S2/{shot}/S2_triangulated_sequence_{shot}_transformed.json" # <- already filtered

# Naming for the new visualization object
# MOTION_CLOUD_OBJ_NAME = f"Observed_GT_Markers_{shot[-3:]}"
# COLLECTION_NAME = f"Observed_GT_Markers_{shot[-3:]}"

MOTION_CLOUD_OBJ_NAME = f"Observed_GT_Markers_{shot[-3:]}_S2"
COLLECTION_NAME = f"Observed_GT_Markers_{shot[-3:]}_S2"

# PARENT_COLLECTION_NAME = shot.capitalize()
PARENT_COLLECTION_NAME =  "Shot_001_S2"

# What to do with vertices for markers not detected in a frame
# Option 1: Hide them by moving them far away
HIDE_LOCATION = (10000.0, 10000.0, 10000.0)
# Option 2: Keep their last known position (more complex, requires storing state)
# For simplicity, we will use the hide location method.
# ----------------------------------------------------

# --- Global dictionary to hold data so the handler can access it efficiently ---
MOTION_VIS_DATA = {
    "all_frames_data": None,
    "ordered_keys_map": None, # Will map marker_key -> vertex_index
    "initial_positions": None # To place points that are not in the first frame
}

# SPHERES

# def setup_motion_data_visualization():
#     """
#     One-time setup function to load all data, create the point cloud object,
#     configure sphere instancing, and register the animation handler.
#     """
#     global MOTION_VIS_DATA
#     import numpy as np

#     print("--- Setting up Motion Data Visualization ---")

#     # --- 1. Load the entire motion data JSON into memory ---
#     print(f"Loading motion data from: {MOTION_DATA_JSON_PATH}")
#     try:
#         with open(MOTION_DATA_JSON_PATH, 'r') as f:
#             all_frames_data = json.load(f)
#         MOTION_VIS_DATA["all_frames_data"] = all_frames_data
#     except Exception as e:
#         print(f"ERROR: Failed to load motion data JSON. Check path. Error: {e}"); return

#     # --- 2. Identify all unique marker corners and their initial positions ---
#     all_unique_marker_keys = set()
#     for frame_data in all_frames_data.values():
#         all_unique_marker_keys.update(frame_data.keys())

#     if not all_unique_marker_keys:
#         print("ERROR: No marker keys found in the JSON file."); return

#     ordered_keys_list = sorted(list(all_unique_marker_keys))
#     ordered_keys_map = {key: i for i, key in enumerate(ordered_keys_list)}
#     MOTION_VIS_DATA["ordered_keys_map"] = ordered_keys_map

#     # Find the first frame for initial positions
#     initial_positions = []
#     initial_frame_key = sorted(all_frames_data.keys(), key=int)[0]
#     initial_frame_data = all_frames_data.get(initial_frame_key, {})

#     for marker_key in ordered_keys_list:
#         pos_data = initial_frame_data.get(marker_key)
#         if pos_data and len(pos_data[0]) == 3:
#             initial_positions.append(pos_data[0])
#         else:
#             initial_positions.append((0.0, 0.0, 0.0))

#     # --- 3. Create the Main Point Cloud Object ---
#     # (This object holds the animation data but will now spawn spheres)
    
#     # Clean up old objects
#     if MOTION_CLOUD_OBJ_NAME in bpy.data.objects:
#         bpy.data.objects.remove(bpy.data.objects[MOTION_CLOUD_OBJ_NAME], do_unlink=True)
    
#     # Clean up the template sphere if it exists from a previous run
#     if f"{MOTION_CLOUD_OBJ_NAME}_Template" in bpy.data.objects:
#         bpy.data.objects.remove(bpy.data.objects[f"{MOTION_CLOUD_OBJ_NAME}_Template"], do_unlink=True)

#     mesh_data = bpy.data.meshes.new(MOTION_CLOUD_OBJ_NAME + "_Mesh")
#     motion_cloud_obj = bpy.data.objects.new(MOTION_CLOUD_OBJ_NAME, mesh_data)

#     # Apply transformations to the Point Cloud
#     rot_x_rad = math.radians(0) # Adjust if needed
#     rot_y_rad = math.radians(0)
#     rot_z_rad = math.radians(0)
#     motion_cloud_obj.rotation_euler = (rot_x_rad, rot_y_rad, rot_z_rad)
#     motion_cloud_obj.location = (0, 0, 0) # Adjust if needed

#     # Load initial vertices
#     mesh_data.from_pydata(initial_positions, [], [])
#     mesh_data.update()

#     # --- 3.5 Create the "Sphere" Mesh for Instancing ---
#     # We create ONE sphere, and Blender copies it to every vertex of the point cloud.
    
#     # Create a Red Material
#     mat_name = "Marker_Red_Mat"
#     if mat_name in bpy.data.materials:
#         mat = bpy.data.materials[mat_name]
#     else:
#         mat = bpy.data.materials.new(name=mat_name)
    
#     # Define the specific color requested
#     # RGBA: (Red, Green, Blue, Alpha)
#     # target_color = (1.0, 0.2, 0.0, 1.0) # ORANGE
#     target_color = (0.0, 1.0, 0.0, 1.0) # GREEN

#     mat.use_nodes = True
#     bsdf = mat.node_tree.nodes.get("Principled BSDF")
#     if bsdf:
#         bsdf.inputs['Base Color'].default_value = target_color # Red
#         # Optional: Add emission so they glow
#         bsdf.inputs['Emission Color'].default_value = (0.0, 0.0, 0.0, 1.0)
#         bsdf.inputs['Emission Strength'].default_value = 1.0
#     mat.diffuse_color = (0.0, 0.0, 0.0, 1.0) # For solid viewport

#     # Create the Sphere Geometry using BMesh
#     bm = bmesh.new()
#     # Radius = Size of your markers

#     bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=6, radius=0.005)
#     sphere_mesh = bpy.data.meshes.new(f"{MOTION_CLOUD_OBJ_NAME}_Sphere_Mesh")
#     bm.to_mesh(sphere_mesh)
#     bm.free()
    
#     # Assign material to the SPHERE (not the point cloud)
#     sphere_mesh.materials.append(mat)

#     # Create the Object
#     template_sphere = bpy.data.objects.new(f"{MOTION_CLOUD_OBJ_NAME}_Template", sphere_mesh)
    
#     # Parent the Sphere to the Point Cloud
#     template_sphere.parent = motion_cloud_obj
    
#     # --- Organize into Collections ---
#     if COLLECTION_NAME not in bpy.data.collections:
#         vis_collection = bpy.data.collections.new(COLLECTION_NAME)
#         if PARENT_COLLECTION_NAME in bpy.data.collections:
#             bpy.data.collections[PARENT_COLLECTION_NAME].children.link(vis_collection)
#         else:
#             parent_collection = bpy.data.collections.new(PARENT_COLLECTION_NAME)
#             bpy.context.scene.collection.children.link(parent_collection)
#             parent_collection.children.link(vis_collection)
#     else:
#         vis_collection = bpy.data.collections[COLLECTION_NAME]

#     # Link both objects to the collection
#     vis_collection.objects.link(motion_cloud_obj)
#     vis_collection.objects.link(template_sphere)

#     # --- 4. Activate Instancing ---
#     # This tells Blender: "Replace every vertex in motion_cloud_obj with the child object (template_sphere)"
#     motion_cloud_obj.instance_type = 'VERTS'
    
#     # Hide the original "dots" (the vertices), show only the spheres
#     motion_cloud_obj.show_instancer_for_viewport = False
#     motion_cloud_obj.show_instancer_for_render = False

#     print(f"Created instanced visualization object '{motion_cloud_obj.name}'.")

#     # --- 5. Register the frame change handler ---
#     bpy.app.handlers.frame_change_pre.clear()
#     bpy.app.handlers.frame_change_pre.append(on_frame_change_motion_vis)
#     on_frame_change_motion_vis(bpy.context.scene)

#     print("\n--- SETUP COMPLETE ---")

def on_frame_change_motion_vis(scene):
    """
    This function runs every time the frame changes and updates vertex positions.
    """
    obj = scene.objects.get(MOTION_CLOUD_OBJ_NAME)
    if not obj or not MOTION_VIS_DATA.get("all_frames_data"):
        return

    # Get the data for the current frame
    current_frame_str = str(scene.frame_current)
    data_for_this_frame = MOTION_VIS_DATA["all_frames_data"].get(current_frame_str, {})

    # Get the dictionary that maps all unique marker keys to their vertex indices
    all_keys_map = MOTION_VIS_DATA["ordered_keys_map"]

    # Get the total number of vertices in our mesh
    num_verts = len(obj.data.vertices)
    # Create a NumPy array to hold the new coordinates for fast update
    new_coords = np.empty(num_verts * 3, dtype=np.float32)

    for marker_key, v_idx in all_keys_map.items():
        if v_idx >= num_verts: continue # Safety check

        position_data = data_for_this_frame.get(marker_key)

        if position_data:
            # Marker exists in this frame, update to its new position
            try:
                # Assuming data format is [[X,Y,Z]]
                new_pos = position_data[0]
                if len(new_pos) == 3:
                     new_coords[v_idx * 3 : v_idx * 3 + 3] = new_pos
                else: # Data malformed, hide it
                    new_coords[v_idx * 3 : v_idx * 3 + 3] = HIDE_LOCATION
            except (TypeError, IndexError): # Handle malformed data
                new_coords[v_idx * 3 : v_idx * 3 + 3] = HIDE_LOCATION
        else:
            # Marker is not in this frame's data, move it to the "hide" location
            new_coords[v_idx * 3 : v_idx * 3 + 3] = HIDE_LOCATION

    # Update all vertex positions at once for performance
    obj.data.vertices.foreach_set("co", new_coords)
    obj.data.update() # Mark mesh data for update

# # --- Run the main setup function ---
# if __name__ == "__main__":
#     setup_motion_data_visualization()

def setup_motion_data_visualization():
    """
    One-time setup function to load all data, create the point cloud object,
    and register the animation handler.
    """
    global MOTION_VIS_DATA
    # Import numpy inside the function where it's used
    import numpy as np

    print("--- Setting up Motion Data Visualization ---")

    # --- 1. Load the entire motion data JSON into memory ---
    print(f"Loading motion data from: {MOTION_DATA_JSON_PATH}")
    try:
        with open(MOTION_DATA_JSON_PATH, 'r') as f:
            all_frames_data = json.load(f)
        MOTION_VIS_DATA["all_frames_data"] = all_frames_data
    except Exception as e:
        print(f"ERROR: Failed to load motion data JSON. Check path. Error: {e}"); return
    print(f"Loaded data for {len(all_frames_data)} frames.")

    # --- 2. Identify all unique marker corners and their initial positions ---
    all_unique_marker_keys = set()
    for frame_data in all_frames_data.values():
        all_unique_marker_keys.update(frame_data.keys())

    if not all_unique_marker_keys:
        print("ERROR: No marker keys found in the JSON file."); return

    # Create an ordered list and a map for consistent indexing
    ordered_keys_list = sorted(list(all_unique_marker_keys))
    ordered_keys_map = {key: i for i, key in enumerate(ordered_keys_list)}
    MOTION_VIS_DATA["ordered_keys_map"] = ordered_keys_map

    # Find the first frame that has data to get initial positions
    initial_positions = []
    initial_frame_key = sorted(all_frames_data.keys(), key=int)[0]
    initial_frame_data = all_frames_data.get(initial_frame_key, {})

    for marker_key in ordered_keys_list:
        pos_data = initial_frame_data.get(marker_key)
        if pos_data and len(pos_data[0]) == 3:
            initial_positions.append(pos_data[0])
        else:
            # If marker not in first frame, place it at origin initially
            initial_positions.append((0.0, 0.0, 0.0))

    print(f"Found {len(ordered_keys_list)} unique marker corners across all frames.")

    # --- 3. Create a single persistent mesh object for visualization ---
    if MOTION_CLOUD_OBJ_NAME in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[MOTION_CLOUD_OBJ_NAME], do_unlink=True)
    if MOTION_CLOUD_OBJ_NAME + "_Mesh" in bpy.data.meshes:
        bpy.data.meshes.remove(bpy.data.meshes[MOTION_CLOUD_OBJ_NAME + "_Mesh"])

    mesh_data = bpy.data.meshes.new(MOTION_CLOUD_OBJ_NAME + "_Mesh")
    motion_cloud_obj = bpy.data.objects.new(MOTION_CLOUD_OBJ_NAME, mesh_data)

    # Create a new material or use an existing one
    material_name = "Motion_Cloud_Material"
    if material_name in bpy.data.materials:
        mat = bpy.data.materials[material_name]
    else:
        mat = bpy.data.materials.new(name=material_name)

    # Enable nodes for the material
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled_bsdf = nodes.get("Principled BSDF")
    if principled_bsdf:
        # Set the base color (RGBA format)
        principled_bsdf.inputs["Base Color"].default_value = (1.0, 0.0, 0.0, 1.0)  # Red color
        principled_bsdf.inputs["Roughness"].default_value = 0.5  # Adjust roughness if needed

    # Set the viewport display color for the material
    mat.diffuse_color = (1.0, 0.0, 0.0, 1.0)  # Red color (RGBA)

    # Assign the material to the mesh
    mesh_data.materials.append(mat)

    # ** THE FIX FOR ROTATION **
    # --- NEW: APPLY CORRECTIVE ROTATION TO OBSERVED MOTION CLOUD ---
    # Define the corrective rotation in degrees
    rot_x_deg = 0 # 90.0
    rot_y_deg = 0.0
    rot_z_deg = 0 # 180.0

    # Convert degrees to radians for Blender's API
    rot_x_rad = math.radians(rot_x_deg)
    rot_y_rad = math.radians(rot_y_deg)
    rot_z_rad = math.radians(rot_z_deg)

    # Apply the rotation to the object's euler rotation property
    motion_cloud_obj.rotation_euler = (rot_x_rad, rot_y_rad, rot_z_rad)

    # Define the translation offsets - FIXED
    # trans_x = 0.001799
    # trans_y = 0.070466
    # trans_z = 0.007204

    trans_x = 0
    trans_y = 0
    trans_z = 0

    # Apply the translation to the object's location property
    motion_cloud_obj.location = (trans_x, trans_y, trans_z)

    # Create the mesh from the initial positions
    mesh_data.from_pydata(initial_positions, [], [])
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

    vis_collection.objects.link(motion_cloud_obj)
    print(f"Created visualization object '{motion_cloud_obj.name}'.")

    # --- 4. Register the frame change handler ---
    # Clear any previous handlers to avoid running multiple instances
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(on_frame_change_motion_vis)

    # Do an initial update for the current frame
    on_frame_change_motion_vis(bpy.context.scene)

    print("\n--- SETUP COMPLETE ---")
    print("Animation handler registered. Play the animation or scrub the timeline to see the recorded marker data.")
    print("Markers not detected in a frame will be moved far away.")
    print("To stop the live update, run: bpy.app.handlers.frame_change_pre.clear()")

# --- Run the main setup function ---
if __name__ == "__main__":
    setup_motion_data_visualization()