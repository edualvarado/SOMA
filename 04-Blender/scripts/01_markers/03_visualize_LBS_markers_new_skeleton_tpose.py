"""
STATUS: Completed - Visualize LBS markers and their deformation in Blender.
Script: visualize_LBS_markers.py
Goal: Take a set of 3D marker points from a canonical pose, the LBS weights for those markers and their bone influences,
in order to visualize how the markers deform when the armature is manipulated.
"""

import bpy
import json
from pathlib import Path

# --- User Configuration ---

# --- MODIFIED: Define a project root to make all paths relative and portable ---
# This assumes the script is located at: .../03-MUSK/04-Blender/scripts/...
# It goes up 5 levels to find the '03-MUSK' project root directory.
project_root = Path(__file__).resolve().parents[4]
print(f"Project root identified as: {project_root}")

# ---

# MODEL
# shot = "FBX-NewSkeleton-TPose"  # Change this to your shot name
shot = "shot_001"  # Change this to your shot name

# CANONICAL DATA IN A-POSE
# CANONICAL_MARKERS_JSON_PATH = "S:/work/03-MUSK/04-Blender/data/registration/canonical_model/canonical_data_tpose_new.json"
CANONICAL_MARKERS_JSON_PATH = "S:/work/03-MUSK/03-Registration/registration/S5/canonical_model/S5_canonical_data_tpose.json"

# --- ADDITION: Define an offset for the marker cloud ---
# Adjust these X, Y, Z values to move the entire point cloud.
# A negative Y value typically moves it forward in Blender's front view.
MARKER_CLOUD_OFFSET = (0.0, 0.0, 0.0)

# NEW SKELETON (INSIDE HUMANS ADAPTED TO STUDIO)
# ARMATURE_OBJECT_NAME = f"root-NewSkeleton-TPose"

# ARMATURE_OBJECT_NAME = f"root_001"
ARMATURE_OBJECT_NAME = f"root_001_S5_TPose"

# MARKER LBS WEIGHTS WITH RESPECT TO NEW SKELETON (INSIDE HUMANS ADAPTED TO STUDIO)
# MARKER_LBS_WEIGHTS_JSON_PATH = "S:/work/03-MUSK/04-Blender/data/weights/canonical_model/lbs_markers/markers_lbs_weights_exported_new_skeleton_tpose.json" 
MARKER_LBS_WEIGHTS_JSON_PATH = "S:/work/03-MUSK/03-Registration/registration/S5/canonical_model/S5_marker_lbs_weights_exported.json" 

# NAMING

# MARKER_CLOUD_OBJ_NAME = f"LBS_Canonical_Markers_S1_{shot}_TPose"
# COLLECTION_NAME = f"LBS_Canonical_Markers_S1_{shot}_TPose"
MARKER_CLOUD_OBJ_NAME = f"LBS_Canonical_Markers_S5_{shot}_TPose"
COLLECTION_NAME = f"LBS_Canonical_Markers_S5_{shot}_TPose"

# PARENT_COLLECTION_NAME = "FBX-NewSkeleton-TPose"
# PARENT_COLLECTION_NAME = "Shot_001"
PARENT_COLLECTION_NAME = "Shot_001_S5"

# ----------------------------------------------------

def visualize_lbs_marker_deformation():
    """
    Creates a point cloud from canonical markers, assigns LBS weights,
    and links it to an armature for deformation testing.
    """

    # --- 1. Load Canonical Marker Coordinates & Their Original String IDs ---
    print(f"Loading canonical marker data from: {CANONICAL_MARKERS_JSON_PATH}")
    try:
        with open(CANONICAL_MARKERS_JSON_PATH, 'r') as f:
            all_frames_data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Canonical marker JSON file not found at: {CANONICAL_MARKERS_JSON_PATH}");
        return
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from: {CANONICAL_MARKERS_JSON_PATH}");
        return

    static_pose_data = all_frames_data.get("0", {})
    if not static_pose_data:
        print("ERROR: No data for frame '0' in canonical marker JSON.");
        return

    marker_coords_list = []
    ordered_marker_keys = []  # Keep track of the order to match with LBS weights JSON
    for key, coord_list_of_list in static_pose_data.items():
        try:
            coord = coord_list_of_list[0]
            if len(coord) == 3:
                marker_coords_list.append(tuple(coord))
                ordered_marker_keys.append(key)
            else:
                print(f"Warning: Invalid coordinate for '{key}': {coord}. Skipping.")
        except (TypeError, IndexError):
            print(f"Warning: Malformed data for key '{key}': {coord_list_of_list}. Skipping.")

    if not marker_coords_list:
        print("ERROR: No valid 3D marker points extracted from canonical JSON.");
        return
    print(f"Loaded {len(marker_coords_list)} marker points.")

    # --- ADDITION: Apply the offset to all marker coordinates ---
    if MARKER_CLOUD_OFFSET != (0.0, 0.0, 0.0):
        print(f"Applying offset {MARKER_CLOUD_OFFSET} to marker coordinates.")
        marker_coords_list = [
            (p[0] + MARKER_CLOUD_OFFSET[0], p[1] + MARKER_CLOUD_OFFSET[1], p[2] + MARKER_CLOUD_OFFSET[2])
            for p in marker_coords_list
        ]

    # --- 2. Create or Get Target Collection ---
    # if COLLECTION_NAME in bpy.data.collections:
    #     vis_collection = bpy.data.collections[COLLECTION_NAME]
    # else:
    #     vis_collection = bpy.data.collections.new(COLLECTION_NAME)
    #     bpy.context.scene.collection.children.link(vis_collection)

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

    # --- 3. Create Marker Point Cloud Object ---
    # Delete object if it exists from a previous run
    if MARKER_CLOUD_OBJ_NAME in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[MARKER_CLOUD_OBJ_NAME], do_unlink=True)
    if MARKER_CLOUD_OBJ_NAME + "_Mesh" in bpy.data.meshes:  # Also clear mesh data
        bpy.data.meshes.remove(bpy.data.meshes[MARKER_CLOUD_OBJ_NAME + "_Mesh"])

    mesh_data = bpy.data.meshes.new(MARKER_CLOUD_OBJ_NAME + "_Mesh")
    marker_cloud_obj = bpy.data.objects.new(MARKER_CLOUD_OBJ_NAME, mesh_data)

    mesh_data.from_pydata(marker_coords_list, [], [])  # Vertices only, no edges or faces
    mesh_data.update()
    vis_collection.objects.link(marker_cloud_obj)
    print(f"Created marker cloud object: '{marker_cloud_obj.name}'")

    # --- 4. Load LBS Weights for Markers ---
    print(f"Loading LBS weights for markers from: {MARKER_LBS_WEIGHTS_JSON_PATH}")
    try:
        with open(MARKER_LBS_WEIGHTS_JSON_PATH, 'r') as f:
            marker_lbs_data = json.load(f)  # Dict: {"marker_id_str": {"bone_indices": [], "weights": []}}
    except FileNotFoundError:
        print(f"ERROR: Marker LBS weights JSON file not found at: {MARKER_LBS_WEIGHTS_JSON_PATH}");
        return
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from: {MARKER_LBS_WEIGHTS_JSON_PATH}");
        return
    print(f"Loaded LBS weights for {len(marker_lbs_data)} marker keys.")

    # --- 5. Get Armature and Create Bone Index to Bone Name Map ---
    armature_obj = bpy.data.objects.get(ARMATURE_OBJECT_NAME)
    if not armature_obj:
        print(f"ERROR: Armature object '{ARMATURE_OBJECT_NAME}' not found in the scene.");
        return
    if armature_obj.type != 'ARMATURE':
        print(f"ERROR: Object '{ARMATURE_OBJECT_NAME}' is not an Armature (type is {armature_obj.type}).");
        return

    # This map assumes the bone_indices in your JSON correspond to the
    # enumeration order of bones in armature_obj.data.bones. This should be
    # consistent if the Blender script that created exported_skin_data.json
    # used this same armature to define its bone_name_to_final_idx map.
    bone_idx_to_name_map = {idx: bone.name for idx, bone in enumerate(armature_obj.data.bones)}
    if not bone_idx_to_name_map:
        print("ERROR: Could not create bone index to name map from armature. Armature may have no bones.")
        return
    print(f"Created bone index map for {len(bone_idx_to_name_map)} bones in armature '{armature_obj.name}'.")

    # --- 6. Assign LBS Weights to Marker Cloud Object ---
    print(f"Assigning LBS weights to vertices of '{marker_cloud_obj.name}'...")
    assigned_weights_count = 0
    for v_idx, marker_key_str in enumerate(ordered_marker_keys):
        if v_idx >= len(marker_cloud_obj.data.vertices):
            print(
                f"Warning: Vertex index {v_idx} for marker '{marker_key_str}' out of range for created mesh. This shouldn't happen.");
            continue

        weight_data_for_marker = marker_lbs_data.get(marker_key_str)
        if not weight_data_for_marker or not weight_data_for_marker.get("bone_indices"):
            # print(f"Info: No LBS weight data found for marker '{marker_key_str}' (vertex {v_idx}). Vertex will not be deformed.")
            continue  # Skip if no weights

        bone_indices_from_json = weight_data_for_marker["bone_indices"]
        weights_from_json = weight_data_for_marker["weights"]

        for i, bone_idx_val in enumerate(bone_indices_from_json):
            weight_value = weights_from_json[i]
            bone_name = bone_idx_to_name_map.get(int(bone_idx_val))  # Ensure bone_idx_val is int

            if bone_name:
                # Get or create the vertex group for this bone name
                if bone_name not in marker_cloud_obj.vertex_groups:
                    marker_cloud_obj.vertex_groups.new(name=bone_name)

                vertex_group = marker_cloud_obj.vertex_groups[bone_name]
                try:
                    vertex_group.add([v_idx], weight_value, 'REPLACE')
                except RuntimeError as e:
                    print(f"Error adding weight for v_idx {v_idx}, bone '{bone_name}': {e}. Vertex might be invalid.")
            else:
                print(
                    f"Warning: Bone index {bone_idx_val} for marker '{marker_key_str}' (vertex {v_idx}) not found in armature's bone map. Skipping this bone influence.")
        assigned_weights_count += 1

    print(f"Assigned weights for {assigned_weights_count} marker points.")

    # --- 7. Link to Armature via Modifier ---
    # Remove existing armature modifiers on the object to avoid duplicates if script is re-run
    for mod in list(marker_cloud_obj.modifiers):  # Iterate over a copy
        if mod.type == 'ARMATURE':
            marker_cloud_obj.modifiers.remove(mod)

    modifier = marker_cloud_obj.modifiers.new(name="ArmatureDeform", type='ARMATURE')
    modifier.object = armature_obj
    # Optional: Parent the marker cloud to the armature (keeps them together if you move the armature in Object Mode)
    # marker_cloud_obj.parent = armature_obj
    # marker_cloud_obj.matrix_parent_inverse = armature_obj.matrix_world.inverted()

    print("\n--- SETUP COMPLETE FOR LBS VISUALIZATION ---")
    print(f"1. Select the armature object named '{ARMATURE_OBJECT_NAME}' in the Outliner.")
    print(f"2. Switch to 'Pose Mode' (Ctrl+Tab in 3D View, or select from dropdown).")
    print(f"3. Rotate some bones (e.g., select an arm bone, press 'R' to rotate).")
    print(f"4. Observe if the points in '{MARKER_CLOUD_OBJ_NAME}' deform correctly and smoothly.")
    print("   Look for points moving with the wrong bones, not moving enough, or moving too rigidly.")


# --- Run the main function ---
if __name__ == "__main__":
    # Ensure User Configuration variables at the top are set correctly
    visualize_lbs_marker_deformation()