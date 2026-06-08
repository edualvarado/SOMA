"""
STATUS: Completed - Preprocess triangulated data to handle marker instance mismatches, save the updated data, and visualize differences by drawing lines.
Script: visualize_differences_markers_filter.py
Goal: Preprocess triangulated data to handle marker instance mismatches, save the updated data, and visualize differences by drawing lines.
"""

import bpy
import json
from mathutils import Vector
from collections import Counter
import os

# --- User Configuration ---
shots = ["shot_001", "shot_003", "shot_004", "shot_005", "shot_006", "shot_007", "shot_008", "shot_009", "shot_011", "shot_012", "shot_013", "shot_014", "shot_015", "shot_016", "shot_017", "shot_018", "shot_019", "shot_020"]  # Add your shot names here
# shots = ["shot_002", "shot_010"]  # Add your shot names here

# Threshold for maximum distance
d_swap = 0.1  # Adjust this value as needed
d_remove = 0.2  # Distance threshold to print marker IDs
frame_number = 15

# --- Functions ---
def preprocess_triangulated_data(triangulated_data, lbs_data):
    """
    Preprocess the triangulated data to handle marker instance mismatches.
    If the distance between a triangulated marker and its canonical counterpart exceeds d_max,
    swap the marker instance (e.g., marker_459_0_1 -> marker_459_1_1).
    """
    print("--- Step 1: Preprocessing Triangulated Data ---")
    swapped_count = 0  # Counter for swapped markers
    total_swapped_distance = 0.0  # Total distance for swapped markers
    total_same_distance = 0.0  # Total distance for non-swapped markers
    same_count = 0  # Counter for non-swapped markers

    for frame_str, triangulated_markers in triangulated_data.items():
        if frame_str not in lbs_data:
            continue

        lbs_markers = lbs_data[frame_str]

        for marker_key, triangulated_position in list(triangulated_markers.items()):
            # Parse the marker ID to extract the base ID, instance, and corner
            parts = marker_key.split("_")
            if len(parts) != 4:
                continue  # Skip invalid marker keys

            base_id = f"{parts[0]}_{parts[1]}"  # e.g., "marker_459"
            instance = parts[2]  # e.g., "0"
            corner = parts[3]  # e.g., "1"

            # Check if the alternate instance exists in the canonical data
            alternate_instance = "1" if instance == "0" else "0"
            alternate_marker_key = f"{base_id}_{alternate_instance}_{corner}"

            if marker_key in lbs_markers:
                p_lbs = Vector(lbs_markers[marker_key][0])
                p_triangulated = Vector(triangulated_position[0])
                distance = (p_triangulated - p_lbs).length

                # If the distance exceeds d_swap, swap the instance
                if distance > d_swap and alternate_marker_key in lbs_markers:
                    # print(f"Swapping {marker_key} with {alternate_marker_key} in frame {frame_str} (distance: {distance:.3f})")
                    triangulated_markers[alternate_marker_key] = triangulated_markers.pop(marker_key)
                    swapped_count += 1
                    total_swapped_distance += distance
                else:
                    same_count += 1
                    total_same_distance += distance

    # Calculate averages
    avg_swapped_distance = total_swapped_distance / swapped_count if swapped_count > 0 else 0.0
    avg_same_distance = total_same_distance / same_count if same_count > 0 else 0.0

    print(f"Swapping Preprocessing complete. Total swapped markers: {swapped_count}")
    print(f"Average distance of swapped markers: {avg_swapped_distance:.3f}")
    print(f"Average distance of markers that remained the same: {avg_same_distance:.3f}")

def save_preprocessed_data(triangulated_data, output_path):
    """
    Save the preprocessed triangulated data to a JSON file.
    """
    print(f"--- Step 2: Saving Preprocessed Data to {output_path} ---")
    try:
        with open(output_path, 'w') as f:
            json.dump(triangulated_data, f, indent=2)
        print("Preprocessed data saved successfully.")
    except Exception as e:
        print(f"ERROR: Failed to save preprocessed data. {e}")

def create_material(name, color):
    """
    Create a material for the difference lines.
    """
    if name in bpy.data.materials:
        return bpy.data.materials[name]

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Create new nodes
    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    output_node.location = (400, 0)

    diffuse_node = nodes.new(type="ShaderNodeBsdfDiffuse")
    diffuse_node.location = (0, 0)
    diffuse_node.inputs[0].default_value = (*color, 1.0)  # RGBA

    # Link nodes
    links.new(diffuse_node.outputs[0], output_node.inputs[0])

    return mat

def visualize_differences(triangulated_data, lbs_data):
    """
    Visualize differences between two sets of markers by drawing red lines.
    """
    print("--- Step 3: Visualizing Differences ---")

    # --- Create Collection ---
    if COLLECTION_NAME in bpy.data.collections:
        collection = bpy.data.collections[COLLECTION_NAME]
    else:
        collection = bpy.data.collections.new(COLLECTION_NAME)
        bpy.context.scene.collection.children.link(collection)

    # Clear existing objects in the collection
    for obj in collection.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # --- Create Materials ---
    thin_line_material = create_material(LINE_MATERIAL_NAME, (1.0, 0.0, 0.0))  # Red
    thick_line_material = create_material(THICK_LINE_MATERIAL_NAME, (1.0, 0.0, 0.0))  # Red (thicker)

    # Process only the specified frame for visualization
    debug_frame = frame_number  # Change this to the frame you want to visualize

    if str(debug_frame) in lbs_data:
        triangulated_markers = triangulated_data.get(str(debug_frame), {})
        lbs_markers = lbs_data[str(debug_frame)]

        for marker_key, lbs_position in lbs_markers.items():
            if marker_key not in triangulated_markers:
                continue

            # Get positions
            p_triangulated = Vector(triangulated_markers[marker_key][0])
            p_lbs = Vector(lbs_position[0])

            # Calculate the distance between the two points
            distance = (p_triangulated - p_lbs).length

            # Create a line connecting the two points
            mesh = bpy.data.meshes.new(f"Line_{marker_key}_{debug_frame}")
            obj = bpy.data.objects.new(f"Line_{marker_key}_{debug_frame}", mesh)
            collection.objects.link(obj)

            # Create vertices and edges
            mesh.from_pydata([p_triangulated, p_lbs], [(0, 1)], [])
            mesh.update()

            # Assign material based on distance
            if distance > d_swap:
                obj.data.materials.append(thick_line_material)
            else:
                obj.data.materials.append(thin_line_material)

        print(f"Finished visualizing lines for frame {debug_frame}.")
    else:
        print(f"Frame {debug_frame} not found in LBS data.")

    print("Finished visualizing differences for the first 1 frames.")

def remove_distant_markers(triangulated_data, lbs_data, d_remove):
    """
    Remove triangulated markers that exceed the distance threshold `d_remove`.
    """
    print("--- Step 4: Removing Distant Markers ---")
    removed_count = 0  # Counter for removed markers
    total_removed_distance = 0.0  # Total distance for removed markers
    total_untouched_distance = 0.0  # Total distance for untouched markers
    untouched_count = 0  # Counter for untouched markers

    for frame_str, triangulated_markers in triangulated_data.items():
        if frame_str not in lbs_data:
            continue

        lbs_markers = lbs_data[frame_str]

        markers_to_remove = []
        for marker_key, triangulated_position in triangulated_markers.items():
            if marker_key in lbs_markers:
                p_lbs = Vector(lbs_markers[marker_key][0])
                p_triangulated = Vector(triangulated_position[0])
                distance = (p_triangulated - p_lbs).length

                if distance > d_remove:
                    # print(f"Removing marker {marker_key} in frame {frame_str} (distance: {distance:.3f})")
                    markers_to_remove.append(marker_key)
                    removed_count += 1
                    total_removed_distance += distance
                else:
                    untouched_count += 1
                    total_untouched_distance += distance

        # Remove markers after iteration to avoid modifying the dictionary while iterating
        for marker_key in markers_to_remove:
            triangulated_markers.pop(marker_key)

    # Calculate averages
    avg_removed_distance = total_removed_distance / removed_count if removed_count > 0 else 0.0
    avg_untouched_distance = total_untouched_distance / untouched_count if untouched_count > 0 else 0.0

    print(f"Total markers removed: {removed_count}")
    print(f"Average distance of removed markers: {avg_removed_distance:.3f}")
    print(f"Average distance of untouched markers: {avg_untouched_distance:.3f}")

if __name__ == "__main__":
    for shot in shots:
        print(f"--- Processing {shot} ---")

        # Update paths for the current shot
        TRIANGULATED_JSON_PATH = f"C:/Users/ealvarad/00-Local/02-Python/Blender/data/registration/{shot}/triangulated_sequence_{shot}_transformed.json"
        LBS_JSON_PATH = f"C:/Users/ealvarad/00-Local/02-Python/Blender/data/displacements/{shot}/canonical/canonical_markers_lbs_{shot}_exported.json"
        OUTPUT_JSON_PATH = f"C:/Users/ealvarad/00-Local/02-Python/Blender/data/registration/{shot}/triangulated_sequence_{shot}_transformed_filtered.json"

        # Update collection and material names for the current shot
        COLLECTION_NAME = f"Differences_{shot[-3:]}"
        LINE_MATERIAL_NAME = "DifferenceLineMaterial"
        THICK_LINE_MATERIAL_NAME = "ThickDifferenceLineMaterial"

        try:
            # --- Load Data ---
            print("--- Loading Input Data ---")
            with open(TRIANGULATED_JSON_PATH, 'r') as f:
                triangulated_data = json.load(f)
            with open(LBS_JSON_PATH, 'r') as f:
                lbs_data = json.load(f)
        except Exception as e:
            print(f"ERROR: Failed to load input JSON files for {shot}. {e}")
            continue

        print(f"Loaded triangulated marker data for {len(triangulated_data)} frames.")
        print(f"Loaded LBS marker data for {len(lbs_data)} frames.")

        # --- Preprocess Data ---
        preprocess_triangulated_data(triangulated_data, lbs_data)

        # --- Remove Distant Markers ---
        remove_distant_markers(triangulated_data, lbs_data, d_remove)

        # --- Save Preprocessed Data ---
        save_preprocessed_data(triangulated_data, OUTPUT_JSON_PATH)

        # --- Visualize Differences ---
        # visualize_differences(triangulated_data, lbs_data)