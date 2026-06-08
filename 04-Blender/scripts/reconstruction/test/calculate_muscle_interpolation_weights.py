"""
Script: calculate_muscle_interpolation_weights.py
Goal: For each muscle, calculates interpolation weights for its vertices based
      only on the markers that have been assigned to that specific muscle.
"""
import bpy
import json
import numpy as np
from scipy.spatial import KDTree

# --- User Configuration ---

# ---
shot = "shot_002"  # Change this to your shot name
# ---

# INPUT
CANONICAL_MARKERS_JSON_PATH = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/registration/canonical_model/canonical_data.json"
MARKER_TO_MUSCLE_MAP_JSON = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/mappings/marker_to_muscle_map.json"
MUSCLE_COLLECTION_NAME = f"canonical_muscle_complex_{shot[-3:]}"  # The collection containing ALL muscle objects

# OUTPUT
OUTPUT_MUSCLE_INTERPOLATION_WEIGHTS_JSON = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/weights/canonical_model/muscle_interpolation_weights.json"

# --- Parameters ---
K_NEAREST_MARKERS = 8  # How many nearby markers on the SAME MUSCLE influence each vertex
FALLOFF_POWER = 2.0

# --------------------

def precompute_muscle_interpolation_weights():
    print("--- Starting Muscle-Specific Interpolation Weight Calculation ---")
    print(f"Using K={K_NEAREST_MARKERS} and Falloff Power: {FALLOFF_POWER}")

    # --- 1. Load Data ---
    print("Loading data files...")
    try:
        with open(CANONICAL_MARKERS_JSON_PATH, 'r') as f:
            canonical_points_raw = json.load(f).get("0", {})
        with open(MARKER_TO_MUSCLE_MAP_JSON, 'r') as f:
            marker_to_muscle_map = json.load(f)
    except Exception as e:
        print(f"ERROR loading JSON: {e}");
        return

    canonical_points = {key: np.array(val[0]) for key, val in canonical_points_raw.items()}
    print("Loaded canonical markers and marker-to-muscle map.")

    # --- 2. Get Muscle Objects and Group Markers by Muscle ---
    muscle_collection = bpy.data.collections.get(MUSCLE_COLLECTION_NAME)
    if not muscle_collection:
        print(f"ERROR: Muscle collection '{MUSCLE_COLLECTION_NAME}' not found.");
        return
    muscle_objects = [obj for obj in muscle_collection.objects if obj.type == 'MESH']
    if not muscle_objects:
        print(f"ERROR: No mesh objects in collection '{MUSCLE_COLLECTION_NAME}'.");
        return

    # Create a reverse map: {muscle_name: [list of marker keys]}
    muscle_to_markers_map = {}
    for marker_key, muscle_name in marker_to_muscle_map.items():
        if muscle_name not in muscle_to_markers_map:
            muscle_to_markers_map[muscle_name] = []
        muscle_to_markers_map[muscle_name].append(marker_key)
    print(f"Grouped markers for {len(muscle_to_markers_map)} muscles.")

    # This will be the final data structure to save
    final_muscle_interpolation_weights = {}

    # --- 3. Main Loop: Iterate through each Muscle Object ---
    print("\nCalculating weights for each muscle...")
    for muscle_obj in muscle_objects:
        muscle_name = muscle_obj.name
        print(f"  Processing muscle: {muscle_name}")

        # Get the list of markers assigned to this muscle
        assigned_marker_keys = muscle_to_markers_map.get(muscle_name)
        if not assigned_marker_keys:
            print(f"    Warning: No markers assigned to muscle '{muscle_name}'. Skipping.")
            final_muscle_interpolation_weights[muscle_name] = {}
            continue

        # Get the 3D coordinates for only this muscle's assigned markers
        assigned_marker_coords = np.array([canonical_points[key] for key in assigned_marker_keys])

        # Check if we have enough markers for the K value
        num_assigned_markers = len(assigned_marker_keys)
        current_k = min(K_NEAREST_MARKERS, num_assigned_markers)
        if current_k == 0: continue

        # Build a KDTree specifically for this muscle's markers
        marker_kdtree_muscle = KDTree(assigned_marker_coords)

        # Get this muscle's vertices in world space
        mesh = muscle_obj.data
        matrix_world = muscle_obj.matrix_world
        muscle_vertices_world = np.array([matrix_world @ v.co for v in mesh.vertices])

        weights_for_this_muscle = {}
        epsilon = 1e-9

        # Inner Loop: Iterate through each vertex of the current muscle
        for v_idx, vertex_world_coord in enumerate(muscle_vertices_world):
            distances, neighbor_indices_local = marker_kdtree_muscle.query(vertex_world_coord, k=current_k)

            # Logic to handle K=1 case from scipy
            if current_k == 1:
                distances = [distances]
                neighbor_indices_local = [neighbor_indices_local]

            # Calculate and normalize inverse distance weights
            inv_dist_weights = [1.0 / (dist ** FALLOFF_POWER + epsilon) for dist in distances]
            sum_inv_dist = sum(inv_dist_weights)

            if sum_inv_dist > 0:
                normalized_weights = [w / sum_inv_dist for w in inv_dist_weights]
                # Get the original string keys for the influencing markers
                influencing_marker_keys = [assigned_marker_keys[int(idx)] for idx in neighbor_indices_local]

                weights_for_this_muscle[str(v_idx)] = {
                    "influencing_markers": influencing_marker_keys,
                    "interpolation_weights": normalized_weights
                }

        final_muscle_interpolation_weights[muscle_name] = weights_for_this_muscle

    # --- 4. Save the final map to a file ---
    print(f"\nSaving muscle interpolation weights to: {OUTPUT_MUSCLE_INTERPOLATION_WEIGHTS_JSON}")
    try:
        with open(OUTPUT_MUSCLE_INTERPOLATION_WEIGHTS_JSON, 'w') as f:
            json.dump(final_muscle_interpolation_weights, f)  # No indent for smaller file size
        print("Save complete.")
    except Exception as e:
        print(f"ERROR writing weights JSON: {e}")


if __name__ == '__main__':
    precompute_muscle_interpolation_weights()