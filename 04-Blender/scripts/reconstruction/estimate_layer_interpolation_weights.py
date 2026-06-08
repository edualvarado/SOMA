"""
Script: calculate_layer_interpolation_weights_from_scene.py (v3)
Goal: Calculates skin interpolation weights with an adjustable falloff power
      and correctly handles K=1 nearest neighbor case.
"""
import bpy
import json
import numpy as np
# You must have scipy installed in Blender's Python for this to work
from scipy.spatial import KDTree

# --- User Configuration ---

# ---
shot = "shot_000"  # Change this to your shot name
# ---

# INPUT
CANONICAL_MARKERS_JSON_PATH = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/registration/canonical_model/canonical_data.json"

# SKIN_MESH_OBJECT_NAME = f"canonical_skin_{shot[-3:]}"  # <<< CHANGE THIS
SKIN_MESH_OBJECT_NAME = f"canonical_muscle_{shot[-3:]}"  # <<< CHANGE THIS

# OUTPUT
# OUTPUT_INTERPOLATION_WEIGHTS_JSON = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/weights/canonical_model/skin_layer_interpolation_weights.json"
OUTPUT_INTERPOLATION_WEIGHTS_JSON = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/weights/canonical_model/muscle_layer_interpolation_weights.json"

# --- Parameters ---
K_NEAREST_MARKERS = 8  # How many nearby markers influence each skin vertex
FALLOFF_POWER = 2.0

# --------------------

def precompute_interpolation_weights_from_scene():
    print("--- Starting Pre-computation of Skin Interpolation Weights (v3) ---")
    print(f"Using K={K_NEAREST_MARKERS} and Falloff Power: {FALLOFF_POWER}")

    # --- 1. Load Canonical Marker Points ---
    try:
        with open(CANONICAL_MARKERS_JSON_PATH, 'r') as f:
            canonical_points_raw = json.load(f).get("0", {})
        ordered_marker_keys = list(canonical_points_raw.keys())
        marker_coords_array = np.array([v[0] for v in canonical_points_raw.values()])
        print(f"Loaded {len(ordered_marker_keys)} canonical marker points.")
    except Exception as e:
        print(f"ERROR loading canonical marker JSON: {e}");
        return

    # --- 2. Get the Skin Mesh Object from the Blender Scene ---
    skin_mesh_obj = bpy.data.objects.get(SKIN_MESH_OBJECT_NAME)
    if not skin_mesh_obj: print(f"ERROR: Object '{SKIN_MESH_OBJECT_NAME}' not found."); return
    if skin_mesh_obj.type != 'MESH': print(f"ERROR: Object '{SKIN_MESH_OBJECT_NAME}' is not a MESH."); return

    mesh = skin_mesh_obj.data
    matrix_world = skin_mesh_obj.matrix_world
    skin_mesh_vertices_world = np.array([matrix_world @ v.co for v in mesh.vertices])
    print(f"Loaded skin mesh with {len(skin_mesh_vertices_world)} vertices.")

    # --- 3. Build KDTree from marker points for fast lookup ---
    print("Building KDTree from marker positions...")
    marker_kdtree = KDTree(marker_coords_array)

    # --- 4. For each skin vertex, find K-nearest markers and calculate weights ---
    print(f"Calculating weights for {len(skin_mesh_vertices_world)} skin vertices...")
    interpolation_weights_data = {}
    epsilon = 1e-9

    for i, vertex_world_coord in enumerate(skin_mesh_vertices_world):
        if i > 0 and i % 5000 == 0:
            print(f"  Processing skin vertex {i} / {len(skin_mesh_vertices_world)}")

        distances, neighbor_marker_indices = marker_kdtree.query(vertex_world_coord, k=K_NEAREST_MARKERS)

        # --- THIS IS THE KEY CHANGE TO HANDLE K=1 ---
        if K_NEAREST_MARKERS == 1:
            # When K=1, the result is a single number, not a list.
            # The weight is 100% from this single closest marker.
            normalized_weights = [1.0]
            # Make sure neighbor_marker_indices is treated as a list-like object
            influencing_marker_keys = [ordered_marker_keys[int(neighbor_marker_indices)]]
        else:
            # This is the original logic for K > 1
            weights = []
            sum_of_inverse_distances = 0.0
            for dist in distances:
                weight = 1.0 / (dist ** FALLOFF_POWER + epsilon)
                weights.append(weight)
                sum_of_inverse_distances += weight

            if sum_of_inverse_distances > 0:
                normalized_weights = [w / sum_of_inverse_distances for w in weights]
                influencing_marker_keys = [ordered_marker_keys[int(idx)] for idx in neighbor_marker_indices]
            else:  # Fallback for rare cases
                normalized_weights = [1.0] * K_NEAREST_MARKERS  # Assign equal weight if all distances are zero or huge
                influencing_marker_keys = [ordered_marker_keys[int(idx)] for idx in neighbor_marker_indices]
        # --- END OF CHANGE ---

        interpolation_weights_data[str(i)] = {
            "influencing_markers": influencing_marker_keys,
            "interpolation_weights": normalized_weights
        }

    # --- 5. Save the computed weights to a file ---
    print(f"\nSaving interpolation weights to: {OUTPUT_INTERPOLATION_WEIGHTS_JSON}")
    try:
        with open(OUTPUT_INTERPOLATION_WEIGHTS_JSON, 'w') as f:
            json.dump(interpolation_weights_data, f, indent=2)
        print("Save complete.")
    except Exception as e:
        print(f"ERROR writing interpolation weights JSON: {e}")


if __name__ == '__main__':
    precompute_interpolation_weights_from_scene()