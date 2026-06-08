"""
Script: estimate_dense_muscle_constrained_deformation.py
Goal: Takes a sparse set of observed local displacements and uses
      muscle-constrained Laplacian interpolation to generate a dense
      displacement field for ALL markers for every frame. To use alone.
"""
import bpy
import json
import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
import os

# --- User Configuration ---

# ---
shot = "shot_014"  # Change this to your shot name
# ---

# INPUT
CANONICAL_MARKERS_JSON_PATH = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/registration/canonical_model/canonical_data.json"
MARKER_LBS_WEIGHTS_JSON_PATH = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/weights/canonical_model/marker_lbs_weights.json"
SPARSE_DISPLACEMENTS_JSON_PATH = f"C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/displacements/{shot}/residuals/observed_residuals_only_{shot}.json"
MARKER_TO_MUSCLE_MAP_JSON = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/mappings/marker_to_muscle_map.json"
ARMATURE_OBJECT_NAME = f"root_{shot[-3:]}"

# OUTPUT
OUTPUT_DENSE_DISPLACEMENTS_JSON_PATH = f"C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/displacements/{shot}/reconstruction/dense_muscle_constrained_displacements_{shot}.json"

# --- Parameters ---
NUM_NEIGHBORS_FOR_SMOOTHING = 8
DATA_TERM_WEIGHT = 100.0

# --------------------

def build_laplacian_matrix(points, num_neighbors):
    """Builds a graph Laplacian matrix for a given set of points."""
    num_points = len(points)
    if num_points == 0: return None

    kdtree = KDTree(points)
    L = lil_matrix((num_points, num_points))

    # Use min(k, num_points-1) to handle small muscle groups
    k = min(num_neighbors, num_points - 1)
    if k <= 0: return L.asformat('csc')

    for i in range(num_points):
        _distances, indices = kdtree.query(points[i], k=k + 1)
        degree = 0
        for j in indices:
            if i == j: continue
            L[i, j] = -1
            L[j, i] = -1
            degree += 1
        L[i, i] = degree
    return L.asformat('csc')


def create_muscle_constrained_dense_displacements():
    print("--- Starting Muscle-Constrained Sparse-to-Dense Calculation ---")

    # --- 1. Load Data ---
    try:
        with open(CANONICAL_MARKERS_JSON_PATH, 'r') as f:
            canonical_points_raw = json.load(f).get("0", {})
        with open(SPARSE_DISPLACEMENTS_JSON_PATH, 'r') as f:
            sparse_displacements_by_frame = json.load(f)
        with open(MARKER_TO_MUSCLE_MAP_JSON, 'r') as f:
            marker_to_muscle_map = json.load(f)
    except Exception as e:
        print(f"ERROR loading JSON: {e}"); return

    print("Loaded source data files.")

    # --- 2. Group Markers by Muscle ---
    muscle_to_markers_map = {}
    for marker_key, muscle_name in marker_to_muscle_map.items():
        if muscle_name not in muscle_to_markers_map:
            muscle_to_markers_map[muscle_name] = []
        muscle_to_markers_map[muscle_name].append(marker_key)
    print(f"Grouped markers for {len(muscle_to_markers_map)} muscles.")

    # --- 3. Pre-compute a Laplacian Matrix for each Muscle Group ---
    muscle_laplacians = {}
    for muscle_name, marker_keys in muscle_to_markers_map.items():
        muscle_marker_coords = np.array([canonical_points_raw[key][0] for key in marker_keys])
        L_muscle = build_laplacian_matrix(muscle_marker_coords, NUM_NEIGHBORS_FOR_SMOOTHING)
        muscle_laplacians[muscle_name] = L_muscle
    print("Built Laplacian matrix for each muscle group.")

    # --- 4. Process Each Frame ---
    print(f"\nProcessing {len(sparse_displacements_by_frame)} frames...")
    all_frames_dense_displacements = {}

    for frame_str, observed_disps in sparse_displacements_by_frame.items():
        frame = int(frame_str)
        if frame % 25 == 0: print(f"  Solving for frame {frame}...")

        dense_displacements_for_frame = {}

        # --- Loop through each muscle group to solve independently ---
        for muscle_name, assigned_marker_keys in muscle_to_markers_map.items():
            num_markers_on_muscle = len(assigned_marker_keys)
            if num_markers_on_muscle == 0: continue

            marker_key_to_local_idx = {key: i for i, key in enumerate(assigned_marker_keys)}

            # Find which markers on THIS muscle were observed in this frame
            observed_in_muscle_keys = [key for key in assigned_marker_keys if key in observed_disps]

            if not observed_in_muscle_keys:
                # No observed anchors, displacement is zero for all markers on this muscle
                for key in assigned_marker_keys: dense_displacements_for_frame[key] = [0.0, 0.0, 0.0]
                continue

            # Set up the linear system for THIS muscle
            L = muscle_laplacians[muscle_name]
            A_smooth = L.T @ L

            observed_local_indices = [marker_key_to_local_idx[key] for key in observed_in_muscle_keys]
            num_observed = len(observed_local_indices)

            rows = np.arange(num_observed)
            cols = np.array(observed_local_indices)
            data = np.ones(num_observed)
            A_data = csc_matrix((data, (rows, cols)), shape=(num_observed, num_markers_on_muscle))

            A = A_smooth + DATA_TERM_WEIGHT * (A_data.T @ A_data)

            # Solve for each axis
            dense_disps_xyz = []
            for axis in range(3):
                observed_disps_axis = np.array([observed_disps[key][axis] for key in observed_in_muscle_keys])
                b = DATA_TERM_WEIGHT * (A_data.T @ observed_disps_axis)
                x = spsolve(A, b)
                dense_disps_xyz.append(x)

            dense_disps_array = np.vstack(dense_disps_xyz).T

            # Store results for this muscle's markers
            for i, key in enumerate(assigned_marker_keys):
                dense_displacements_for_frame[key] = dense_disps_array[i].tolist()

        all_frames_dense_displacements[frame_str] = dense_displacements_for_frame

    # --- 5. Save Final Dense Displacements ---
    print(f"\nSaving DENSE muscle-constrained displacement data to: {OUTPUT_DENSE_DISPLACEMENTS_JSON_PATH}")

    # Ensure the directory exists
    output_dir = os.path.dirname(OUTPUT_DENSE_DISPLACEMENTS_JSON_PATH)
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(OUTPUT_DENSE_DISPLACEMENTS_JSON_PATH, 'w') as f:
            json.dump(all_frames_dense_displacements, f)
        print("Save complete.")
    except Exception as e:
        print(f"ERROR writing dense displacements JSON: {e}")


if __name__ == '__main__':
    create_muscle_constrained_dense_displacements()