"""
Script: estimate_refined_two_pass_deformation.py
Goal: Performs a second pass refinement on a globally smoothed displacement field.
      For each muscle, it enforces observed data while using the global field
      as a prior for unobserved points, ensuring anatomical boundaries are respected.
      To use along with estimate_dense_deformation.py.
"""

import json
import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import lil_matrix, csc_matrix, identity
from scipy.sparse.linalg import spsolve

# --- User Configuration ---

# ---
shot = "shot_014"  # Change this to your shot name
# ---

# INPUT
CANONICAL_MARKERS_JSON_PATH = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/registration/canonical_model/canonical_data.json"
SPARSE_DISPLACEMENTS_JSON_PATH = f"C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/displacements/{shot}/residuals/observed_residuals_only_{shot}.json"
MARKER_TO_MUSCLE_MAP_JSON = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/mappings/marker_to_muscle_map.json"
GLOBAL_DENSE_DISPLACEMENTS_JSON_PATH = f"C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/displacements/{shot}/reconstruction/dense_local_displacements_{shot}.json"

# OUTPUT
OUTPUT_REFINED_DISPLACEMENTS_JSON_PATH = f"C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/displacements/{shot}/reconstruction/refined_two_pass_displacements_{shot}.json"

# --- Parameters ---
NUM_NEIGHBORS_FOR_SMOOTHING = 8
# Weight for strictly following observed markers
OBSERVED_DATA_WEIGHT = 100.0
# Weight for loosely following the globally smoothed result for unobserved markers
GLOBAL_PRIOR_WEIGHT = 1.0

# --------------------

def build_laplacian_matrix(points, num_neighbors):
    """Builds a graph Laplacian matrix for a given set of points."""
    num_points = len(points)
    if num_points < 2: return csc_matrix((num_points, num_points))
    kdtree = KDTree(points)
    L = lil_matrix((num_points, num_points))
    k = min(num_neighbors, num_points - 1)
    if k <= 0: return L.asformat('csc')
    for i in range(num_points):
        _distances, indices = kdtree.query(points[i], k=k + 1)
        degree = 0
        for j in indices:
            if i == j: continue
            L[i, j] = -1;
            L[j, i] = -1;
            degree += 1
        L[i, i] = degree
    return L.asformat('csc')


def create_refined_displacements():
    print("--- Starting Two-Pass Refinement Calculation ---")

    # --- 1. Load Data ---
    try:
        with open(CANONICAL_MARKERS_JSON_PATH, 'r') as f:
            canonical_points_raw = json.load(f).get("0", {})
        with open(SPARSE_DISPLACEMENTS_JSON_PATH, 'r') as f:
            sparse_displacements_by_frame = json.load(f)
        with open(MARKER_TO_MUSCLE_MAP_JSON, 'r') as f:
            marker_to_muscle_map = json.load(f)
        with open(GLOBAL_DENSE_DISPLACEMENTS_JSON_PATH, 'r') as f:
            global_displacements_by_frame = json.load(f)
    except Exception as e:
        print(f"ERROR loading JSON: {e}"); return
    print("Loaded all source data files.")

    # --- 2. Group Markers by Muscle and Pre-compute Laplacians ---
    muscle_to_markers_map = {}
    for marker_key, muscle_name in marker_to_muscle_map.items():
        if muscle_name not in muscle_to_markers_map: muscle_to_markers_map[muscle_name] = []
        muscle_to_markers_map[muscle_name].append(marker_key)

    muscle_laplacians = {}
    for muscle_name, marker_keys in muscle_to_markers_map.items():
        muscle_marker_coords = np.array([canonical_points_raw[key][0] for key in marker_keys])
        muscle_laplacians[muscle_name] = build_laplacian_matrix(muscle_marker_coords, NUM_NEIGHBORS_FOR_SMOOTHING)
    print("Built Laplacian matrix for each muscle group.")

    # --- 3. Process Each Frame ---
    print(f"\nProcessing {len(sparse_displacements_by_frame)} frames...")
    all_frames_refined_displacements = {}

    for frame_str, observed_disps in sparse_displacements_by_frame.items():
        if int(frame_str) % 25 == 0: print(f"  Refining frame {frame_str}...")

        global_disps_t = global_displacements_by_frame.get(frame_str, {})
        refined_displacements_for_frame = {}

        # Loop through each muscle group to solve independently
        for muscle_name, assigned_marker_keys in muscle_to_markers_map.items():
            num_markers_on_muscle = len(assigned_marker_keys)
            if num_markers_on_muscle == 0: continue

            marker_key_to_local_idx = {key: i for i, key in enumerate(assigned_marker_keys)}

            # Identify observed and unobserved markers FOR THIS MUSCLE
            observed_keys = [k for k in assigned_marker_keys if k in observed_disps]
            unobserved_keys = [k for k in assigned_marker_keys if k not in observed_disps]

            # --- Set up the linear system Ax = b for THIS muscle ---
            L = muscle_laplacians[muscle_name]
            A_smooth = L.T @ L

            # Create selector matrices and target vectors for both observed and unobserved points
            obs_local_indices = [marker_key_to_local_idx[k] for k in observed_keys]
            unobs_local_indices = [marker_key_to_local_idx[k] for k in unobserved_keys]

            A_obs = csc_matrix(
                (np.ones(len(obs_local_indices)), (np.arange(len(obs_local_indices)), obs_local_indices)),
                shape=(len(obs_local_indices), num_markers_on_muscle))
            A_unobs = csc_matrix(
                (np.ones(len(unobs_local_indices)), (np.arange(len(unobs_local_indices)), unobs_local_indices)),
                shape=(len(unobs_local_indices), num_markers_on_muscle))

            # Combine into the final system matrix A
            A = A_smooth + OBSERVED_DATA_WEIGHT * (A_obs.T @ A_obs) + GLOBAL_PRIOR_WEIGHT * (A_unobs.T @ A_unobs)

            # Solve for each axis
            refined_disps_xyz = []
            for axis in range(3):
                # Build the target vector b from both observed data and the global prior
                b_obs = np.array([observed_disps[k][axis] for k in observed_keys]) if observed_keys else np.array([])
                b_unobs = np.array(
                    [global_disps_t.get(k, [0, 0, 0])[axis] for k in unobserved_keys]) if unobserved_keys else np.array(
                    [])

                b = OBSERVED_DATA_WEIGHT * (A_obs.T @ b_obs) + GLOBAL_PRIOR_WEIGHT * (A_unobs.T @ b_unobs)

                x = spsolve(A, b)
                refined_disps_xyz.append(x)

            refined_disps_array = np.vstack(refined_disps_xyz).T

            # Store results for this muscle's markers
            for i, key in enumerate(assigned_marker_keys):
                refined_displacements_for_frame[key] = refined_disps_array[i].tolist()

        all_frames_refined_displacements[frame_str] = refined_displacements_for_frame

    # --- 4. Save Final Refined Displacements ---
    print(f"\nSaving REFINED two-pass displacement data to: {OUTPUT_REFINED_DISPLACEMENTS_JSON_PATH}")
    with open(OUTPUT_REFINED_DISPLACEMENTS_JSON_PATH, 'w') as f:
        json.dump(all_frames_refined_displacements, f)
    print("Save complete.")


if __name__ == '__main__':
    create_refined_displacements()