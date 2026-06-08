"""
Script: estimate_dense_deformation.py
Goal: Takes a sparse set of observed local displacements and uses Laplacian
      interpolation to generate a dense displacement field for ALL markers
      for every frame. To use alone.
"""
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
SPARSE_DISPLACEMENTS_JSON_PATH = f"C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/displacements/{shot}/residuals/observed_residuals_only_{shot}.json"

# OUTPUT
OUTPUT_DENSE_DISPLACEMENTS_JSON_PATH = f"C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/displacements/{shot}/reconstruction/dense_local_displacements_{shot}.json"

# Number of neighbors to define the "smoothness" relationship between markers
NUM_NEIGHBORS_FOR_SMOOTHING = 8
# Weight of the data term. Higher means observed markers are followed more strictly.
DATA_TERM_WEIGHT = 100.0

# --------------------

def build_laplacian_matrix(points, num_neighbors):
    """Builds a graph Laplacian matrix based on K-Nearest Neighbors."""
    print(f"Building graph Laplacian for {len(points)} points...")
    num_points = len(points)
    kdtree = KDTree(points)

    # lil_matrix is efficient for building sparse matrices row by row
    L = lil_matrix((num_points, num_points))

    # The graph Laplacian L(i,j) is -1 if i,j are neighbors, and L(i,i) is the degree of i.
    for i in range(num_points):
        # Query for k+1 because the point itself is the closest
        _distances, indices = kdtree.query(points[i], k=num_neighbors + 1)

        degree = 0
        for j in indices:
            if i == j:
                continue  # Don't connect a point to itself
            L[i, j] = -1
            L[j, i] = -1  # Assuming symmetric graph
            degree += 1
        L[i, i] = degree

    # Convert to CSC format for fast matrix operations later
    return csc_matrix(L)


def create_dense_displacements():
    """Main function to run the sparse-to-dense interpolation."""
    print("--- Starting Sparse-to-Dense Displacement Calculation ---")

    # --- 1. Load Data ---
    print("Loading data files...")
    try:
        with open(CANONICAL_MARKERS_JSON_PATH, 'r') as f:
            canonical_points_raw = json.load(f).get("0", {})
        with open(SPARSE_DISPLACEMENTS_JSON_PATH, 'r') as f:
            sparse_displacements_by_frame = json.load(f)
    except Exception as e:
        print(f"ERROR loading JSON: {e}"); return

    ordered_marker_keys = sorted(list(canonical_points_raw.keys())) # Get sorted canonical marker keys
    marker_key_to_idx_map = {key: i for i, key in enumerate(ordered_marker_keys)} # Map keys to indices

    # Extract canonical points into a numpy array for efficient processing
    canonical_points_array = np.array([canonical_points_raw[key][0] for key in ordered_marker_keys])
    num_markers = len(ordered_marker_keys)

    # --- 2. Pre-compute Graph Laplacian ---
    # This defines the "smoothness" relationship between all markers
    L = build_laplacian_matrix(canonical_points_array, NUM_NEIGHBORS_FOR_SMOOTHING)
    # The energy term for smoothness is based on L.T * L
    A_smooth = L.T @ L

    # --- 3. Process each frame ---
    print(f"\nProcessing {len(sparse_displacements_by_frame)} frames...")
    all_frames_dense_displacements = {}

    for frame_str, observed_disps in sparse_displacements_by_frame.items():
        if int(frame_str) % 25 == 0:
            print(f"  Solving for frame {frame_str}...")

        # --- Set up the linear system Ax = b for this frame ---
        # A is the combination of the smoothness term and the data term constraints
        # b contains the known values for the constraints

        observed_indices = [marker_key_to_idx_map[key] for key in observed_disps.keys()]
        num_observed = len(observed_indices)

        if num_observed == 0:  # If no markers observed, displacement is zero for all
            all_frames_dense_displacements[frame_str] = {key: [0.0, 0.0, 0.0] for key in ordered_marker_keys}
            continue

        # Create the data term matrix (anchors)
        # It's a sparse matrix that is 1 for observed markers, 0 otherwise
        # Example: If observed_indices = [15, 88, 120] in this frame:
        rows = np.arange(num_observed) # Creates: [0, 1, 2]
        cols = np.array(observed_indices) # Is: [15, 88, 120]
        data = np.ones(num_observed)  # Creates: [1.0, 1.0, 1.0]
        A_data = csc_matrix((data, (rows, cols)), shape=(num_observed, num_markers)) # Size: (3, 2310)

        # Combine smoothness and data matrices
        A = A_smooth + DATA_TERM_WEIGHT * (A_data.T @ A_data)

        # ---

        # The result vector `b` needs to be solved for each axis (X, Y, Z)
        dense_displacements_xyz = []
        for axis in range(3):  # For X, Y, and Z
            # b contains the known displacement values for the observed anchors
            observed_disps_axis = np.array([disp[axis] for disp in observed_disps.values()])
            b = DATA_TERM_WEIGHT * (A_data.T @ observed_disps_axis)

            # Solve the sparse linear system: A * x = b
            x = spsolve(A, b)
            dense_displacements_xyz.append(x)

        # Combine the X,Y,Z results back into a (num_markers, 3) array
        dense_displacements_array = np.vstack(dense_displacements_xyz).T

        # Store in the final dictionary
        all_frames_dense_displacements[frame_str] = {
            ordered_marker_keys[i]: dense_displacements_array[i].tolist()
            for i in range(num_markers)
        }

    # --- 4. Save Final Dense Displacements ---
    print(f"\nSaving DENSE displacement data to: {OUTPUT_DENSE_DISPLACEMENTS_JSON_PATH}")

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
    create_dense_displacements()