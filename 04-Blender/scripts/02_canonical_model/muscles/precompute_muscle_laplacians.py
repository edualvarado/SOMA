"""
Script: precompute_muscle_laplacians.py
Goal: For each muscle mesh in a collection, compute its graph Laplacian matrix
      based on vertex connectivity and save all matrices to a single .npz file.
"""

import bpy
import numpy as np
from scipy.sparse import lil_matrix, save_npz

# --- User Configuration ---

# ---
shot = "shot_002"  # Change this to your shot name
# ---

MUSCLE_COLLECTION_NAME = f"canonical_muscle_complex_{shot[-3:]}" # Collection containing ALL muscle objects
OUTPUT_LAPLACIANS_PATH = "C:/Users/ealvarad/00-Local/02-Python/Blender/data/reconstruction/muscle_laplacians.npz"

# ----------------------------------------------------


def build_mesh_laplacian(mesh):
    """Builds a graph Laplacian matrix from a mesh's edge connectivity."""
    num_verts = len(mesh.vertices)
    if num_verts == 0:
        # Return an empty matrix if there are no vertices
        return csc_matrix((0, 0), dtype=np.float32)

    # lil_matrix is efficient for building sparse matrices incrementally
    L = lil_matrix((num_verts, num_verts), dtype=np.float32)

    # Create a list of edge connections for faster access
    edges = np.zeros((len(mesh.edges), 2), dtype=int)
    mesh.edges.foreach_get("vertices", edges.ravel())

    # --- THIS IS THE CORRECTED LOGIC ---

    # 1. Populate the off-diagonal elements (-1 for neighbors)
    for v1_idx, v2_idx in edges:
        L[v1_idx, v2_idx] = -1
        L[v2_idx, v1_idx] = -1

    # 2. Calculate the degree (number of neighbors) for each vertex
    #    The degree is the negative of the sum of the off-diagonal elements in a row.
    degree = -np.array(L.sum(axis=1)).flatten()

    # 3. Set the diagonal elements to the calculated degree
    #    The original script had L.set_diagonal(degree), which is incorrect.
    #    This loop is the correct way for lil_matrix.
    for i in range(num_verts):
        L[i, i] = degree[i]

    # ------------------------------------

    # Convert to CSC format for fast matrix operations later
    return L.asformat('csc')


def precompute_all_muscle_laplacians():
    print("--- Starting Muscle Laplacian Pre-computation ---")

    # --- 1. Get Muscle Objects from the Scene Collection ---
    muscle_collection = bpy.data.collections.get(MUSCLE_COLLECTION_NAME)
    if not muscle_collection:
        print(f"ERROR: Muscle collection '{MUSCLE_COLLECTION_NAME}' not found.")
        return

    source_muscle_objects = [obj for obj in muscle_collection.objects if obj.type == 'MESH']
    if not source_muscle_objects:
        print(f"ERROR: No mesh objects in collection '{MUSCLE_COLLECTION_NAME}'.")
        return

    print(f"Found {len(source_muscle_objects)} muscle objects to process.")

    # --- 2. Build and collect Laplacian for each muscle ---
    laplacian_matrices = {}
    for muscle_obj in source_muscle_objects:
        print(f"  Building Laplacian for '{muscle_obj.name}'...")
        mesh_data = muscle_obj.data
        L_muscle = build_mesh_laplacian(mesh_data)  # Uses the corrected build function from before
        laplacian_matrices[muscle_obj.name] = L_muscle

    # --- 3. Save all matrices to a single compressed .npz file (Corrected) ---
    print(f"\nSaving Laplacian matrices to: {OUTPUT_LAPLACIANS_PATH}")
    try:
        # Use numpy.savez_compressed instead of scipy.sparse.save_npz
        # The ** operator unpacks the dictionary into keyword arguments, e.g.,
        # Bicep_L=matrix_for_bicep, Tricep_L=matrix_for_tricep, ...
        np.savez_compressed(OUTPUT_LAPLACIANS_PATH, **laplacian_matrices)
        print("Save complete.")
    except Exception as e:
        print(f"ERROR writing .npz file: {e}")

if __name__ == '__main__':
    precompute_all_muscle_laplacians()