"""
Script: solve_final_muscle_deformation.py
Goal: A heavy, one-time processing script that solves for the dense deformation
      of all muscle meshes based on sparse observed marker displacements,
      using the muscle's own Laplacian for smoothness.
"""

import bpy
import json
import numpy as np
from mathutils import Vector
from scipy.sparse import lil_matrix, csc_matrix, identity, vstack
from scipy.sparse.linalg import spsolve

# --- User Configuration ---

# ---
shot = "shot_002"  # Change this to your shot name
FRAME_LIMIT = 120  # Limit to first 1000 frames for performance
# ---

ARMATURE_OBJECT_NAME = f"root_{shot[-3:]}"
MUSCLE_COLLECTION_NAME = f"canonical_muscle_complex_{shot[-3:]}" # Collection containing ALL muscle objects

# --- Input File Paths ---
CANONICAL_MARKERS_JSON_PATH = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/registration/canonical_model/canonical_data.json"
MARKER_LBS_WEIGHTS_JSON_PATH = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/weights/canonical_model/marker_lbs_weights.json"
SPARSE_DISPLACEMENTS_JSON_PATH = f"C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/displacements/{shot}/observed_residuals_only_{shot}.json"
BARYCENTRIC_MAP_JSON_PATH = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/mappings/marker_barycentric_map.json"
MUSCLE_LAPLACIANS_NPZ_PATH = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/mappings/muscle_laplacians.npz"

# --- Output File Path ---
OUTPUT_DEFORMATION_CACHE_JSON = f"C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/displacements/{shot}/final_precomputed_deformation_cache_{shot}_{FRAME_LIMIT}.json"

# --- Parameters ---
DATA_TERM_WEIGHT = 100.0  # How strictly to follow the observed marker displacements
DATA_SCALE_FACTOR = 0.001  # To convert displacement units if needed (e.g., mm to m)



# ----------------------------------------------------

def solve_all_muscle_deformations():
    print("--- Starting Final Muscle Deformation Calculation ---")

    # --- 1. Load All Data and Objects ---
    print("Loading data files...")
    try:
        with open(CANONICAL_MARKERS_JSON_PATH, 'r') as f:
            canonical_points_raw = json.load(f).get("0", {})
        with open(MARKER_LBS_WEIGHTS_JSON_PATH, 'r') as f:
            marker_lbs_weights = json.load(f)
        with open(SPARSE_DISPLACEMENTS_JSON_PATH, 'r') as f:
            sparse_displacements_by_frame = json.load(f)
        with open(BARYCENTRIC_MAP_JSON_PATH, 'r') as f:
            marker_barycentric_map = json.load(f)
        laplacian_data = np.load(MUSCLE_LAPLACIANS_NPZ_PATH, allow_pickle=True)
        muscle_laplacians = {name: laplacian_data[name].item() for name in laplacian_data.files}
    except Exception as e:
        print(f"ERROR: Failed to load data files: {e}"); return

    armature_obj = bpy.data.objects.get(ARMATURE_OBJECT_NAME)
    muscle_collection = bpy.data.collections.get(MUSCLE_COLLECTION_NAME)
    if not armature_obj or not muscle_collection: print("ERROR: Armature or Muscle Collection not found."); return
    source_muscle_objects = {obj.name: obj for obj in muscle_collection.objects if obj.type == 'MESH'}

    # --- 2. Prepare Bind Data and Mappings ---
    bind_pose_rotations = {idx: (armature_obj.matrix_world @ b.matrix_local).to_3x3() for idx, b in
                           enumerate(armature_obj.data.bones)}
    primary_bone_map = {key: wd["bone_indices"][np.argmax(wd["weights"])] for key, wd in marker_lbs_weights.items() if
                        wd and wd.get("bone_indices")}
    print("Prepared all necessary data and mappings.")

    # --- 3. Process Each Frame ---
    print(f"\nProcessing {len(sparse_displacements_by_frame)} frames...")
    final_deformation_cache = {}
    sorted_frame_keys = sorted(sparse_displacements_by_frame.keys(), key=int)

    for frame_str in sorted_frame_keys:
        frame = int(frame_str)
        if frame % 10 == 0: print(f"  Solving for frame {frame}...")

        if frame == FRAME_LIMIT:
            break

        bpy.context.scene.frame_set(frame)
        posed_bone_rotations = {b.name: b.matrix.to_3x3() for b in armature_obj.pose.bones}
        observed_disps_t = sparse_displacements_by_frame[frame_str]

        # Pre-calculate the A-pose world-space displacements for all observed markers this frame
        observed_world_disps_t = {}
        for marker_key, d_local_list in observed_disps_t.items():
            primary_bone_idx = primary_bone_map.get(marker_key)
            if primary_bone_idx is not None:
                R_bind = bind_pose_rotations[primary_bone_idx]
                observed_world_disps_t[marker_key] = R_bind @ Vector(d_local_list)

        deformations_for_this_frame = {}
        # --- Loop through each muscle to solve its deformation ---
        for muscle_name, muscle_obj in source_muscle_objects.items():

            L_muscle = muscle_laplacians.get(muscle_name)
            if L_muscle is None: continue

            num_verts = len(muscle_obj.data.vertices)

            # Find which observed markers constrain THIS muscle
            observed_markers_on_this_muscle = [
                key for key in observed_disps_t.keys()
                if marker_barycentric_map.get(key, {}).get("muscle_name") == muscle_name
            ]

            if not observed_markers_on_this_muscle:
                # No observed markers, so no non-rigid deformation for this muscle in this frame
                p_unposed = np.array([muscle_obj.matrix_world @ v.co for v in muscle_obj.data.vertices])
                deformations_for_this_frame[muscle_name] = p_unposed.tolist()
                continue

            # --- Set up the linear system Ax = b for THIS muscle ---
            A_smooth = L_muscle.T @ L_muscle

            # Build the barycentric constraint matrix `B`
            num_constraints = len(observed_markers_on_this_muscle)
            rows, cols, data = [], [], []
            target_disps_list = []
            for i, marker_key in enumerate(observed_markers_on_this_muscle):
                bary_info = marker_barycentric_map[marker_key]
                target_disps_list.append(observed_world_disps_t[marker_key])
                for j, v_idx in enumerate(bary_info["vertex_indices"]):
                    rows.append(i);
                    cols.append(v_idx);
                    data.append(bary_info["bary_coords"][j])

            B_matrix = csc_matrix((data, (rows, cols)), shape=(num_constraints, num_verts))
            A_data = B_matrix.T @ B_matrix

            # Combine for final system matrix A
            A = A_smooth + DATA_TERM_WEIGHT * A_data

            # Solve for each axis
            target_disps_array = np.array(target_disps_list)
            dense_disps_xyz = []
            for axis in range(3):
                b = DATA_TERM_WEIGHT * (B_matrix.T @ target_disps_array[:, axis])
                x = spsolve(A, b)
                dense_disps_xyz.append(x)

            D_muscle_t = np.vstack(dense_disps_xyz).T

            # Calculate final vertex positions
            P_unposed = np.array([muscle_obj.matrix_world @ v.co for v in muscle_obj.data.vertices])
            P_muscle_final_t = P_unposed + (D_muscle_t * DATA_SCALE_FACTOR)
            deformations_for_this_frame[muscle_name] = P_muscle_final_t.tolist()

        final_deformation_cache[frame_str] = deformations_for_this_frame

    # --- 4. Save the huge cache file ---
    print(f"\nSaving final pre-computed deformation cache to: {OUTPUT_DEFORMATION_CACHE_JSON}")
    try:
        with open(OUTPUT_DEFORMATION_CACHE_JSON, 'w') as f:
            json.dump(final_deformation_cache, f)
        print("Save complete.")
    except Exception as e:
        print(f"ERROR writing cache JSON: {e}")


if __name__ == '__main__':
    solve_all_muscle_deformations()