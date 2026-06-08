"""
Script: refine_muscle_deformation_two_pass.py
Goal: A heavy, one-time script that performs a second-pass refinement on a
      globally smoothed displacement field to generate the final, anatomically
      constrained muscle deformation cache.
"""
import bpy
import json
import numpy as np
from mathutils import Vector, Matrix
from scipy.sparse import lil_matrix, csc_matrix, identity
from scipy.sparse.linalg import spsolve

# --- User Configuration ---

# ---
shot = "shot_002"  # Change this to your shot name
FRAME_LIMIT = 200  # Limit to first 1000 frames for performance
# ---



MUSCLE_COLLECTION_NAME = f"canonical_muscle_complex_{shot[-3:]}" # Collection containing ALL muscle objects
ARMATURE_OBJECT_NAME = f"root_{shot[-3:]}"

# --- Input File Paths ---
CANONICAL_MARKERS_JSON_PATH = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/registration/canonical_model/canonical_data.json"
MARKER_LBS_WEIGHTS_JSON_PATH = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/weights/canonical_model/marker_lbs_weights.json"
BARYCENTRIC_MAP_JSON_PATH = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/mappings/marker_barycentric_map.json"
MUSCLE_LAPLACIANS_NPZ_PATH = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/mappings/muscle_laplacians.npz"
SPARSE_DISPLACEMENTS_JSON_PATH = f"C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/displacements/{shot}/observed_residuals_only_{shot}.json"
# This is the output from your FIRST pass (global dense smoothing)
GLOBAL_DENSE_DISPLACEMENTS_JSON_PATH = f"C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/displacements/{shot}/dense_local_displacements_{shot}.json"

# --- Output File Path ---
OUTPUT_REFINED_DEFORMATION_CACHE_JSON = f"C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/displacements/{shot}/final_refined_deformation_cache_{shot}_{FRAME_LIMIT}.json"

# --- Parameters ---
DATA_SCALE_FACTOR = 0.001
OBSERVED_DATA_WEIGHT = 100.0  # High confidence in observed markers
GLOBAL_PRIOR_WEIGHT = 1.0  # Lower confidence "suggestion" from the global pass


# ----------------------------------------------------

def refine_all_muscle_deformations():
    print("--- Starting Two-Pass Refinement Calculation ---")

    # --- 1. Load All Data and Objects ---
    print("Loading all necessary data files...")
    try:
        with open(SPARSE_DISPLACEMENTS_JSON_PATH, 'r') as f:
            sparse_displacements_by_frame = json.load(f)
        with open(GLOBAL_DENSE_DISPLACEMENTS_JSON_PATH, 'r') as f:
            global_displacements_by_frame = json.load(f)
        with open(BARYCENTRIC_MAP_JSON_PATH, 'r') as f:
            marker_barycentric_map = json.load(f)
        with open(MARKER_LBS_WEIGHTS_JSON_PATH, 'r') as f:
            marker_lbs_weights = json.load(f)
        laplacian_data = np.load(MUSCLE_LAPLACIANS_NPZ_PATH, allow_pickle=True)
        muscle_laplacians = {name: laplacian_data[name].item() for name in laplacian_data.files}
    except Exception as e:
        print(f"ERROR: Failed to load data files: {e}"); return

    armature_obj = bpy.data.objects.get(ARMATURE_OBJECT_NAME)
    muscle_collection = bpy.data.collections.get(MUSCLE_COLLECTION_NAME)
    if not armature_obj or not muscle_collection: print("ERROR: Armature or Muscle Collection not found."); return
    source_muscle_objects = {obj.name: obj for obj in muscle_collection.objects if obj.type == 'MESH'}

    # --- 2. Pre-compute Bind Data and Mappings ---
    bind_pose_rotations = {idx: (armature_obj.matrix_world @ b.matrix_local).to_3x3() for idx, b in
                           enumerate(armature_obj.data.bones)}
    primary_bone_map = {key: wd["bone_indices"][np.argmax(wd["weights"])] for key, wd in marker_lbs_weights.items() if
                        wd and wd.get("bone_indices")}

    # --- 3. Process each frame ---
    print(f"\nProcessing {len(sparse_displacements_by_frame)} frames...")
    final_deformation_cache = {}
    sorted_frame_keys = sorted(sparse_displacements_by_frame.keys(), key=int)

    for frame_str in sorted_frame_keys:
        if int(frame_str) % 10 == 0: print(f"  Refining frame {frame_str}...")


        if int(frame_str) == FRAME_LIMIT:
            break

        bpy.context.scene.frame_set(int(frame_str))
        observed_disps_t = sparse_displacements_by_frame[frame_str]
        global_disps_t = global_displacements_by_frame.get(frame_str, {})

        # Pre-calculate A-pose world displacements for all markers involved in this frame
        all_marker_world_disps_t = {}
        for marker_key, d_local_list in global_disps_t.items():  # Use global results as the base
            primary_bone_idx = primary_bone_map.get(marker_key)
            if primary_bone_idx is not None:
                # If marker was observed, use its more accurate displacement data
                d_local_to_use = observed_disps_t.get(marker_key, d_local_list)
                R_bind = bind_pose_rotations[primary_bone_idx]
                all_marker_world_disps_t[marker_key] = R_bind @ Vector(d_local_to_use)

        deformations_for_this_frame = {}
        # --- Loop through each muscle to solve its deformation ---
        for muscle_name, muscle_obj in source_muscle_objects.items():
            L_muscle = muscle_laplacians.get(muscle_name)
            if L_muscle is None: continue

            num_verts = len(muscle_obj.data.vertices)

            # Find all markers that constrain THIS muscle
            constraints = []
            for marker_key, bary_info in marker_barycentric_map.items():
                if bary_info["muscle_name"] == muscle_name:
                    is_observed = marker_key in observed_disps_t
                    target_disp = all_marker_world_disps_t.get(marker_key)
                    if target_disp:
                        constraints.append({
                            "is_observed": is_observed,
                            "target_disp": target_disp,
                            "bary_info": bary_info
                        })

            if not constraints:
                p_unposed = np.array([muscle_obj.matrix_world @ v.co for v in muscle_obj.data.vertices])
                deformations_for_this_frame[muscle_name] = p_unposed.tolist()
                continue

            # --- Set up the linear system Ax = b for THIS muscle ---
            A_smooth = L_muscle.T @ L_muscle

            # Build Barycentric constraint matrix `B` and weighted target vector `b_data`
            num_constraints = len(constraints)
            rows, cols, data = [], [], []
            target_disps_per_axis = {0: [], 1: [], 2: []}
            constraint_weights = []

            for i, constraint in enumerate(constraints):
                target_disps_per_axis[0].append(constraint["target_disp"][0])
                target_disps_per_axis[1].append(constraint["target_disp"][1])
                target_disps_per_axis[2].append(constraint["target_disp"][2])
                constraint_weights.append(OBSERVED_DATA_WEIGHT if constraint["is_observed"] else GLOBAL_PRIOR_WEIGHT)

                for j, v_idx in enumerate(constraint["bary_info"]["vertex_indices"]):
                    rows.append(i);
                    cols.append(v_idx);
                    data.append(constraint["bary_info"]["bary_coords"][j])

            B_matrix = csc_matrix((data, (rows, cols)), shape=(num_constraints, num_verts))
            W_data = csc_matrix((np.sqrt(constraint_weights), (np.arange(num_constraints), np.arange(num_constraints))))

            A_data = (W_data @ B_matrix).T @ (W_data @ B_matrix)
            A = A_smooth + A_data

            # Solve for each axis
            dense_disps_xyz = []
            for axis in range(3):
                b_data = np.array(target_disps_per_axis[axis])
                b = (W_data @ B_matrix).T @ (W_data @ b_data)
                x = spsolve(A, b)
                dense_disps_xyz.append(x)

            D_muscle_t = np.vstack(dense_disps_xyz).T

            # Calculate final vertex positions
            P_unposed = np.array([muscle_obj.matrix_world @ v.co for v in muscle_obj.data.vertices])
            P_muscle_final_t = P_unposed + (D_muscle_t * DATA_SCALE_FACTOR)
            deformations_for_this_frame[muscle_name] = P_muscle_final_t.tolist()

        final_deformation_cache[frame_str] = deformations_for_this_frame

    # --- 4. Save the final cache file ---
    print(f"\nSaving REFINED deformation cache to: {OUTPUT_REFINED_DEFORMATION_CACHE_JSON}")
    with open(OUTPUT_REFINED_DEFORMATION_CACHE_JSON, 'w') as f:
        json.dump(final_deformation_cache, f)
    print("Save complete.")


if __name__ == '__main__':
    refine_all_muscle_deformations()