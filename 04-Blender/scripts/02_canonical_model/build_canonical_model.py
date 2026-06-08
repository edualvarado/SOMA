"""
Script: build_canonical_model.py
Goal: Takes the UV 2D detections (like 'markers-skin-final-corrected.json') and performs in Blender Barycentric
Interpolation to build the 3D marker detection as an output and missed JSON file.

JSON structure (corners_markers and corners_charuco):

{
  "0": {
    "corners_markers": [
      [["X","Y"],["X","Y"],["X","Y"],["X","Y"]]],
    "id_markers": [[ID]]
    }
}

Example usage:
    python build_canonical_model.py --folder /CT/MUSK/static00/280424-testbench/v2-scans/inside-humans/ --board configs/suits/charuco-suit.json

Run the script with --help to see all available options.
"""

# import bpy
# from mathutils import Vector # Blender's vector type

import glob
import numpy as np
import os
import json


UV_IMAGE_WIDTH = 4096  # Replace with actual width of your UV texture image
UV_IMAGE_HEIGHT = 4096 # Replace with actual height of your UV texture image

# V-coordinate: Does your 'Y' pixel start from top (True) or bottom (False)?
INVERT_V_PIXEL_COORD = True # Common for images where (0,0) is top-left, so we need to invert height.
INVERT_U_PIXEL_COORD = False # Do not need to invert U-coordinates, as they are left in UV maps.

# Activate when running on Blender
BLENDER = True

if BLENDER:
    import bpy
    from mathutils import Vector  # Blender's vector type
    from mathutils.geometry import *

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# --- Helper function for Barycentric Coordinates (Triangle) ---
def calculate_barycentric_2d(p_uv, a_uv, b_uv, c_uv):
    """Calculates barycentric coordinates of point p_uv with respect to 2D triangle a_uv, b_uv, c_uv.
       Returns (w_a, w_b, w_c) for a, b, c respectively, or None if degenerate or outside.
       Note: This version also implicitly checks if the point is inside by ensuring weights are in [0,1]
    """
    v0 = b_uv - a_uv
    v1 = c_uv - a_uv
    v2 = p_uv - a_uv

    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-9:  # Denominator is zero or very small (degenerate triangle)
        return None

    w_b = (d11 * d20 - d01 * d21) / denom # Weight for b_uv
    w_c = (d00 * d21 - d01 * d20) / denom # Weight for c_uv
    w_a = 1.0 - w_b - w_c              # Weight for a_uv

    # Check if point is inside the triangle (including edges)
    epsilon = 1e-6 # Tolerance for floating point comparisons
    if (w_a >= (0.0 - epsilon) and w_a <= (1.0 + epsilon) and
        w_b >= (0.0 - epsilon) and w_b <= (1.0 + epsilon) and
        w_c >= (0.0 - epsilon) and w_c <= (1.0 + epsilon)):
        return w_a, w_b, w_c
    else:
        return None # Point is outside the triangle


def calculate_raw_barycentric_weights_2d(p_uv, a_uv, b_uv, c_uv):
    """
    Calculates and returns raw barycentric weights (w_a, w_b, w_c)
    for point p_uv with respect to 2D triangle a_uv, b_uv, c_uv.
    Returns None if the triangle is degenerate (denominator is too small).
    The weights correspond to vertices a_uv, b_uv, c_uv respectively.
    """
    v0 = b_uv - a_uv  # Vector from a to b
    v1 = c_uv - a_uv  # Vector from a to c
    v2 = p_uv - a_uv  # Vector from a to p

    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:  # Use a small epsilon for degeneracy check
        # print(f"DEBUG: Degenerate triangle detected for UVs: a={a_uv}, b={b_uv}, c={c_uv}. Denom: {denom}")
        return None

    # Calculate weights for b and c relative to edge (a,b) and (a,c)
    w_b_component = (d11 * d20 - d01 * d21) / denom
    w_c_component = (d00 * d21 - d01 * d20) / denom

    # The weight for a is 1 minus the other two
    w_a_final = 1.0 - w_b_component - w_c_component
    w_b_final = w_b_component
    w_c_final = w_c_component

    return w_a_final, w_b_final, w_c_final

def main():
    print("===============================")
    print("== 2. Building Blender Model ==")
    print("===============================")

    # --- 1. Load Detections from Input JSON ---
    # detection_folder = "C:/Users/ealvarad/00-Local/02-Python/Blender/data/registration/canonical_model"
    detection_folder = "S:/work/03-MUSK/02-Canonical-Model/S5/uv_detections_charuco-suit"

    print(f"Searching for JSONs in: {detection_folder}")

    markers = glob.glob(os.path.join(detection_folder, 'markers-skin-final-corrected.json'))
    markers = sorted(markers)

    output_json = os.path.join(detection_folder, "source", "canonical_model", "output.json")
    missed_json = os.path.join(detection_folder, "source", "canonical_model", "missed.json")

    print(f"output_json {output_json}")
    print(f"==========================")

    detections_uv = {}

    for markers in markers[:]:
        # Loop over the list of JSON files
        detections_uv = load_json(markers)

    # --- 2. Prepare Target UV Points from JSON Data ---
    # This dictionary will store: { ("marker_ID_corner_idx_str"): mathutils.Vector((u,v)), ... }
    target_uv_points_map = {}

    # Assuming we process frame "0" (adjust if your frame key is different)
    frame_key = "0"
    frame_data = detections_uv[frame_key]
    marker_corners_pixel_list = frame_data.get("corners_markers", [])
    marker_ids_list = frame_data.get("id_markers", [])

    # print(f"marker_corners_pixel_list: {marker_corners_pixel_list}")
    # print(f"marker_ids_list: {marker_ids_list}")

    """
    Original pixel coordinates assume origin in top-left corner of the image.
    Blender uses UV map with origin in bottom-left corner by default.
    
    marker_corners_pixel_list: [[[1465.0, 214.0], [1383.0, 208.0], [1363.0, 145.0], [1442.0, 151.0]], [[700.0, 227.0], [621.0, 219.0], [639.0, 170.0], [724.0, 176.0]],...
    marker_ids_list: [[459], [470], [436]...
    """

    if len(marker_corners_pixel_list) != len(marker_ids_list):
        print(
            f"Warning: Mismatch in lengths of 'corners_markers' ({len(marker_corners_pixel_list)}) and 'id_markers' ({len(marker_ids_list)}) lists for frame '{frame_key}'.")

    for i, pixel_corners_for_one_marker in enumerate(marker_corners_pixel_list):
        if i < len(marker_ids_list) and marker_ids_list[i] and isinstance(marker_ids_list[i], list):
            marker_id = marker_ids_list[i][0] # id_markers is a list of lists, e.g., [[ID1], [ID2]]

            if isinstance(pixel_corners_for_one_marker, list) and len(pixel_corners_for_one_marker) == 4:
                for corner_idx, corner_pixel_values in enumerate(pixel_corners_for_one_marker):
                    try:
                        # Ensure pixel values are numbers, not strings, before conversion
                        px = float(corner_pixel_values[0])
                        py = float(corner_pixel_values[1])

                        if INVERT_U_PIXEL_COORD:
                            u_normalized = 1.0 - (px / UV_IMAGE_WIDTH)
                        else:
                            u_normalized = px / UV_IMAGE_WIDTH

                        if INVERT_V_PIXEL_COORD:
                            v_normalized = 1.0 - (py / UV_IMAGE_HEIGHT)
                        else:
                            v_normalized = py / UV_IMAGE_HEIGHT

                        if marker_id == 265:
                            print(f"marker_id: {marker_id}, corner_idx: {corner_idx}, pixel: {corner_pixel_values}, normalized: ({u_normalized}, {v_normalized})")

                        if marker_id == 459:
                            print(f"marker_id: {marker_id}, corner_idx: {corner_idx}, pixel: {corner_pixel_values}, normalized: ({u_normalized}, {v_normalized})")


                        # Create a unique string identifier for this specific corner
                        feature_id_str = f"marker_{marker_id}_corner_{corner_idx}"

                        if marker_id == 265:
                            print(f"feature_id_str: {feature_id_str}")

                        if marker_id == 459:
                            print(f"feature_id_str: {feature_id_str}")


                        if BLENDER:

                            #---

                            # target_uv_points_map[feature_id_str] = Vector((u_normalized, v_normalized))

                            if feature_id_str not in target_uv_points_map:
                                target_uv_points_map[feature_id_str] = []
                            target_uv_points_map[feature_id_str].append(Vector((u_normalized, v_normalized)))

                            #---

                    except (ValueError, TypeError, IndexError) as e:
                        print(
                            f"Warning: Could not parse pixel coordinates for marker ID {marker_id}, corner index {corner_idx}. Data: {corner_pixel_values}. Error: {e}")
            else:
                print(
                    f"Warning: Marker ID {marker_id} (at original index {i}) does not have 4 valid corners. Data: {pixel_corners_for_one_marker}")
        else:
            print(f"Warning: Missing, empty, or invalid marker ID at original index {i} in 'id_markers' list.")

    if not target_uv_points_map:
        raise Exception(
            "ERROR: No valid target UV points were prepared from the JSON. Check JSON structure, paths, and parameters.")

    counter_points = 0
    for feature_id_str, target_uv in target_uv_points_map.items():
        for target in target_uv:
            counter_points += 1

    print(f"Successfully prepared {counter_points} target UV points for mapping.")
    print(f"==========================")

    if BLENDER:
        # --- 3. Create New Mesh Object ---
        # Get the active object (ensure it's the correct mesh)
        obj = bpy.context.active_object
        if not obj or obj.type != 'MESH':
            raise Exception("ERROR: The active object in Blender is not a MESH. Please select the correct suit model.")

        # Ensure we're working with evaluated geometry (considering modifiers)
        # depsgraph = bpy.context.evaluated_depsgraph_get()
        # eval_obj = obj.evaluated_get(depsgraph)
        # mesh = eval_obj.data
        # --- OR --- work directly on original data if no complex modifiers affect geometry/UVs
        mesh = obj.data

        # Get the active UV layer
        if not mesh.uv_layers.active:
            raise Exception("ERROR: The selected mesh has no active UV layer.")
        active_uv_layer = mesh.uv_layers.active.data

        # Ensure loop triangles are calculated for iterating over mesh triangles directly
        mesh.calc_loop_triangles()
        if not mesh.loop_triangles:
            print("Warning: Mesh has no loop triangles. The mesh might be empty or invalid.")

        # --- 4. Iterate Through Target UV Points and Find Corresponding 3D Points on Mesh ---
        canonical_3d_points_world_coords = {}
        missed_3d_points_world_coords = {}

        points_mapped_count = 0
        points_not_found_on_mesh_count = 0

        epsilon_for_sum_check = 1e-6  # For checking sum of weights after clamping

        for feature_id_str, target_uv in target_uv_points_map.items():
            found_on_mesh = False
            # Iterates through each triangle of the mesh
            for mesh_triangle in mesh.loop_triangles:
                try:
                    # Get UV coordinates for the three vertices of this mesh triangle
                    uv_v1 = active_uv_layer[mesh_triangle.loops[0]].uv
                    uv_v2 = active_uv_layer[mesh_triangle.loops[1]].uv
                    uv_v3 = active_uv_layer[mesh_triangle.loops[2]].uv

                    # if feature_id_str == "marker_265_corner_0":
                    #     print(f"{feature_id_str}: {target_uv}")
                    #
                    # if feature_id_str == "marker_459_corner_0":
                    #     print(f"{feature_id_str}: {target_uv}")

                    # ---

                    # is_inside = intersect_point_tri_2d(target_uv, uv_v1, uv_v2, uv_v3)

                    # ---

                    for target in target_uv:

                        is_inside = intersect_point_tri_2d(target, uv_v1, uv_v2, uv_v3)

                        if is_inside:
                            # Calculate barycentric weights for the target_uv with respect to this UV triangle
                            barycentric_weights = calculate_raw_barycentric_weights_2d(target, uv_v1, uv_v2, uv_v3)

                            """
                            if not barycentric_weights:
                                print(
                                    f"CONFLICT! Blender says point {target_uv} IS inside triangle {uv_v1, uv_v2, uv_v3}, but your function says NO.")
                            else:
                                print(
                                    f"VICTORY! Blender says point {target_uv} IS inside triangle {uv_v1, uv_v2, uv_v3}, AND your function says YES.")
                            """

                            if barycentric_weights:  # If weights are valid (point is inside or on edge)

                                if feature_id_str == "marker_265_corner_0":
                                    print(f"Match for {feature_id_str}: {target}. Found in {uv_v1}, {uv_v2}, {uv_v3}")

                                # w1, w2, w3 = barycentric_weights
                                w_a_raw, w_b_raw, w_c_raw = barycentric_weights

                                # Step 3: Clamp weights to [0,1] to handle precision issues for points on/near edges
                                w_a_clamped = max(0.0, min(1.0, w_a_raw))
                                w_b_clamped = max(0.0, min(1.0, w_b_raw))
                                w_c_clamped = max(0.0, min(1.0, w_c_raw))  # Clamp all three independently initially

                                # Step 4: Re-normalize clamped weights to ensure they sum to 1
                                # This is important if clamping changed their sum significantly
                                sum_clamped_weights = w_a_clamped + w_b_clamped + w_c_clamped

                                if sum_clamped_weights > epsilon_for_sum_check:  # Avoid division by zero
                                    w_a_final = w_a_clamped / sum_clamped_weights
                                    w_b_final = w_b_clamped / sum_clamped_weights
                                    w_c_final = w_c_clamped / sum_clamped_weights
                                elif sum_clamped_weights == 0.0 and (w_a_raw == 0.0 and w_b_raw == 0.0 and w_c_raw == 0.0):
                                    # This can happen if point is exactly one of the vertices, and others are 0.
                                    # For example, if target_uv == uv_v1, raw_weights might be (1,0,0)
                                    # Determine which vertex it's closest to if all clamped are 0 (unlikely if intersect_point_tri_2d is true)
                                    # A simpler approach for exact vertex match (raw weights would be like 1,0,0):
                                    if abs(w_a_raw - 1.0) < epsilon_for_sum_check:
                                        w_a_final, w_b_final, w_c_final = 1, 0, 0
                                    elif abs(w_b_raw - 1.0) < epsilon_for_sum_check:
                                        w_a_final, w_b_final, w_c_final = 0, 1, 0
                                    elif abs(w_c_raw - 1.0) < epsilon_for_sum_check:
                                        w_a_final, w_b_final, w_c_final = 0, 0, 1
                                    else:  # All weights very small, effectively degenerate after clamping. Skip.
                                        # print(f"DEBUG ({feature_id_str}): All clamped weights zero. Raw: {raw_weights}")
                                        continue
                                else:  # Sum of clamped weights is near zero, but not all raw were zero
                                    # This case should be rare if intersect_point_tri_2d is true.
                                    # print(f"DEBUG ({feature_id_str}): Sum of clamped weights near zero. Raw: {raw_weights}, Clamped sum: {sum_clamped_weights}")
                                    continue  # Skip this problematic case

                                # Get 3D local coordinates for the vertices of this mesh triangle
                                vertex_idx_v1 = mesh.loops[mesh_triangle.loops[0]].vertex_index
                                vertex_idx_v2 = mesh.loops[mesh_triangle.loops[1]].vertex_index
                                vertex_idx_v3 = mesh.loops[mesh_triangle.loops[2]].vertex_index

                                P1_local = mesh.vertices[vertex_idx_v1].co
                                P2_local = mesh.vertices[vertex_idx_v2].co
                                P3_local = mesh.vertices[vertex_idx_v3].co

                                # Interpolate 3D local coordinates using the barycentric weights
                                P_target_local = w_a_final * P1_local + w_b_final * P2_local + w_c_final * P3_local

                                # Convert local object coordinates to world coordinates
                                P_target_world = obj.matrix_world @ P_target_local  # '@' is matrix multiplication

                                #---

                                # canonical_3d_points_world_coords[feature_id_str] = P_target_world[:]  # Store as list [x,y,z]

                                if feature_id_str not in canonical_3d_points_world_coords:
                                    canonical_3d_points_world_coords[feature_id_str] = []
                                canonical_3d_points_world_coords[feature_id_str].append(P_target_world[:])

                                #---

                                found_on_mesh = True
                                points_mapped_count += 1

                                if points_mapped_count % 100 == 0:  # Print progress every 100 points
                                    print(f"Mapped {points_mapped_count} / {counter_points} points...")
                                break  # Found the containing triangle for this target_uv, move to the next target_uv
                            else:
                                print(
                                    f"CONFLICT! Blender says point {target_uv} is NOT inside triangle {uv_v1, uv_v2, uv_v3}, but your function says YES.")

                except IndexError:
                    # This might happen with malformed mesh data or if loop_triangles are not properly synced
                    print(
                        f"IndexError encountered while processing mesh triangle {mesh_triangle.index}. Skipping this triangle.")
                    continue

            if not found_on_mesh:
                points_not_found_on_mesh_count += 1

                # ---

                # missed_3d_points_world_coords[feature_id_str] = target_uv[:]
                missed_3d_points_world_coords[feature_id_str] = target_uv[0][:]  # Store only the first UV point for this feature ID

                # ---

                print(
                    f"Warning: Could not find a containing triangle on the mesh for feature '{feature_id_str}' with UV coordinate {target_uv}")

        print(f"\n--- Mapping Process Complete ---")
        print(f"Successfully mapped {points_mapped_count} points to 3D world coordinates.")
        if points_not_found_on_mesh_count > 0:
            print(f"Failed to find matching triangles on the mesh for {points_not_found_on_mesh_count} UV points.")

        # --- 5. Save Output JSON ---
        try:
            save_json(canonical_3d_points_world_coords, output_json)
            save_json(missed_3d_points_world_coords, missed_json)

            print(f"Successfully wrote 3D point data to: {output_json}")
            print(f"Successfully wrote 3D point data to: {missed_json}")
        except Exception as e:
            print(f"ERROR: Could not write output JSON file. Error: {e}")

if __name__ == "__main__":
    main()