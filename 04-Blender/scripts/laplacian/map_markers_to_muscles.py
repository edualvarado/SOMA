"""
Script: map_markers_to_muscles.py
Goal: For each canonical marker point, cast a ray inwards from the skin surface
      to determine which muscle mesh it corresponds to. Saves this mapping.
"""
import bpy
import json
from mathutils import Vector

# --- User Configuration ---

# ---
shot = "shot_002"  # Change this to your shot name
# ---

# INPUT
CANONICAL_MARKERS_JSON_PATH = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/registration/canonical_model/canonical_data.json"
SKIN_MESH_OBJECT_NAME = f"canonical_skin_{shot[-3:]}"  # The mesh that markers are "on"
MUSCLE_COLLECTION_NAME = f"canonical_muscle_complex_{shot[-3:]}"  # The collection containing ALL muscle objects

# OUTPUT
OUTPUT_MARKER_TO_MUSCLE_MAP_JSON = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/mappings/marker_to_muscle_map.json"

# ----------------------------------------------------

def map_markers_to_muscles():
    print("--- Starting Marker-to-Muscle Mapping ---")

    # --- 1. Load Canonical Marker Positions ---
    print(f"Loading canonical marker data from: {CANONICAL_MARKERS_JSON_PATH}")
    try:
        with open(CANONICAL_MARKERS_JSON_PATH, 'r') as f:
            static_pose_data = json.load(f).get("0", {})
        canonical_points = {key: Vector(val[0]) for key, val in static_pose_data.items()}
    except Exception as e:
        print(f"ERROR loading canonical marker JSON: {e}");
        return
    if not canonical_points:
        print("ERROR: No marker points loaded from JSON.");
        return
    print(f"Loaded {len(canonical_points)} canonical marker positions.")

    # --- 2. Get Scene Objects ---
    skin_obj = bpy.data.objects.get(SKIN_MESH_OBJECT_NAME)
    if not skin_obj or skin_obj.type != 'MESH':
        print(f"ERROR: Skin mesh object '{SKIN_MESH_OBJECT_NAME}' not found or is not a mesh.");
        return

    muscle_collection = bpy.data.collections.get(MUSCLE_COLLECTION_NAME)
    if not muscle_collection:
        print(f"ERROR: Muscle collection '{MUSCLE_COLLECTION_NAME}' not found.");
        return

    muscle_objects = [obj for obj in muscle_collection.objects if obj.type == 'MESH']
    if not muscle_objects:
        print(f"ERROR: No mesh objects found in collection '{MUSCLE_COLLECTION_NAME}'.");
        return
    print(f"Found {len(muscle_objects)} muscle objects to test for hits.")

    # Get the evaluated dependency graph for accurate ray casting on posed/modified objects
    depsgraph = bpy.context.evaluated_depsgraph_get()

    # --- 3. Map each marker to a muscle via ray casting ---
    marker_to_muscle_map = {}
    markers_mapped_count = 0
    markers_missed_count = 0

    print("\nCasting rays from skin markers to find underlying muscles...")
    for marker_key, marker_coord_world in canonical_points.items():

        # A. Find the closest point and normal on the skin mesh surface
        # We need the evaluated object from the dependency graph for accurate results
        skin_obj_eval = skin_obj.evaluated_get(depsgraph)

        # Use object.closest_point_on_mesh to get location and normal
        # Note: This method returns values in the object's LOCAL space
        is_hit, location_local, normal_local, face_index = skin_obj_eval.closest_point_on_mesh(
            skin_obj.matrix_world.inverted() @ marker_coord_world)

        if not is_hit:
            print(f"Warning: Could not find a close point on skin for marker '{marker_key}'. Skipping.")
            markers_missed_count += 1
            continue

        # Convert normal from local space to world space by rotating it
        # We use the inverse transpose of the matrix for transforming normals
        normal_world = (skin_obj.matrix_world.to_3x3().inverted_safe().transposed() @ normal_local).normalized()

        # B. Cast a ray from the marker position INWARDS (opposite of the normal)
        ray_origin = marker_coord_world
        ray_direction = -normal_world  # Go inwards from the skin

        best_hit_distance = float('inf')
        hit_muscle_name = None

        # Test against each muscle object
        for muscle_obj in muscle_objects:
            # We need the muscle object's world matrix to transform the ray
            matrix_world_inv = muscle_obj.matrix_world.inverted()
            ray_origin_local = matrix_world_inv @ ray_origin
            ray_direction_local = matrix_world_inv.to_3x3() @ ray_direction

            muscle_obj_eval = muscle_obj.evaluated_get(depsgraph)

            # Perform the ray cast in the muscle's local space
            is_hit_muscle, location, normal, index = muscle_obj_eval.ray_cast(ray_origin_local, ray_direction_local)

            if is_hit_muscle:
                # Calculate world-space distance of the hit
                hit_location_world = muscle_obj.matrix_world @ location
                distance = (hit_location_world - ray_origin).length

                # Check if this hit is closer than any previous hit
                if distance < best_hit_distance:
                    best_hit_distance = distance
                    hit_muscle_name = muscle_obj.name

        # C. Store the result
        if hit_muscle_name:
            marker_to_muscle_map[marker_key] = hit_muscle_name
            markers_mapped_count += 1
        else:
            print(f"Warning: Ray for marker '{marker_key}' did not hit any muscle.")
            markers_missed_count += 1

    print(f"\n--- Mapping Complete ---")
    print(f"Successfully mapped {markers_mapped_count} markers to a muscle.")
    if markers_missed_count > 0:
        print(f"Could not map {markers_missed_count} markers.")

    # --- 4. Save the map to a JSON file ---
    print(f"Saving marker-to-muscle map to: {OUTPUT_MARKER_TO_MUSCLE_MAP_JSON}")
    try:
        with open(OUTPUT_MARKER_TO_MUSCLE_MAP_JSON, 'w') as f:
            json.dump(marker_to_muscle_map, f, indent=4)
        print("Save complete.")
    except Exception as e:
        print(f"ERROR writing final JSON file: {e}")


if __name__ == '__main__':
    map_markers_to_muscles()