"""
Script: map_markers_to_barycentric_coords.py
Goal: A one-time pre-computation script that finds the precise location of each
      canonical marker on its assigned muscle's surface and saves this relationship
      using barycentric coordinates.
"""

import bpy
import json
from mathutils import Vector

# --- User Configuration ---

# ---
shot = "shot_002"  # Change this to your shot name
# ---

CANONICAL_MARKERS_JSON_PATH = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/registration/canonical_model/canonical_data.json"
MARKER_TO_MUSCLE_MAP_JSON = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/mappings/marker_to_muscle_map.json"
MUSCLE_COLLECTION_NAME = f"canonical_muscle_complex_{shot[-3:]}" # Collection containing ALL muscle objects

# Path for the output file
OUTPUT_BARYCENTRIC_MAP_JSON = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/mappings/marker_barycentric_map.json"


# ---------------------------------------------------

def get_barycentric_coords_3d(point_on_tri, v0, v1, v2):
    """
    Calculates the barycentric coordinates of a point known to be on a 3D triangle.
    Returns weights corresponding to v0, v1, v2.
    """
    # Using the vector cross product method for area calculation
    total_area = ((v1 - v0).cross(v2 - v0)).length
    if total_area < 1e-9:  # Degenerate triangle
        return 1.0, 0.0, 0.0  # Default to first vertex

    # Weight for v2 is the area of the sub-triangle p,v0,v1
    w2 = ((v1 - point_on_tri).cross(v0 - point_on_tri)).length / total_area
    # Weight for v1 is the area of the sub-triangle p,v0,v2
    w1 = ((v0 - point_on_tri).cross(v2 - point_on_tri)).length / total_area
    # Weight for v0
    w0 = 1.0 - w1 - w2

    return w0, w1, w2


def create_barycentric_map_from_scene():
    print("--- Starting Marker to Muscle Barycentric Mapping from Scene ---")

    # --- 1. Load Data ---
    print("Loading data files...")
    try:
        with open(CANONICAL_MARKERS_JSON_PATH, 'r') as f:
            canonical_points_raw = json.load(f).get("0", {})
        with open(MARKER_TO_MUSCLE_MAP_JSON, 'r') as f:
            marker_to_muscle_map = json.load(f)
    except Exception as e:
        print(f"ERROR loading JSON files: {e}"); return

    canonical_points = {key: Vector(val[0]) for key, val in canonical_points_raw.items()}
    print(f"Loaded {len(canonical_points)} canonical markers and their muscle assignments.")

    # --- 2. Get Muscle Objects from the Scene Collection ---
    muscle_collection = bpy.data.collections.get(MUSCLE_COLLECTION_NAME)
    if not muscle_collection:
        print(f"ERROR: Muscle collection '{MUSCLE_COLLECTION_NAME}' not found.");
        return

    # Create a dictionary for fast lookup of muscle objects by name
    muscle_objects = {obj.name: obj for obj in muscle_collection.objects if obj.type == 'MESH'}
    if not muscle_objects:
        print(f"ERROR: No mesh objects in collection '{MUSCLE_COLLECTION_NAME}'.");
        return
    print(f"Found {len(muscle_objects)} muscle objects to process.")

    depsgraph = bpy.context.evaluated_depsgraph_get()

    # --- 3. For each marker, find its barycentric coordinates on its assigned muscle ---
    print("\nCalculating barycentric coordinates for each marker...")
    barycentric_map_data = {}

    for marker_key, p_world in canonical_points.items():
        muscle_name = marker_to_muscle_map.get(marker_key)

        if not muscle_name:
            continue

        muscle_obj = muscle_objects.get(muscle_name)
        if not muscle_obj:
            continue

        # We need the object's evaluated state from the dependency graph
        muscle_obj_eval = muscle_obj.evaluated_get(depsgraph)

        # closest_point_on_mesh needs the query point in the object's LOCAL space
        p_local = muscle_obj.matrix_world.inverted() @ p_world

        # Find the closest point and face index on the muscle mesh surface
        is_hit, location_local, _normal, face_index = muscle_obj_eval.closest_point_on_mesh(p_local)

        if not is_hit:
            print(f"Warning: Could not project marker '{marker_key}' onto muscle '{muscle_name}'. Skipping.")
            continue

        # Get the vertices of that face
        face = muscle_obj.data.polygons[face_index]
        if len(face.vertices) != 3:
            print(
                f"Warning: Face {face_index} on muscle '{muscle_name}' is not a triangle. Skipping marker '{marker_key}'.")
            continue

        v_indices = face.vertices[:]  # Get the 3 vertex indices
        v0, v1, v2 = [muscle_obj.data.vertices[i].co for i in v_indices]  # Get vertex coords in local space

        # Calculate barycentric coordinates of the hit point (also in local space)
        bary_coords = get_barycentric_coords_3d(location_local, v0, v1, v2)

        barycentric_map_data[marker_key] = {
            "muscle_name": muscle_name,
            "face_index": int(face_index),
            "vertex_indices": v_indices,
            "bary_coords": [bary_coords[0], bary_coords[1], bary_coords[2]]
        }

    print(f"Calculated barycentric mapping for {len(barycentric_map_data)} markers.")

    # --- 4. Save the map to a file ---
    print(f"Saving barycentric map to: {OUTPUT_BARYCENTRIC_MAP_JSON}")
    try:
        with open(OUTPUT_BARYCENTRIC_MAP_JSON, 'w') as f:
            json.dump(barycentric_map_data, f, indent=4)
        print("Save complete.")
    except Exception as e:
        print(f"ERROR writing JSON: {e}")


if __name__ == '__main__':
    create_barycentric_map_from_scene()