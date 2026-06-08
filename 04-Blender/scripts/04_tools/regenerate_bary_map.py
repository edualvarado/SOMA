import numpy as np
import json
import os
import trimesh
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

def _rot_x(points, deg=0.0):
    """Rotates an array of points (N,3) around the X axis."""
    if deg == 0.0: return points
    R = Rotation.from_euler('x', deg, degrees=True).as_matrix()
    return (points @ R.T)

def generate_marker_barycentric_map(p_bind, skin_layer, output_path, marker_ids, mesh_rot_x_deg=0.0):
    """
    Generate a barycentric map for markers based on the closest triangle in the skin mesh.
    """
    barycentric_map = {}

    # 1. Rotate MESH (Only if needed)
    skin_layer_rotated = skin_layer.copy()
    
    if mesh_rot_x_deg != 0.0:
        print(f"[MAP] Rotating Mesh by {mesh_rot_x_deg} deg X...")
        angle_rad = np.deg2rad(mesh_rot_x_deg)
        rotation_matrix_x = np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad), 0],
            [0, np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 0, 1]
        ])
        skin_layer_rotated.apply_transform(rotation_matrix_x)

    # 2. Build KDTree
    print("[MAP] Building KDTree...")
    kdtree = KDTree(skin_layer_rotated.vertices)

    print(f"[MAP] Processing {len(p_bind)} markers...")
    
    success_count = 0
    
    for marker_idx, marker_pos in enumerate(p_bind):
        marker_id = marker_ids[marker_idx]
        marker_pos_query = marker_pos.reshape(1, 3)

        # 3. Find Closest Point
        closest_dist, closest_vertex_index = kdtree.query(marker_pos)
        
        # 4. Check connected faces
        connected_faces = skin_layer_rotated.vertex_faces[closest_vertex_index]
        connected_faces = connected_faces[connected_faces != -1]

        face_index = None
        best_bary = None
        
        # Search for face containing the projection
        for face in connected_faces:
            v_indices = skin_layer_rotated.faces[face]
            v0, v1, v2 = skin_layer_rotated.vertices[v_indices]
            triangle = np.array([v0, v1, v2]).reshape(1, 3, 3)
            bary_coords = trimesh.triangles.points_to_barycentric(triangle, marker_pos_query)[0]

            if np.all(bary_coords >= -0.01) and np.all(bary_coords <= 1.01):
                face_index = face
                best_bary = bary_coords
                break
        
        # Fallback to closest face if projection falls outside
        if face_index is None:
            face_index = connected_faces[0]
            v_indices = skin_layer_rotated.faces[face_index]
            v0, v1, v2 = skin_layer_rotated.vertices[v_indices]
            triangle = np.array([v0, v1, v2]).reshape(1, 3, 3)
            best_bary = trimesh.triangles.points_to_barycentric(triangle, marker_pos_query)[0]

        vertex_indices = skin_layer_rotated.faces[face_index]
        
        # Sanity Check for S2 "Starburst" issue
        if np.max(np.abs(best_bary)) > 5.0:
            print(f"[ALERT] Marker {marker_id} has HUGE barycentric coords: {best_bary}. CHECK SCALING!")

        barycentric_map[marker_id] = {
            "face_index": int(face_index),
            "vertex_indices": vertex_indices.tolist(),
            "bary_coords": [best_bary.tolist()]
        }
        success_count += 1

    # Save
    with open(output_path, 'w') as f:
        json.dump(barycentric_map, f, indent=4)

    print(f"[SUCCESS] Saved to: {output_path}")

def main():
    # --- CONFIGURATION ---
    SUBJECT = "S2"  # <--- CHANGE TO "S2" AFTER RUNNING S1
    
    # Paths
    base_dir = rf"/CT/SOMA/static00/{SUBJECT}"
    mesh_path = os.path.join(base_dir, "layers", "tpose", f"skin_layer-{SUBJECT}-TPose.obj")
    
    # Handle filename difference if S1 is just 'canonical_data_tpose.json'
    if SUBJECT == "S1":
        marker_path = os.path.join(base_dir, "canonical_model", f"canonical_data_tpose.json")
    else:
        marker_path = os.path.join(base_dir, "canonical_model", f"{SUBJECT}_canonical_data_tpose.json")
    
    output_path = os.path.join(base_dir, "canonical_model", "generated_marker_barycentric_map_v2.json")

    # --- YOUR CONFIRMED SETTINGS ---
    if SUBJECT == "S1":
        # Your confirmed visual parameters:
        MARKER_SCALE = 1.0
        MARKER_ROT_X = -90.0 
        MESH_ROT_X   = 0.0   
    else:
        # S2 Settings (Starburst Fix)
        # Note: Verify S2 visually first if possible, but 0.001 is almost certainly required.
        MARKER_SCALE = 1.0
        MARKER_ROT_X = -90.0     # Assuming S2 doesn't need the -90 rotation
        MESH_ROT_X   = 0.0
        
    print(f"--- GENERATING MAP: {SUBJECT} ---")
    print(f"Scale: {MARKER_SCALE} | Marker Rot: {MARKER_ROT_X} | Mesh Rot: {MESH_ROT_X}")

    # 1. Load Mesh
    if not os.path.exists(mesh_path):
        print(f"Mesh not found: {mesh_path}"); return
    skin_layer = trimesh.load(mesh_path, process=False)

    # 2. Load Markers
    if not os.path.exists(marker_path):
        print(f"Markers not found: {marker_path}"); return
    
    with open(marker_path, 'r') as f:
        data = json.load(f)
        canonical_data = data.get("0", data)

    marker_ids = sorted(canonical_data.keys())
    
    # 3. Process Markers (Scale & Rotate)
    p_bind = []
    for mid in marker_ids:
        raw_pos = np.array(canonical_data[mid][0])
        # A. Scale
        pos = raw_pos * MARKER_SCALE
        p_bind.append(pos)
    
    p_bind = np.array(p_bind)
    
    # B. Rotate Markers
    if MARKER_ROT_X != 0.0:
        p_bind = _rot_x(p_bind, MARKER_ROT_X)

    # 4. Generate (Passes Mesh Rotation)
    generate_marker_barycentric_map(p_bind, skin_layer, output_path, marker_ids, mesh_rot_x_deg=MESH_ROT_X)

if __name__ == "__main__":
    main()