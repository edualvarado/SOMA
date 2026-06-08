"""
Script: export_lbs_weights_markers.py
Goal: It takes the canonical model and motion model data, the .obj mesh of the suit, and the exported skin weights
from Blender 'exported_skin_weights.json', to estimate the weights for the marker corner points on the suit mesh:
'markers_lbs_weights_exported.json'.

"""

import json
import trimesh
import numpy as np
from pathlib import Path
from scipy.spatial import KDTree
trimesh.util.attach_to_log()

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# --- User Configuration ---

# --- MODIFIED: Define a project root to make all paths relative and portable ---
# This assumes the script is located at: .../03-MUSK/04-Blender/scripts/...
# It goes up 5 levels to find the '03-MUSK' project root directory.
project_root = Path(__file__).resolve().parents[4]
print(f"Project root identified as: {project_root}")

# SKIN WEIGHTS FOR THE NEW SKELETON (INSIDE HUMANS ADAPTED TO STUDIO)
# exported_skin_weights = project_root / "04-Blender/data/weights/canonical_model/lbs_skin/skin_lbs_weights_exported_new_skeleton.json"
exported_skin_weights = project_root / "04-Blender/data/weights/canonical_model/lbs_skin/skin_lbs_weights_exported_new_skeleton_tpose.json"

# ORIGINAL MESH IN A-POSE
# mesh_obj = project_root / "04-Blender/data/finalObj.obj"
mesh_obj = project_root / "04-Blender/data/finalObj_tpose_new.obj"

# CANONICAL DATA IN A-POSE
# canonical_model_path = project_root / "04-Blender/data/registration/canonical_model/canonical_data.json"
canonical_model_path = project_root / "04-Blender/data/registration/canonical_model/canonical_data_tpose_new.json"

# MARKERS WEIGHTS FOR THE NEW SKELETON (INSIDE HUMANS ADAPTED TO STUDIO)
# output_marker_lbs_weights = project_root / "04-Blender/data/weights/canonical_model/lbs_markers/markers_lbs_weights_exported_new_skeleton.json"
output_marker_lbs_weights = project_root / "04-Blender/data/weights/canonical_model/lbs_markers/markers_lbs_weights_exported_new_skeleton_tpose.json"

# ----------------------------------------------------

# --- 1. Load Canonical Marker Points ---
print(f"Loading canonical marker data from: {canonical_model_path}")
with open(canonical_model_path, "r") as f:
    P_scan_points_dict = json.load(f)
P_scan_points_dict_static = P_scan_points_dict.get("0", {})
p_scan_marker_coords_list = []
p_scan_marker_ids = []
for marker_key, coord_list_of_list in P_scan_points_dict_static.items():
    p_scan_marker_ids.append(marker_key)
    p_scan_marker_coords_list.append(coord_list_of_list[0])
P_scan_marker_coords_array = np.array(p_scan_marker_coords_list, dtype=np.float64)
print(f"Loaded {len(P_scan_marker_coords_array)} marker corner points.")

# --- 2. Load and Align Suit Mesh Geometry ---
print(f"Loading suit mesh geometry from: {mesh_obj}...")
suit_mesh = trimesh.load_mesh(mesh_obj, process=False)

if suit_mesh is None:
    print(f"ERROR: Failed to load the mesh from {mesh_obj}.")

print(f"Suit mesh loaded: {len(suit_mesh.vertices)} vertices, {len(suit_mesh.faces)} faces.")

angle_rad = np.deg2rad(90.0)
angle_y_rad = np.deg2rad(0.0)   # Placeholder for rotation in Y-axis
angle_z_rad = np.deg2rad(0.0)   # Placeholder for rotation in Z-axis

# Define a translation offset (x, y, z) to align the mesh
# Modify these values as needed to align your mesh
# translation_offset = np.array([0.0, 0.0, 0.0])
translation_offset = np.array([0.001799, 0.070466, 0.007204])

rotation_matrix_x = np.array([
    [1, 0, 0, 0],
    [0, np.cos(angle_rad), -np.sin(angle_rad), 0],
    [0, np.sin(angle_rad), np.cos(angle_rad), 0],
    [0, 0, 0, 1]
])

rotation_matrix_y = np.array([
    [np.cos(angle_y_rad), 0, np.sin(angle_y_rad), 0],
    [0, 1, 0, 0],
    [-np.sin(angle_y_rad), 0, np.cos(angle_y_rad), 0],
    [0, 0, 0, 1]
])

rotation_matrix_z = np.array([
    [np.cos(angle_z_rad), -np.sin(angle_z_rad), 0, 0],
    [np.sin(angle_z_rad), np.cos(angle_z_rad), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

combined_rotation_matrix = rotation_matrix_x @ rotation_matrix_y @ rotation_matrix_z

# Create a translation matrix
translation_matrix = np.eye(4)
translation_matrix[:3, 3] = translation_offset

# Combine rotation and translation into a single transformation matrix
final_transform_matrix = translation_matrix @ combined_rotation_matrix

suit_mesh.apply_transform(final_transform_matrix)

# --- 3. Load Per-Vertex Skinning Weights ---
print(f"Loading exported mesh skin weights from: {exported_skin_weights}...")
with open(exported_skin_weights, 'r') as f:
    exported_skin_weights_data_list = json.load(f)
print(f"Loaded skin weights for {len(exported_skin_weights_data_list)} mesh vertices.")

# --- 4. Match Trimesh Vertices to Exported Weight Data ---
# Apply the same alignment transform to the exported coordinates
exported_coords_np_initial = np.array([item['coords_local'] for item in exported_skin_weights_data_list])
exported_coords_homogeneous = np.hstack((exported_coords_np_initial, np.ones((exported_coords_np_initial.shape[0], 1))))
transformed_exported_coords_homogeneous = (rotation_matrix_x @ exported_coords_homogeneous.T).T
transformed_exported_coords_np = transformed_exported_coords_homogeneous[:, :3]

print("Building KDTree from TRANSFORMED exported skin data coordinates...")
kdtree_exported_verts = KDTree(transformed_exported_coords_np)
trimesh_vertex_to_skin_data = [None] * len(suit_mesh.vertices)
match_count = 0
COORDINATE_MATCH_THRESHOLD = 1e-5

print("Matching Trimesh vertices to exported skin data...")
for trimesh_v_idx, trimesh_v_coord in enumerate(suit_mesh.vertices):
    distance, closest_exported_list_idx = kdtree_exported_verts.query(trimesh_v_coord)
    if distance < COORDINATE_MATCH_THRESHOLD:
        original_blender_data = exported_skin_weights_data_list[closest_exported_list_idx]
        # --- MODIFIED: Store bone_names along with indices and weights ---
        trimesh_vertex_to_skin_data[trimesh_v_idx] = {
            "bone_indices": original_blender_data["bone_indices"],
            "weights": original_blender_data["weights"],
            "bone_names": original_blender_data.get("bone_names", []) # Use .get for safety
        }
        match_count += 1
    else:
        print(f"Vertex {trimesh_v_idx} did not match. Distance: {distance}")
        
print(f"Matched {match_count} / {len(suit_mesh.vertices)} trimesh vertices to exported skin data.")

# --- 5. Transfer Weights from Mesh to Markers via Barycentric Interpolation ---
print("Interpolating LBS weights for marker corner points...")
closest_points_on_surface, _, triangle_indices_for_markers = suit_mesh.nearest.on_surface(P_scan_marker_coords_array)
triangles_containing_markers = suit_mesh.triangles[triangle_indices_for_markers]
bary_coords_for_markers = trimesh.triangles.points_to_barycentric(
    triangles=triangles_containing_markers,
    points=closest_points_on_surface
)

marker_corner_lbs_weights_output = {}
successfully_interpolated_count = 0

for i, marker_id_str in enumerate(p_scan_marker_ids):
    face_vertex_indices_global = suit_mesh.faces[triangle_indices_for_markers[i]]
    bary = bary_coords_for_markers[i]

    vertex_data_for_face_vertices = [trimesh_vertex_to_skin_data[idx] for idx in face_vertex_indices_global]
    if not all(vertex_data_for_face_vertices):
        marker_corner_lbs_weights_output[marker_id_str] = {"bone_indices": [], "weights": [], "bone_names": []}
        continue

    # --- MODIFIED: Dictionary now stores (weight, name) tuples to keep them paired ---
    accumulated_weights_for_marker_point = {}  # {bone_idx: (accumulated_weight, bone_name)}

    for local_vert_idx, skin_data in enumerate(vertex_data_for_face_vertices):
        barycentric_weight = bary[local_vert_idx]
        bone_indices = skin_data["bone_indices"]
        weights_on_vertex = skin_data["weights"]
        bone_names_on_vertex = skin_data["bone_names"]

        for k, bone_idx in enumerate(bone_indices):
            weight_val = weights_on_vertex[k]
            bone_name = bone_names_on_vertex[k]
            
            # Accumulate weight, but store the name alongside it
            current_weight, _ = accumulated_weights_for_marker_point.get(bone_idx, (0.0, bone_name))
            new_weight = current_weight + barycentric_weight * weight_val
            accumulated_weights_for_marker_point[bone_idx] = (new_weight, bone_name)

    # --- MODIFIED: Unpack the (weight, name) tuples and normalize ---
    final_bone_indices_list = []
    final_weights_list = []
    final_bone_names_list = []
    sum_of_weights = sum(val[0] for val in accumulated_weights_for_marker_point.values())

    if sum_of_weights > 1e-6:
        for bone_idx, (total_weight, bone_name) in accumulated_weights_for_marker_point.items():
            final_bone_indices_list.append(bone_idx)
            final_weights_list.append(total_weight / sum_of_weights)
            final_bone_names_list.append(bone_name)

        marker_corner_lbs_weights_output[marker_id_str] = {
            "bone_indices": final_bone_indices_list,
            "weights": final_weights_list,
            "bone_names": final_bone_names_list, # Add the new field
        }
        successfully_interpolated_count += 1
    else:
        marker_corner_lbs_weights_output[marker_id_str] = {"bone_indices": [], "weights": [], "bone_names": []}

print(f"Successfully interpolated LBS weights for {successfully_interpolated_count} marker points.")

# --- 6. Save the Calculated Marker LBS Weights ---
save_json(marker_corner_lbs_weights_output, output_marker_lbs_weights)
print(f"Successfully saved LBS weights for marker corners to: {output_marker_lbs_weights}")