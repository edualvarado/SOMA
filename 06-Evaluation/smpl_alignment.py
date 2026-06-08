import os
import json
import numpy as np
import trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree

# ==========================================
# 1. CONFIGURATION (From your provided file)
# ==========================================
HEIGHT_ADJUSTMENT_METERS = -0.255 
FRONT_ADJUSTMENT_METERS = 0.005

HIT_TPOSE_MESH_PATH = Path(rf"/CT/SOMA/static00/frame_000000_tpose.obj")
GT_BIND_MARKERS_PATH = Path(rf"/CT/SOMA/static00/S1/canonical_model/S1_canonical_data_tpose.json")
OUTPUT_MAP_PATH = Path("soma_to_smpl_barycentric_map.json")

def _rot_x(points, deg=0.0):
    R = Rotation.from_euler('x', deg, degrees=True).as_matrix()
    return (points @ R.T).astype(np.float32)

def generate_smpl_mapping():
    print(f"--- Alignment: Height={HEIGHT_ADJUSTMENT_METERS}m, Depth={FRONT_ADJUSTMENT_METERS}m ---")
    
    # 1. Load SOMA T-Pose Markers
    with open(GT_BIND_MARKERS_PATH, 'r') as f:
        full_json = json.load(f)
    canonical_data = full_json.get("0", full_json)
    marker_ids = sorted(canonical_data.keys())
    
    p_bind = np.array([canonical_data[mid][0] for mid in marker_ids], dtype=np.float32)
    p_bind = _rot_x(p_bind, deg=-90.0)
    
    markers_to_remove = [933, 1320, 1327, 1961]
    valid_indices = [i for i in range(len(p_bind)) if i not in markers_to_remove]
    active_marker_ids = [marker_ids[i] for i in valid_indices]
    p_soma = p_bind[valid_indices]
    
    # 2. Load HIT SMPL Mesh
    hit_mesh = trimesh.load(HIT_TPOSE_MESH_PATH, process=False)
    
    # 3. Perform Centroid Alignment
    soma_centroid = p_soma.mean(axis=0)
    hit_centroid = hit_mesh.vertices.mean(axis=0)
    p_soma_aligned = p_soma - soma_centroid + hit_centroid
    
    # Apply Manual Tweaks
    p_soma_aligned[:, 1] += HEIGHT_ADJUSTMENT_METERS
    p_soma_aligned[:, 2] += FRONT_ADJUSTMENT_METERS
    
    # 4. COMPUTE BARYCENTRIC MAPPING
    print("Computing surface-to-marker mapping...")
    closest_points, distances, triangle_ids = trimesh.proximity.closest_point(hit_mesh, p_soma_aligned)
    
    triangles = hit_mesh.triangles[triangle_ids]
    bary_coords = trimesh.triangles.points_to_barycentric(triangles, closest_points)
    vertex_indices = hit_mesh.faces[triangle_ids]
    
    # 5. RECONSTRUCT VIRTUAL MARKERS (Validation)
    # We use the barycentric formula: P = w0*V0 + w1*V1 + w2*V2
    v0 = hit_mesh.vertices[vertex_indices[:, 0]]
    v1 = hit_mesh.vertices[vertex_indices[:, 1]]
    v2 = hit_mesh.vertices[vertex_indices[:, 2]]
    
    # These are the markers EXACTLY on the SMPL skin
    p_smpl_virtual = (
        bary_coords[:, 0:1] * v0 +
        bary_coords[:, 1:2] * v1 +
        bary_coords[:, 2:3] * v2
    )
    
    # 6. SAVE MAPPING JSON
    mapping_dict = {}
    for i, m_id in enumerate(active_marker_ids):
        mapping_dict[m_id] = {
            "vertex_indices": vertex_indices[i].tolist(),
            "bary_coords": [bary_coords[i].tolist()]
        }
    with open(OUTPUT_MAP_PATH, "w") as f:
        json.dump(mapping_dict, f, indent=4)
    
    # 7. EXPORT VERIFICATION PLY
    print("Exporting multi-color PLY for inspection...")
    
    # Red = Original SOMA scan markers
    soma_spheres = [trimesh.creation.icosphere(subdivisions=2, radius=0.007).apply_translation(pt) for pt in p_soma_aligned]
    soma_mesh = trimesh.util.concatenate(soma_spheres)
    soma_mesh.visual.vertex_colors = [255, 0, 0, 255] # Semi-transparent red
    
    # Blue = Virtual markers projected on SMPL skin
    smpl_spheres = [trimesh.creation.icosphere(subdivisions=2, radius=0.005).apply_translation(pt) for pt in p_smpl_virtual]
    smpl_marker_mesh = trimesh.util.concatenate(smpl_spheres)
    smpl_marker_mesh.visual.vertex_colors = [0, 0, 255, 255] # Solid blue
    
    hit_mesh.visual.vertex_colors = [200, 200, 200, 255] # Gray Body
    
    combined = trimesh.util.concatenate([hit_mesh, soma_mesh, smpl_marker_mesh])
    combined.export("final_alignment_check_multicolor.ply")
    
    print(f"✅ Mapping saved to: {OUTPUT_MAP_PATH}")
    print(f"✅ Open 'final_alignment_check_multicolor.ply'")
    print("   RED  = SOMA Scan markers")
    print("   BLUE = Virtual markers on SMPL skin (Your JSON mapping)")

if __name__ == "__main__":
    generate_smpl_mapping()