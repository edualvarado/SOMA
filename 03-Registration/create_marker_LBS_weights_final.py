"""
Script: create_marker_LBS_weights_final.py
Goal:   Estimate Linear Blend Skinning (LBS) weights for each marker corner
        point on the suit, by transferring weights from the nearest mesh
        vertices via barycentric interpolation.

Pipeline:
  1. Load the canonical marker 3D positions (from 02-Canonical-Model).
  2. Load the suit mesh (.obj) and apply a +90° X-axis rotation to align it
     with the Z-up coordinate system used by the JSON data.
  3. Load the per-vertex skin LBS weights exported from Blender.
  4. Match each mesh vertex to its Blender-exported weight entry via KDTree.
  5. For each marker corner, find the closest triangle on the mesh, compute
     barycentric coordinates, and interpolate the LBS weights from the 3
     triangle vertices.
  6. Save the resulting per-marker weights to JSON.

Inputs (all derived from --subject, overridable with --base_path):
  {base_path}/{subject}_canonical_data.json
  {base_path}/{subject}_skin_lbs_weights_exported.json
  {base_path}/skin_layer-{subject}-APose.obj

Output:
  {base_path}/{subject}_marker_lbs_weights_exported.json

Usage:
    python create_marker_LBS_weights_final.py --subject S1
    python create_marker_LBS_weights_final.py --subject S2 --base_path /custom/path
"""

import json
import argparse
import numpy as np
import trimesh
from pathlib import Path
from loguru import logger
from scipy.spatial import KDTree


def load_canonical_markers(canonical_model_path: Path):
    """Loads marker corner 3D positions from canonical_data.json.
    Returns a list of marker key strings and a (N, 3) coordinate array."""
    with open(canonical_model_path, "r") as f:
        data = json.load(f)
    static_frame = data.get("0", {})
    marker_ids = []
    marker_coords = []
    for marker_key, coord_list_of_list in static_frame.items():
        marker_ids.append(marker_key)
        marker_coords.append(coord_list_of_list[0])  # format is [[x,y,z]]
    return marker_ids, np.array(marker_coords, dtype=np.float64)


def load_and_align_mesh(mesh_obj: Path) -> trimesh.Trimesh:
    """Loads the suit mesh and rotates it +90° around X to match the Z-up JSON coordinate system."""
    suit_mesh = trimesh.load_mesh(mesh_obj, process=False)
    logger.info("Suit mesh loaded: {} vertices, {} faces.", len(suit_mesh.vertices), len(suit_mesh.faces))
    angle_rad = np.deg2rad(90.0)
    rotation_matrix_x = np.array([
        [1, 0,                 0,                0],
        [0, np.cos(angle_rad), -np.sin(angle_rad), 0],
        [0, np.sin(angle_rad),  np.cos(angle_rad), 0],
        [0, 0,                 0,                1]
    ])
    suit_mesh.apply_transform(rotation_matrix_x)
    return suit_mesh


def match_vertices_to_weights(suit_mesh: trimesh.Trimesh, skin_weights_data: list,
                               threshold: float = 1e-5) -> list:
    """
    Matches each trimesh vertex to its exported Blender weight entry using a KDTree.
    The exported JSON coordinates are already in Z-up space — no rotation is applied.
    Returns a list aligned with suit_mesh.vertices, where each entry is a skin data
    dict (bone_indices, weights, bone_names) or None if no match was found.
    """
    exported_coords = np.array([item['coords_local'] for item in skin_weights_data])
    kdtree = KDTree(exported_coords)

    vertex_to_skin_data = [None] * len(suit_mesh.vertices)
    match_count = 0

    for v_idx, v_coord in enumerate(suit_mesh.vertices):
        distance, closest_idx = kdtree.query(v_coord)
        if distance < threshold:
            entry = skin_weights_data[closest_idx]
            vertex_to_skin_data[v_idx] = {
                "bone_indices": entry["bone_indices"],
                "weights":      entry["weights"],
                "bone_names":   entry.get("bone_names", []),
            }
            match_count += 1

    logger.info("Matched {} / {} mesh vertices to exported skin data.", match_count, len(suit_mesh.vertices))
    if match_count < len(suit_mesh.vertices):
        logger.warning("{} vertices have no weight match — they will produce empty weights.",
                       len(suit_mesh.vertices) - match_count)
    return vertex_to_skin_data


def interpolate_marker_weights(suit_mesh: trimesh.Trimesh, marker_ids: list,
                                marker_coords: np.ndarray, vertex_to_skin_data: list) -> dict:
    """
    For each marker corner, finds the closest triangle on the mesh, computes barycentric
    coordinates, and interpolates the LBS weights (bone_indices, weights, bone_names)
    from the 3 triangle vertices. Weights are normalized to sum to 1.
    """
    closest_points, _, triangle_indices = suit_mesh.nearest.on_surface(marker_coords)
    triangles  = suit_mesh.triangles[triangle_indices]
    bary_coords = trimesh.triangles.points_to_barycentric(triangles=triangles, points=closest_points)

    output = {}
    success_count = 0

    for i, marker_id in enumerate(marker_ids):
        face_vertex_indices = suit_mesh.faces[triangle_indices[i]]
        bary = bary_coords[i]  # [w0, w1, w2] for the 3 triangle vertices

        vertex_data = [vertex_to_skin_data[idx] for idx in face_vertex_indices]
        if not all(vertex_data):
            output[marker_id] = {"bone_indices": [], "weights": [], "bone_names": []}
            continue

        # Accumulate barycentric-weighted contributions: {bone_idx: (accumulated_weight, bone_name)}
        accumulated = {}
        for local_idx, skin_data in enumerate(vertex_data):
            bary_w = bary[local_idx]
            for k, bone_idx in enumerate(skin_data["bone_indices"]):
                bone_name = skin_data["bone_names"][k]
                current_w, _ = accumulated.get(bone_idx, (0.0, bone_name))
                accumulated[bone_idx] = (current_w + bary_w * skin_data["weights"][k], bone_name)

        # Normalize so weights sum to 1
        total = sum(v[0] for v in accumulated.values())
        if total > 1e-6:
            bone_indices, weights, bone_names = [], [], []
            for bone_idx, (w, name) in accumulated.items():
                bone_indices.append(bone_idx)
                weights.append(w / total)
                bone_names.append(name)
            output[marker_id] = {"bone_indices": bone_indices, "weights": weights, "bone_names": bone_names}
            success_count += 1
        else:
            output[marker_id] = {"bone_indices": [], "weights": [], "bone_names": []}

    logger.info("Successfully interpolated LBS weights for {} / {} marker corners.",
                success_count, len(marker_ids))
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Compute LBS weights for suit marker corners via barycentric interpolation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--subject", type=str, required=True,
        help="Subject identifier (e.g. S1). Used to build default input/output paths."
    )
    parser.add_argument(
        "--base_path", type=Path, default=None,
        help="Override base path. Defaults to 'registration/{subject}/canonical_model'."
    )
    args = parser.parse_args()

    base = args.base_path or Path(f"registration/{args.subject}/canonical_model")
    s    = args.subject

    canonical_model_path = base / f"{s}_canonical_data.json"
    mesh_obj             = base / f"skin_layer-{s}-APose.obj"
    skin_weights_path    = base / f"{s}_skin_lbs_weights_exported.json"
    output_path          = base / f"{s}_marker_lbs_weights_exported.json"

    logger.info("==============================================")
    logger.info("== create_marker_LBS_weights_final.py      ==")
    logger.info("==============================================")
    logger.info("Subject:        {}", s)
    logger.info("Base path:      {}", base)
    logger.info("Canonical data: {}", canonical_model_path)
    logger.info("Mesh:           {}", mesh_obj)
    logger.info("Skin weights:   {}", skin_weights_path)
    logger.info("Output:         {}", output_path)

    logger.info("--- Step 1: Loading canonical marker positions...")
    marker_ids, marker_coords = load_canonical_markers(canonical_model_path)
    logger.info("Loaded {} marker corner points.", len(marker_ids))

    logger.info("--- Step 2: Loading and aligning suit mesh...")
    suit_mesh = load_and_align_mesh(mesh_obj)

    logger.info("--- Step 3: Loading per-vertex skin weights...")
    with open(skin_weights_path, "r") as f:
        skin_weights_data = json.load(f)
    logger.info("Loaded skin weights for {} mesh vertices.", len(skin_weights_data))

    logger.info("--- Step 4: Matching mesh vertices to weight data...")
    vertex_to_skin_data = match_vertices_to_weights(suit_mesh, skin_weights_data)

    logger.info("--- Step 5: Interpolating marker LBS weights...")
    marker_weights = interpolate_marker_weights(suit_mesh, marker_ids, marker_coords, vertex_to_skin_data)

    logger.info("--- Step 6: Saving results...")
    base.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(marker_weights, f, indent=4)
    logger.success("Saved marker LBS weights to: {}", output_path)


if __name__ == "__main__":
    main()
