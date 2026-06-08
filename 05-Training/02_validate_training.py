import os
import glob
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from tqdm import tqdm
from typing import Optional

import trimesh
from trimesh.smoothing import laplacian_calculation

from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree
import scipy.sparse as sp

import viser
import viser.transforms as tf
from pymotion.io.bvh import BVH
from pymotion.ops.skeleton import fk
import pymotion.rotations.ortho6d as sixd
import pymotion.rotations.quat as quat
import pymotion.rotations.quat_torch as quat_torch
import pymotion.ops.skeleton as sk

def save_obj_fast(filepath, verts, faces):
    """Quickly exports an OBJ file without freezing the UI."""
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.export(filepath)

# --- CONFIGURATION (Match your training config) ---
# Centralized config to keep parameters organized
CONFIG = {
    "subject": "S1",
    # Paths (We will build absolute paths dynamically below)
    "base_path_suffix": r"static00", 
    "shot": "shot_004",
    # System
    "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

# CHECKPOINTS #
# ----------- #
# Add new entries below. Change ACTIVE_CHECKPOINT to switch between them.

CHECKPOINTS = {
    # Stage 1: E_data + E_smooth only
    "smoothness": {
        "path": "./checkpoints/01_adding_smoothness/full_MLP_w_data_1_w_smooth_musc_1_w_smooth_skin_5_w_spring_musc_0_w_spring_skin_0_w_tangent_musc_0_w_tangent_skin_0_epoch_9.pth",
        "result": "Deformation visible but spiky and noisy",
    },
    # Stage 2: + E_spring (stretch resistance)
    "stretch": {
        "path": "./checkpoints/02_adding_stretch/full_MLP_w_data_1_w_smooth_musc_005_w_smooth_skin_05_w_spring_musc_1_w_spring_skin_1_w_tangent_musc_0_w_tangent_skin_0_epoch_6.pth",
        "result": "Best so far at this stage",
    },
    # Stage 3: + E_tangential
    "tangential": {
        "path": "./checkpoints/03_adding_tan/full_MLP_w_data_1_w_smooth_musc_0005_w_smooth_skin_005_w_spring_musc_1_w_spring_skin_1_w_tangent_musc_005_w_tangent_skin_001_epoch_9.pth",
        "result": "Accordion artifact — needs fix",
    },
    # Stage 4: + E_biharmonic (equal_weight=True)
    "biharmonic": {
        "path": "./checkpoints/04_bi_new/full_MLP_w_data_1_w_smooth_musc_0005_w_smooth_skin_005_w_spring_musc_1_w_spring_skin_1_w_tangent_musc_005_w_tangent_skin_001_w_biharmonic_musc_1_w_biharmonic_skin_25_epoch_6.pth",
        "result": "ep5: good (more wrinkles?), ep6: good, ep7: last good",
    },
    # Stage 5a: tuning tangential weights (tan_skin=1)
    "tan_change_a": {
        "path": "./checkpoints/05_changing_tan/full_MLP_w_data_1_w_smooth_musc_0005_w_smooth_skin_005_w_spring_musc_1_w_spring_skin_1_w_tangent_musc_1_w_tangent_skin_1_w_biharmonic_musc_1_w_biharmonic_skin_25_epoch_6.pth",
        "result": "Good",
    },
    # Stage 5b: tuning tangential weights (tan_skin=0.1)
    "tan_change_b": {
        "path": "./checkpoints/05_changing_tan/full_MLP_w_data_1_w_smooth_musc_0005_w_smooth_skin_005_w_spring_musc_1_w_spring_skin_1_w_tangent_musc_1_w_tangent_skin_01_w_biharmonic_musc_1_w_biharmonic_skin_25_epoch_6.pth",
        "result": "Good",
    },
    # Stage 5c: tuning smooth + tangential (vol=0) — used for visualization
    "tan_change_c": {
        "path": "./checkpoints/05_changing_tan/full_MLP_w_data_1_w_smooth_musc_001_w_smooth_skin_005_w_spring_musc_1_w_spring_skin_1_w_tangent_musc_1_w_tangent_skin_01_w_biharmonic_musc_1_w_biharmonic_skin_25_w_vol_musc_0_w_vol_skin_0_epoch_6.pth",
        "result": "Works — used for visualization",
    },
    # Stage 6a: + E_vol on skin only (w_data=50)
    "volume_skin_50": {
        "path": "./checkpoints/06_adding_volume/full_MLP_w_data_50_w_smooth_musc_100_w_smooth_skin_200_w_spring_musc_1000_w_spring_skin_1000_w_tangent_musc_1_w_tangent_skin_01_w_biharmonic_musc_100_w_biharmonic_skin_200_w_vol_musc_0_w_vol_skin_10_epoch_5.pth",
        "result": "Best with volume so far",
    },
    # Stage 6b: + E_vol on skin only (w_data=500)
    "volume_skin_500": {
        "path": "./checkpoints/06_adding_volume/full_MLP_w_data_500_w_smooth_musc_100_w_smooth_skin_200_w_spring_musc_1000_w_spring_skin_1000_w_tangent_musc_1_w_tangent_skin_01_w_biharmonic_musc_100_w_biharmonic_skin_200_w_vol_musc_0_w_vol_skin_10_epoch_6.pth",
        "result": "Best with volume (w_data=500)",
    },
    # Stage 7: + E_vol on both muscle and skin
    "volume_both": {
        "path": "./checkpoints/07_adding_volume_both/full_MLP_w_data_50_w_smooth_musc_100_w_smooth_skin_200_w_spring_musc_1000_w_spring_skin_1000_w_tangent_musc_1_w_tangent_skin_01_w_biharmonic_musc_100_w_biharmonic_skin_200_w_vol_musc_10_w_vol_skin_10_epoch_5.pth",
        "result": "Final",
    },
    # Stage 8: + E_vol on both, with mask
    "volume_both_mask": {
        "path": "./checkpoints/08_adding_volume_both_clean/full_MLP_w_data_50_w_smooth_musc_100_w_smooth_skin_200_w_spring_musc_1000_w_spring_skin_1000_w_tangent_musc_1_w_tangent_skin_01_w_biharmonic_musc_100_w_biharmonic_skin_200_w_vol_musc_10_w_vol_skin_10_epoch_5.pth",
        "result": "Final (with mask)",
    }
}

# ------------------------------------------------------------------
ACTIVE_CHECKPOINT = "volume_both_mask"  # <-- Change this to switch checkpoints
CHECKPOINT_PATH = CHECKPOINTS[ACTIVE_CHECKPOINT]["path"]
# ------------------------------------------------------------------

# --- PATH SETUP ---
# Adjust this to match your cluster structure
BASE_DIR = rf"/CT/SOMA/{CONFIG['base_path_suffix']}/{CONFIG['subject']}"

PATHS = {
    "raw": os.path.join(BASE_DIR, "raw"),
    "processed": os.path.join(BASE_DIR, "preprocessed_vFinal_clean"),
    "layers_tpose": os.path.join(BASE_DIR, "layers", "tpose"),
    "canonical": os.path.join(BASE_DIR, "canonical_model"),
}

# ==============================================================================
# 1. UTILITY FUNCTIONS (Copied from Training Script for Consistency)
# ==============================================================================

def move_tensors_to_device(tensor_dict, device):
    """
    Moves a dictionary of tensors to the specified device.
    Gracefully handles None values and non-tensor types.
    
    Args:
        tensor_dict (dict): Dictionary of {name: tensor} pairs
        device: PyTorch device (cuda, cpu, etc.)
    
    Returns:
        dict: Dictionary with all PyTorch tensors moved to device
    
    Example:
        tensors = {
            'vertices': vertex_tensor,
            'weights': weight_tensor,
            'mask': None  # Skipped
        }
        gpu_tensors = move_tensors_to_device(tensors, device)
    """
    device_dict = {}
    for key, value in tensor_dict.items():
        if value is not None and isinstance(value, torch.Tensor):
            device_dict[key] = value.to(device)
        else:
            device_dict[key] = value
    return device_dict

def calculate_vertex_mass(trimesh_mesh, device):
    """
    Calculates the per-vertex area (Mass Matrix diagonal) for a Trimesh object.
    Uses Barycentric Area (1/3 of incident face areas) for stability.
    """
    # 1. Get Face Areas from Trimesh (make a copy to avoid read-only warning)
    face_areas = torch.from_numpy(trimesh_mesh.area_faces.copy()).float().to(device) # (F,)
    
    # 2. Get Faces
    faces = torch.from_numpy(trimesh_mesh.faces).long().to(device) # (F, 3)
    
    # 3. Scatter add face areas to vertices
    num_verts = len(trimesh_mesh.vertices)
    vertex_areas = torch.zeros(num_verts, device=device)
    
    # Add 1/3 of face area to each of the 3 vertices
    val = face_areas / 3.0
    
    # Scatter for each column of the faces
    vertex_areas.scatter_add_(0, faces[:, 0], val)
    vertex_areas.scatter_add_(0, faces[:, 1], val)
    vertex_areas.scatter_add_(0, faces[:, 2], val)
    
    return vertex_areas

def compute_vertex_stability_mask(m_bind, s_bind, faces):
    """
    Creates a (V, 1) mask where:
    1.0 = Vertex is part of a valid, volumetric body part.
    0.0 = Vertex is part of a degenerate area (face/hands) and should be PINNED.
    """
    device = m_bind.device
    num_faces = faces.shape[0]
    num_verts = m_bind.shape[0]
    
    # 1. Gather Prism Vertices
    m0 = m_bind[faces[:, 0]]; m1 = m_bind[faces[:, 1]]; m2 = m_bind[faces[:, 2]]
    s0 = s_bind[faces[:, 0]]; s1 = s_bind[faces[:, 1]]; s2 = s_bind[faces[:, 2]]
    
    P = torch.stack([m0, m1, m2, s0, s1, s2], dim=2) # (F, 3, 6)
    
    # 2. Compute Shape Matrix S (Determinant check)
    # ... (Reuse the logic from your Loss function) ...
    # Simplified determinat calculation for speed:
    means = P.mean(dim=2, keepdim=True)
    P_bar = P - means
    
    # Weight Matrix W (Standard Centering)
    W = torch.zeros((6, 6), device=device)
    W.fill_diagonal_(4.0)
    edges_idx = [(0,1), (1,2), (2,0), (3,4), (4,5), (5,3), (0,3), (1,4), (2,5)]
    for r, c in edges_idx: W[r,c] = W[c,r] = 2.0
    W_batch = W.unsqueeze(0).expand(num_faces, -1, -1)
    
    S = torch.bmm(torch.bmm(P_bar, W_batch), P_bar.transpose(1, 2))
    
    # 3. Identify Valid Prisms
    dets = torch.det(S).abs()
    valid_prism_mask = dets > 0 # (F,) boolean: Threshold to KEEP HEAD/HANDS UNTOUCHED
    
    # 4. Map Valid Prisms to Vertices
    # Start with all zeros (assume everything is bad)
    vertex_mask = torch.zeros(num_verts, 1, device=device)
    
    # Get all unique vertex indices that belong to valid faces
    valid_faces = faces[valid_prism_mask] # (N_valid, 3)
    valid_indices = torch.unique(valid_faces)
    
    # Set those vertices to 1.0
    vertex_mask[valid_indices] = 1.0
    
    return vertex_mask

def compute_edge_weights(trimesh_mesh):
    """
    Calculates area-based weights for edges to ensure discretization invariance.
    Formula: weight_edge = (area_face_A + area_face_B) / 3
    
    Args:
        trimesh_mesh: The loaded Trimesh object.
        device: Torch device.
    Returns:
        edge_weights: (Num_Edges,) float tensor, normalized to sum to 1.0 (optional but recommended).
    """
    # 1. Get Face Areas (F,)
    face_areas = trimesh_mesh.area_faces
    
    # 2. Get the mapping from Faces to Unique Edges (F, 3)
    # This gives us the 3 global edge indices for every face
    face_edges = trimesh_mesh.faces_unique_edges
    
    # 3. Accumulate Area to Edges
    num_edges = len(trimesh_mesh.edges_unique)
    edge_weights_np = np.zeros(num_edges, dtype=np.float32)
    
    # Iterate over the 3 edges of every face
    # We add area/3 to each incident edge. 
    # Since an internal edge is shared by 2 faces, it will sum (area_A/3 + area_B/3).
    # This perfectly matches the paper formula.
    val = face_areas / 3.0
    
    # We use numpy.add.at for unbuffered summation (like scatter_add)
    # Flatten face_edges to (F*3,) and repeat val 3 times
    np.add.at(edge_weights_np, face_edges.flatten(), np.repeat(val, 3))
    
    # 4. Normalize (Optional, but good for learning rate stability)
    # We normalize so the sum of weights is 1.0 (or num_edges, depending on preference).
    # Using Sum=1 makes the loss independent of mesh resolution.
    sum_weights = np.sum(edge_weights_np) + 1e-8
    edge_weights_np /= sum_weights
    
    return torch.from_numpy(edge_weights_np).float()

def compute_edge_lengths(verts, edges):
    """Computes the length of every edge in the mesh."""
    # Ensure edges is a writable array
    if isinstance(edges, np.ndarray) and not edges.flags.writeable:
        edges = edges.copy()
    p1 = verts[:, edges[:, 0], :] 
    p2 = verts[:, edges[:, 1], :]
    return torch.norm(p1 - p2, dim=-1)

def compute_mesh_properties(trimesh_obj, device):
    """
    Computes all geometric properties for a mesh (edges, areas, normals, Laplacian).
    
    Args:
        trimesh_obj: Trimesh object
        device: PyTorch device
    
    Returns:
        Dictionary containing:
            - 'vertices_np': (V, 3) numpy vertices
            - 'faces_np': (F, 3) numpy faces  
            - 'vertices': (V, 3) torch tensor
            - 'faces': (F, 3) torch tensor
            - 'edges': (E, 2) numpy edge indices
            - 'edge_rest_lengths': (E,) numpy edge rest lengths
            - 'edge_weights': (E,) torch normalized edge weights
            - 'face_rest_areas': (F,) torch face areas
            - 'normals': (V, 3) torch vertex normals
            - 'vertex_mass': (V,) torch vertex areas
            - 'laplacian_sp': scipy sparse Laplacian matrix
            - 'laplacian': torch sparse Laplacian tensor
            - 'laplacian_degree': (V,) torch Laplacian diagonal
    """
    
    # --- BASIC GEOMETRY ---
    vertices_np = trimesh_obj.vertices.astype(np.float32)
    faces_np = trimesh_obj.faces.astype(np.int32)
    
    vertices = torch.from_numpy(vertices_np).float()
    faces = torch.from_numpy(faces_np).long()
    
    # --- EDGES ---
    edges_np = trimesh_obj.edges_unique  # (E, 2)
    
    # Calculate rest edge lengths
    v0 = vertices_np[edges_np[:, 0]]
    v1 = vertices_np[edges_np[:, 1]]
    edge_rest_lengths = np.linalg.norm(v0 - v1, axis=1).astype(np.float32)
    
    # Calculate edge weights (area-based)
    edge_weights = compute_edge_weights(trimesh_obj)
    
    # --- FACE AREAS ---
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    e1 = v1 - v0
    e2 = v2 - v0
    
    cross_prod = torch.cross(e1, e2, dim=1)
    face_rest_areas = 0.5 * torch.norm(cross_prod, dim=1)
    
    # --- VERTEX NORMALS ---
    normals_np = trimesh_obj.vertex_normals.astype(np.float32)
    normals = torch.from_numpy(normals_np).float()
    
    # --- VERTEX MASS (Barycentric areas) ---
    vertex_mass = calculate_vertex_mass(trimesh_obj, device)
    
    # --- LAPLACIAN ---
    laplacian_sp = laplacian_calculation(trimesh_obj, equal_weight=False)
    
    # Convert to Sparse Torch Tensor
    indices = np.vstack((laplacian_sp.tocoo().row, laplacian_sp.tocoo().col))
    values = laplacian_sp.tocoo().data
    laplacian = torch.sparse_coo_tensor(
        torch.from_numpy(indices).long(),
        torch.from_numpy(values).float(),
        torch.Size(laplacian_sp.shape)
    ).coalesce()
    
    # Extract diagonal (degree) with safety clamp
    laplacian_degree_np = laplacian_sp.tocsr().diagonal()
    laplacian_degree_np = np.maximum(laplacian_degree_np, 1.0)  # Prevent division by zero
    laplacian_degree = torch.from_numpy(laplacian_degree_np).float()
    
    return {
        'vertices_np': vertices_np,
        'faces_np': faces_np,
        'vertices': vertices,
        'faces': faces,
        'edges': edges_np,
        'edge_rest_lengths': edge_rest_lengths,
        'edge_weights': edge_weights,
        'face_rest_areas': face_rest_areas,
        'normals': normals,
        'vertex_mass': vertex_mass,
        'laplacian_sp': laplacian_sp,
        'laplacian': laplacian,
        'laplacian_degree': laplacian_degree
    }

def get_rest_joint_locations(bvh_obj, scale=1.0):
    """
    Calculates the global joint positions for the rest pose from the BVH skeleton.
    """
    _, _, parents, offsets, _, _ = bvh_obj.get_data()

    # Set root to -1
    parents[0] = -1

    for i, parent in enumerate(parents):
        if parent == i:
            raise ValueError(f"Joint {i} is its own parent!")
        if parent >= len(parents):
            raise ValueError(f"Invalid parent index {parent} for joint {i}!")

    offsets *= scale
    j_rest = np.zeros_like(offsets)
    for i in range(len(parents)):
        if parents[i] == -1:
            j_rest[i] = offsets[i]
            # print("Joint", i, "is root offset", offsets[i], "rest pos", j_rest[i])
        else:
            j_rest[i] = j_rest[parents[i]] + offsets[i]
            # print("Joint", i, "parent", parents[i], "offset", offsets[i], "rest pos", j_rest[i])

    return j_rest, parents, offsets

def get_rest_joint_locations_zero_offset(bvh_obj, scale=1.0):
    """
    Calculates the global joint positions for the rest pose from the BVH skeleton.
    The root joint offset is set to zero, so the root pivot is at the origin.
    """
    _, _, parents, offsets, _, _ = bvh_obj.get_data()

    # Set root to -1
    parents[0] = -1

    for i, parent in enumerate(parents):
        if parent == i:
            raise ValueError(f"Joint {i} is its own parent!")
        if parent >= len(parents):
            raise ValueError(f"Invalid parent index {parent} for joint {i}!")

    # Make a local copy of offsets and zero the root (hips) rest offset so the root pivot is at the origin
    # This is done such as the hips has a zero offset in the rest pose
    offsets = offsets.copy()
    if offsets.shape[0] > 0:
        offsets[0] = np.zeros(3, dtype=offsets.dtype)

    offsets *= scale
    j_rest = np.zeros_like(offsets)
    for i in range(len(parents)):
        if parents[i] == -1:
            j_rest[i] = offsets[i]
            # print("Joint", i, "is root offset", offsets[i], "rest pos", j_rest[i])
        else:
            j_rest[i] = j_rest[parents[i]] + offsets[i]
            # print("Joint", i, "parent", parents[i], "offset", offsets[i], "rest pos", j_rest[i])

    return j_rest, parents, offsets

def sixd_to_rotmat(sixd_reps):
    x_raw = sixd_reps[..., 0:3]
    y_raw = sixd_reps[..., 3:6]
    x = F.normalize(x_raw, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    return torch.stack((x, y, z), dim=-1)

def lbs_working_batch_rotmat(vertices, rot_mats, weights, j_rest, parents, root_position):
    batch_size = vertices.shape[0]
    num_joints = j_rest.shape[0]
    device = vertices.device
    weights = weights.to(vertices.dtype)
    j_rest = j_rest.to(vertices.dtype)
    root_position = root_position.to(vertices.dtype)

    ident4 = torch.eye(4, device=device, dtype=vertices.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    G_rest = torch.zeros(batch_size, num_joints, 4, 4, device=device, dtype=vertices.dtype)
    for i in range(num_joints):
        if parents[i] == -1:
            T = ident4.clone()
            T[:, :3, 3] = j_rest[i]
        else:
            off = (j_rest[i] - j_rest[parents[i]]).unsqueeze(0)
            T_off = ident4.clone()
            T_off[:, :3, 3] = off
            T = G_rest[:, parents[i]] @ T_off
        G_rest[:, i] = T
    G_rest_inv = torch.inverse(G_rest)

    G_posed = torch.zeros_like(G_rest)
    for i in range(num_joints):
        T_local = ident4.clone()
        T_local[:, :3, :3] = rot_mats[:, i]
        if parents[i] == -1:
            T_local[:, :3, 3] = root_position
            G_posed[:, i] = T_local
        else:
            T_local[:, :3, 3] = (j_rest[i] - j_rest[parents[i]]).unsqueeze(0)
            G_posed[:, i] = G_posed[:, parents[i]] @ T_local

    skinning = G_posed @ G_rest_inv
    homo = torch.cat([vertices, torch.ones(batch_size, vertices.shape[1], 1, device=device, dtype=vertices.dtype)], dim=-1)
    blended = torch.einsum('vj,bjmn->bvmn', weights, skinning)
    out = blended @ homo.unsqueeze(-1)
    return out[:, :, :3, 0]

def barycentric_interpolation_batch(deformed_vertices, bary_verts, bary_weights):
    v0 = deformed_vertices[:, bary_verts[:, 0], :] 
    v1 = deformed_vertices[:, bary_verts[:, 1], :]
    v2 = deformed_vertices[:, bary_verts[:, 2], :]
    interpolated_positions = (
        bary_weights[:, 0].unsqueeze(0).unsqueeze(-1) * v0 +
        bary_weights[:, 1].unsqueeze(0).unsqueeze(-1) * v1 +
        bary_weights[:, 2].unsqueeze(0).unsqueeze(-1) * v2
    )
    return interpolated_positions

def convert_weights_to_npy(json_path, num_rows, bvh_joint_names):
    # Simplified loader for inference
    with open(json_path, 'r') as f:
        weights_data = json.load(f)
    num_joints = len(bvh_joint_names)
    joint_name_to_index = {name: i for i, name in enumerate(bvh_joint_names)}
    weights_matrix = np.zeros((num_rows, num_joints), dtype=np.float32)
    for v_idx, vertex_info in enumerate(weights_data):
        if v_idx >= num_rows: continue
        bone_names = vertex_info.get("bone_names", [])
        weights = vertex_info.get("weights", [])
        for bone_name, weight in zip(bone_names, weights):
            if bone_name in joint_name_to_index:
                weights_matrix[v_idx, joint_name_to_index[bone_name]] = weight
    return weights_matrix

def get_inverse_bind_matrices(j_rest):
    """
    Calculates the Inverse Bind Matrices (IBMs) for the skinned mesh.
    Since our rest pose has Identity rotation, the IBM is just the inverse translation.
    j_rest: (J, 3) tensor or numpy
    """
    if torch.is_tensor(j_rest):
        j_rest = j_rest.detach().cpu().numpy()
        
    num_joints = j_rest.shape[0]
    ibms = np.eye(4)[None].repeat(num_joints, 0) # (J, 4, 4)
    
    # The IBM transforms from Mesh Space (World) to Bone Space.
    # Since Bone Space at rest is just translated by j_rest, the inverse is -j_rest.
    ibms[:, :3, 3] = -j_rest
    return ibms

def load_obj_simple(file_path):
    vertices = []
    faces = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append([float(i) for i in line.strip().split()[1:]])
            elif line.startswith('f '):
                face = [int(i.split('/')[0]) - 1 for i in line.strip().split()[1:]]
                faces.append(face)
    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)

def load_and_stack_muscles(muscle_dir, file_pattern="*.obj"):
    """
    Loads individual muscle meshes, stacks them into a single vertex/face array 
    for the model, and builds a Block-Diagonal Laplacian for regularization.
    
    Returns:
        merged_vertices (np.array): (Total_V, 3)
        merged_faces (np.array): (Total_F, 3)
        L_block_diag (scipy.sparse): (Total_V, Total_V) Block Diagonal Matrix
        muscle_masks (dict): {muscle_name: (start_index, end_index)}
    """
    mesh_files = sorted(glob.glob(os.path.join(muscle_dir, file_pattern)))
    if not mesh_files:
        raise ValueError(f"No muscle files found in {muscle_dir}")

    all_vertices = []
    all_faces = []
    laplacians = []
    muscle_vertex_ranges = {} # To store start/end indices for each muscle
    muscle_face_ranges = {} # To store start/end indices for each muscle

    vertex_offset = 0
    face_offset = 0

    logger.info(f"[MESH] FOUND {len(mesh_files)} INDEPENDENT MUSCLES. STACKING...")

    for f_path in mesh_files:
        muscle_name = os.path.basename(f_path).split('.')[0]

        # Load individual mesh
        # [FIX] We MUST merge vertices. 
        # Raw OBJs have duplicate vertices at UV seams. If we don't weld them,
        # the NN lookup picks one, moves it, and leaves the duplicate behind, tearing the mesh.
        mesh = trimesh.load(f_path, process=False)
        mesh.merge_vertices()
                
        # 1. Store Vertices
        current_verts = mesh.vertices
        all_vertices.append(current_verts)
        
        # 2. Store Faces (offset by current vertex count to keep unique indices)
        all_faces.append(mesh.faces + vertex_offset)
        
        # 3. Calculate Independent Laplacian (Uniform is safer)
        # This Laplacian ONLY knows about this specific muscle.
        # [FIX] Use cotangent weights if possible for better physics, but uniform is stable
        L_sub = laplacian_calculation(mesh, equal_weight=False)
        laplacians.append(L_sub)
        
        # 4. Store Ranges for VERTICES
        num_verts = len(current_verts)
        muscle_vertex_ranges[muscle_name] = (vertex_offset, vertex_offset + num_verts)

        # 5. Store Ranges for FACES
        num_faces = len(mesh.faces)
        muscle_face_ranges[muscle_name] = (face_offset, face_offset + num_faces)
        
        logger.debug(f"[MESH] Added {muscle_name}: vertices [{vertex_offset}, {vertex_offset + num_verts}]!")
        
        vertex_offset += num_verts
        face_offset += num_faces

    # --- MERGE FOR MODEL ---
    # Stack all vertices into (Total_V, 3)
    merged_vertices = np.vstack(all_vertices).astype(np.float32)
    # Stack all faces into (Total_F, 3)
    merged_faces = np.vstack(all_faces).astype(np.int32)
    
    # --- MERGE FOR REGULARIZATION ---
    # Create a Block Diagonal Matrix
    # [ L_1  0   0  ]
    # [  0  L_2  0  ]
    # [  0   0  L_3 ]
    # This ensures smoothness can NEVER propagate between muscles.
    L_block_diag = sp.block_diag(laplacians, format='coo')
    
    return merged_vertices, merged_faces, L_block_diag, muscle_vertex_ranges, muscle_face_ranges
    
def _rot_x(points, deg=0.0):
    R = Rotation.from_euler('x', deg, degrees=True).as_matrix()
    return (points @ R.T)

# ==============================================================================
# 2. MODEL DEFINITION
# ==============================================================================

# --- LINEAR BLENDSHAPES CLASS ---
class MuscleBlendshapeModel(nn.Module):
    def __init__(self, num_vertices, num_joints=24):
        super().__init__()
        self.num_vertices = num_vertices # 23752
        self.num_joints = num_joints # 24

        # --- The Learnable Parameter Matrix P_muscle ---
        # Dimensions: (3 * V) x (9 * (J - 1))
        # Rows: Flattened vertex displacements (x1, y1, z1, x2...)
        # Cols: Flattened rotation matrix elements (r11, r12, r13...) for all joints (excluding root)
        
        # We initialize with ZEROS. 
        # Why? Because in the Rest Pose (Identity Rotation), R - I = 0.
        # We want the output displacement to be exactly 0 at the start.
        self.P_muscle = nn.Parameter(torch.zeros(num_vertices * 3, 9 * (num_joints - 1))) # Shape: (23752 * 3, 207)

        # --- Skin Residual Matrix (New Layer) ---
        # Allows the skin to have detail/sliding independent of the muscle bulk.
        # Initialized to 0 so we start with Skin == Muscle.
        self.P_skin = nn.Parameter(torch.zeros(num_vertices * 3, 9 * (num_joints - 1))) # Shape: (23752 * 3, 207)

    def forward(self, rot_mats, rot_identity):
        """
        Args:
            rot_mats: Input poses. Supports (B,J,3,2) [6D], (B,J,6) [Flat 6D], or (B,J,3,3) [Matrix]
            rot_identity: The Identity Rotation Matrix (B, J, 3, 3) used for subtraction (R - I)
        """

        # --- 0. FIX: Handle Flattened Input (B, 144) ---
        if rot_mats.dim() == 2:
            # Reshape (B, 144) -> (B, 24, 6)
            batch_size = rot_mats.shape[0]
            rot_mats = rot_mats.view(batch_size, self.num_joints, 6)

        # 1. Input Normalization -> Ensure we have (B, J, 3, 3)
        if rot_mats.dim() == 4 and rot_mats.shape[-2:] == (3, 2):
            a = rot_mats[..., 0]
            b = rot_mats[..., 1]
            rot6d = torch.cat([a, b], dim=-1)          # (B,J,6)
            R = sixd_to_rotmat(rot6d)                  # (B,J,3,3)
        elif rot_mats.dim() == 3 and rot_mats.shape[-1] == 6:
            R = sixd_to_rotmat(rot_mats)
        elif rot_mats.dim() == 4 and rot_mats.shape[-2:] == (3, 3):
            R = rot_mats
        else:
            raise ValueError(f"Unsupported rot_mats shape {rot_mats.shape}; expected (B,J,3,2), (B,J,6), or (B,J,3,3)")

        batch_size = R.shape[0] # e.g. 32
        # print("R.shape:", R.shape)

        # 2. Compute Relative Rotations (R - I)
        # We skip Joint 0 (Root) as blendshapes are usually driven by local pose, not global orientation.
        # Shape: (B, J-1, 3, 3)
        rel = (R[:, 1:, :, :] - rot_identity[:, 1:, :, :]) # (B, J-1, 3, 3)

        # 3. Flatten Inputs for Matrix Multiplication
        # Shape: (B, 207) -> 9 elements * 23 joints
        r_star = rel.reshape(batch_size, -1) # (B, 9*(J-1))\
        # print("rel.shape:", rel.shape)
        # print("r_star.shape:", r_star.shape)

        # 4. Calculate Muscle Displacement
        # Eq: D = P * r_star^T
        # P is (V*3, Params). r_star is (B, Params).
        # We transpose r_star to (Params, B) for the math.
        # Result: (V*3, B) -> Transpose back to (B, V*3)
        disp_muscle = self.P_muscle @ r_star.T # (V*3, B)
        disp_muscle = disp_muscle.T.reshape(batch_size, self.num_vertices, 3) # (B, V, 3)
        # print("disp.shape:", disp.shape)

        # 5. Calculate Skin Residual Displacement
        # Same logic, different learnable matrix
        disp_resid = self.P_skin @ r_star.T # (V*3, B)
        disp_resid = disp_resid.T.reshape(batch_size, self.num_vertices, 3) # (B, V, 3)

        # 6. Combine for Final Skin Output
        # Skin = Muscle Shape + Residual Detail
        disp_skin = disp_muscle + disp_resid

        return disp_muscle, disp_skin
    
# --- NON-LINEAR BLENDSHAPES CLASS ---
class MuscleMLPModel(nn.Module):
    def __init__(self, num_vertices, num_joints=24, hidden_dim=512):
        """
        Non-Linear variation of the blendshapes.
        Instead of a single matrix multiplication (D = P * r), we use an MLP:
        D = MLP(r)
        
        Args:
            num_vertices (int): Number of vertices in the mesh (e.g. 23752).
            num_joints (int): Number of joints (e.g. 24).
            hidden_dim (int): Size of the hidden layers.
        """
        super().__init__()
        self.num_vertices = num_vertices
        self.num_joints = num_joints
        
        # Input dimension is the flattened relative rotation matrix elements
        # 9 elements * (24 - 1) joints = 207 dimensions
        input_dim = 9 * (num_joints - 1)
        output_dim = num_vertices * 3

        # --- Muscle MLP ---
        self.muscle_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),       # <--- CRITICAL FIX
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),       # <--- CRITICAL FIX
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, output_dim) # Output layer (No BN)
        )

        # --- Skin Residual MLP ---
        self.skin_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),       # <--- CRITICAL FIX
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),       # <--- CRITICAL FIX
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, output_dim)
        )

        # Initialize output layers to zero
        # This ensures that at the start of training (and at Rest Pose), 
        # the model predicts exactly 0 displacement.
        nn.init.zeros_(self.muscle_mlp[-1].weight)
        nn.init.zeros_(self.muscle_mlp[-1].bias)
        nn.init.zeros_(self.skin_mlp[-1].weight)
        nn.init.zeros_(self.skin_mlp[-1].bias)

    def forward(self, rot_mats, rot_identity):
        # 1. Input Normalization (Same as Linear Model)
        if rot_mats.dim() == 4 and rot_mats.shape[-2:] == (3, 2):
            a = rot_mats[..., 0]
            b = rot_mats[..., 1]
            rot6d = torch.cat([a, b], dim=-1)
            R = sixd_to_rotmat(rot6d)
        elif rot_mats.dim() == 3 and rot_mats.shape[-1] == 6:
            R = sixd_to_rotmat(rot_mats)
        elif rot_mats.dim() == 4 and rot_mats.shape[-2:] == (3, 3):
            R = rot_mats
        else:
            raise ValueError(f"Unsupported shape {rot_mats.shape}")

        batch_size = R.shape[0]

        # 2. Compute Relative Rotations (R - I)
        # Skip root joint (index 0)
        rel = (R[:, 1:, :, :] - rot_identity[:, 1:, :, :]) 

        # 3. Flatten Inputs
        # Shape: (B, 207)
        r_star = rel.reshape(batch_size, -1)

        # 4. Forward Pass through MLPs
        # Muscle Displacement
        disp_muscle = self.muscle_mlp(r_star)           # (B, V*3)
        disp_muscle = disp_muscle.view(batch_size, self.num_vertices, 3) # (B, V, 3)

        # Skin Residual Displacement
        disp_resid = self.skin_mlp(r_star)              # (B, V*3)
        disp_resid = disp_resid.view(batch_size, self.num_vertices, 3)   # (B, V, 3)

        # 5. Combine
        disp_skin = disp_muscle + disp_resid

        return disp_muscle, disp_skin

class MuscleUNetModel(nn.Module):
    def __init__(self, num_vertices, num_joints=24, hidden_dims=[512, 1024]):
        """
        U-Net style architecture for non-linear blendshapes.
        Uses an Encoder-Decoder structure with skip connections (concatenations)
        to prevent vanishing gradients and preserve original pose signals.
        """
        super().__init__()
        self.num_vertices = num_vertices
        self.num_joints = num_joints
        
        # Input dimension: 9 elements * (24 - 1) joints = 207 dimensions
        input_dim = 9 * (num_joints - 1)
        output_dim = num_vertices * 3

        # ==========================================
        # MUSCLE U-NET
        # ==========================================
        # Encoder (Downsampling path)
        self.m_enc1 = self._block(input_dim, hidden_dims[0])
        self.m_enc2 = self._block(hidden_dims[0], hidden_dims[1])
        
        # Bottleneck
        self.m_bottleneck = self._block(hidden_dims[1], hidden_dims[1])
        
        # Decoder (Upsampling path with Skip Connections)
        # Input size is doubled because we concatenate features from the encoder
        self.m_dec1 = self._block(hidden_dims[1] + hidden_dims[1], hidden_dims[0]) # Cats bottleneck with enc2
        self.m_dec2 = self._block(hidden_dims[0] + hidden_dims[0], hidden_dims[0]) # Cats dec1 with enc1
        
        # Final Projection to Vertices (No activation/BN)
        self.m_out = nn.Linear(hidden_dims[0], output_dim)

        # ==========================================
        # SKIN RESIDUAL U-NET
        # ==========================================
        # Encoder
        self.s_enc1 = self._block(input_dim, hidden_dims[0])
        self.s_enc2 = self._block(hidden_dims[0], hidden_dims[1])
        
        # Bottleneck
        self.s_bottleneck = self._block(hidden_dims[1], hidden_dims[1])
        
        # Decoder
        self.s_dec1 = self._block(hidden_dims[1] + hidden_dims[1], hidden_dims[0])
        self.s_dec2 = self._block(hidden_dims[0] + hidden_dims[0], hidden_dims[0])
        
        self.s_out = nn.Linear(hidden_dims[0], output_dim)

        # ==========================================
        # INITIALIZATION
        # ==========================================
        # Initialize output layers to zero (so the model starts at Rest Pose = 0 deformation)
        nn.init.zeros_(self.m_out.weight)
        nn.init.zeros_(self.m_out.bias)
        nn.init.zeros_(self.s_out.weight)
        nn.init.zeros_(self.s_out.bias)

    def _block(self, in_features, out_features, dropout=0.1):
        """Standard U-Net building block."""
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.GELU(), # GELU generally performs better than ReLU for smooth geometry regression
            nn.Dropout(dropout)
        )

    def forward(self, rot_mats, rot_identity):
        # 1. Standardize and process inputs
        if rot_mats.dim() == 4 and rot_mats.shape[-2:] == (3, 2):
            a, b = rot_mats[..., 0], rot_mats[..., 1]
            rot6d = torch.cat([a, b], dim=-1)
            R = sixd_to_rotmat(rot6d)
        elif rot_mats.dim() == 3 and rot_mats.shape[-1] == 6:
            R = sixd_to_rotmat(rot_mats)
        elif rot_mats.dim() == 4 and rot_mats.shape[-2:] == (3, 3):
            R = rot_mats
        else:
            raise ValueError(f"Unsupported shape {rot_mats.shape}")

        batch_size = R.shape[0]

        # Skip root joint (index 0) and calculate relative rotations
        rel = (R[:, 1:, :, :] - rot_identity[:, 1:, :, :]) 
        r_star = rel.reshape(batch_size, -1)

        # ------------------------------------------
        # MUSCLE FORWARD PASS
        # ------------------------------------------
        m_e1 = self.m_enc1(r_star)
        m_e2 = self.m_enc2(m_e1)
        m_b  = self.m_bottleneck(m_e2)
        
        # Skip connection: concat bottleneck and enc2
        m_d1 = self.m_dec1(torch.cat([m_b, m_e2], dim=1))
        # Skip connection: concat dec1 and enc1
        m_d2 = self.m_dec2(torch.cat([m_d1, m_e1], dim=1))
        
        disp_muscle = self.m_out(m_d2).view(batch_size, self.num_vertices, 3)

        # ------------------------------------------
        # SKIN FORWARD PASS
        # ------------------------------------------
        s_e1 = self.s_enc1(r_star)
        s_e2 = self.s_enc2(s_e1)
        s_b  = self.s_bottleneck(s_e2)
        
        s_d1 = self.s_dec1(torch.cat([s_b, s_e2], dim=1))
        s_d2 = self.s_dec2(torch.cat([s_d1, s_e1], dim=1))
        
        disp_resid = self.s_out(s_d2).view(batch_size, self.num_vertices, 3)

        # Final skin is muscle base + skin residuals
        disp_skin = disp_muscle + disp_resid

        return disp_muscle, disp_skin

# Full End-to-End Model
# This model integrates the muscle blendshape model, deformation transfer, and LBS.   
class SOMAModel(nn.Module):
    """
    Full End-to-End SOMA Model: Integrates deformation model (muscle/skin blendshapes),
    skeleton-driven LBS, and marker tracking via barycentric interpolation.
    
    Architecture:
    1. Muscle & Skin Blendshape Models: Learn deformation as D = P @ (R - I)
    2. LBS (Linear Blending Skinning): Skeleton-driven vertex weighting
    3. Barycentric Interpolation: Marker positions from deformed mesh vertices
    4. Optional Regularization: Edge smoothness, volume preservation, etc.
    """
    
    def __init__(self,
                 # ---- MESH GEOMETRY ----
                 m_bind, s_bind, b_bind, m_faces, s_faces, b_faces,
                 # ---- MESH PROPERTIES ----
                 m_rest_areas, s_rest_areas, b_rest_areas,
                 m_vertex_mass, s_vertex_mass, b_vertex_mass,
                 m_normals, s_normals, b_normals,
                 # ---- MESH REGULARIZATION ----
                 l_muscle, l_muscle_degree,
                 l_skin, l_bone, musc_edge_weights, skin_edge_weights, bone_edge_weights,
                 # ---- SKELETON (LBS) ----
                 skin_weights, j_rest, parents, offsets,
                 # ---- MARKERS (Barycentric) ----
                 p_bind, bary_verts, bary_weights,
                 # ---- OPTIONAL ----
                 active_muscle_mask=None,
                 target_marker_mask=None,
                 stability_mask=None):
        """
        Args:
            ---- MESH GEOMETRY ----
            m_bind (V, 3): Muscle bind pose vertices
            s_bind (V, 3): Skin bind pose vertices
            m_faces (F, 3): Muscle face indices
            s_faces (F, 3): Skin face indices
            
            ---- MESH PROPERTIES ----
            m_rest_areas (F,): Muscle face rest areas
            s_rest_areas (F,): Skin face rest areas
            m_vertex_mass (V,): Muscle vertex barycentric mass
            s_vertex_mass (V,): Skin vertex barycentric mass
            m_normals (V, 3): Muscle vertex normals (normalized)
            s_normals (V, 3): Skin vertex normals (normalized)
            
            ---- MESH REGULARIZATION ----
            l_muscle (V, V) sparse: Muscle Laplacian matrix
            l_muscle_degree (V,): Muscle Laplacian diagonal (degree)
            l_skin (V, V) sparse: Skin Laplacian matrix
            musc_edge_weights (E,): Muscle edge weights for regularization
            skin_edge_weights (E,): Skin edge weights for regularization
            
            ---- SKELETON (LBS) ----
            skin_weights (V, J): LBS blend weights for each vertex
            j_rest (J, 3): Joint rest positions
            parents (J,): Kinematic parent indices
            offsets (J, 3): Joint local offsets
            
            ---- MARKERS (Barycentric) ----
            p_bind (M, 3): Canonical marker positions
            bary_verts (M, 3): Barycentric vertex indices per marker
            bary_weights (M, 3): Barycentric interpolation weights
            
            ---- OPTIONAL ----
            active_muscle_mask (V, 1): Which vertices are active for deformation
            target_marker_mask (M,): Which markers to include in loss
            stability_mask (V, 1): 1.0=deformable, 0.0=rigid (face/hands)
        """
        super().__init__()
        
        # ========================================================================================
        # 1. MESH GEOMETRY BUFFERS
        # ========================================================================================
        self.register_buffer('m_bind', m_bind)
        self.register_buffer('s_bind', s_bind)
        self.register_buffer('b_bind', b_bind)
        self.register_buffer('m_faces', m_faces)
        self.register_buffer('s_faces', s_faces)
        self.register_buffer('b_faces', b_faces)
        
        # ========================================================================================
        # 2. MESH PROPERTIES BUFFERS
        # ========================================================================================
        self.register_buffer('m_rest_areas', m_rest_areas)
        self.register_buffer('s_rest_areas', s_rest_areas)        
        self.register_buffer('b_rest_areas', b_rest_areas)
        self.register_buffer('m_vertex_mass', m_vertex_mass)
        self.register_buffer('s_vertex_mass', s_vertex_mass)
        self.register_buffer('b_vertex_mass', b_vertex_mass)
        
        # Normalize normals to unit length (safety)
        m_normals_normalized = m_normals / (torch.norm(m_normals, dim=-1, keepdim=True) + 1e-8)
        s_normals_normalized = s_normals / (torch.norm(s_normals, dim=-1, keepdim=True) + 1e-8)
        self.register_buffer('m_normals', m_normals_normalized)
        self.register_buffer('s_normals', s_normals_normalized)
        
        # ========================================================================================
        # 3. MESH REGULARIZATION BUFFERS (Laplacians & Edge Weights)
        # ========================================================================================
        self.register_buffer('l_muscle', l_muscle)
        self.register_buffer('l_muscle_degree', l_muscle_degree)
        self.register_buffer('l_skin', l_skin)
        self.register_buffer('musc_edge_weights', musc_edge_weights)
        self.register_buffer('skin_edge_weights', skin_edge_weights)
        
        # ========================================================================================
        # 4. SKELETON (LBS) BUFFERS
        # ========================================================================================
        self.register_buffer('skin_weights', skin_weights)
        self.register_buffer('j_rest', j_rest)
        self.register_buffer('parents', parents)
        self.register_buffer('offsets', offsets)
        
        # ========================================================================================
        # 5. MARKER BUFFERS (Barycentric Interpolation)
        # ========================================================================================
        self.register_buffer('p_bind', p_bind)
        self.register_buffer('bary_verts', bary_verts)
        self.register_buffer('bary_weights', bary_weights)
        
        # ========================================================================================
        # 6. OPTIONAL MASKS
        # ========================================================================================
        if active_muscle_mask is not None:
            self.register_buffer('active_muscle_mask', active_muscle_mask)
        else:
            self.active_muscle_mask = None

        if target_marker_mask is not None:
            self.register_buffer('target_marker_mask', target_marker_mask)
        else:
            self.target_marker_mask = None

        if stability_mask is not None:
            self.register_buffer('stability_mask', stability_mask)
        else:
            self.stability_mask = None
        
        # ========================================================================================
        # 7. LEARNABLE DEFORMATION MODELS
        # ========================================================================================
        # Instantiate the Linear Blendshape Model
        # This learns muscle and skin deformation as D = P @ (R - I)
        
        # self.deformation_model = MuscleBlendshapeModel(
        #     num_vertices=self.m_bind.shape[0],
        #     num_joints=self.j_rest.shape[0]
        # )

        self.deformation_model = MuscleMLPModel(
            num_vertices=self.m_bind.shape[0],
            num_joints=self.j_rest.shape[0]
        )

        # self.deformation_model = MuscleUNetModel(
        #     num_vertices=self.m_bind.shape[0],
        #     num_joints=self.j_rest.shape[0]
        # )
        
        logger.debug(f"[MODEL] SOMAModel initialized with {self.m_bind.shape[0]} vertices and {self.j_rest.shape[0]} joints")
        logger.debug(f"[MODEL] Markers: {self.p_bind.shape[0]}, LBS Joints: {self.skin_weights.shape[1]}")
    
    def forward(self, rot_input, rot_identity_input, root_positions=None):
        # Normalize to rotation matrices
        if rot_input.dim() == 4 and rot_input.shape[-2:] == (3, 3):
            rot_mats = rot_input
        elif rot_input.dim() == 3 and rot_input.shape[-1] == 6:
            rot_mats = sixd_to_rotmat(rot_input) # Case for rot_input
        elif rot_input.dim() == 4 and rot_input.shape[-2:] == (3, 2):
            a, b = rot_input[..., 0], rot_input[..., 1]
            rot_mats = sixd_to_rotmat(torch.cat([a, b], dim=-1))
        else:
            raise ValueError(f"Unsupported rot_input shape {rot_input.shape}")

        # FIX: Ensure rotation matrices match debug convention ---
        # rot_mats = rot_mats.transpose(-2, -1)

        batch_size = rot_mats.shape[0]

        if root_positions is None:
            root_positions = torch.zeros(rot_mats.shape[0], 3, device=rot_mats.device, dtype=self.s_bind.dtype)

        # ===========

        # 1. The muscle blendshapes are driven by the relative rotations
        d_muscle_raw, d_skin_raw = self.deformation_model(rot_mats, rot_identity_input)  # (B,V,3)

        # --- LBS BASELINE OVERRIDE ---
        # batch_size, num_verts = rot_mats.shape[0], self.m_bind.shape[0]
        # d_muscle_raw = torch.zeros((batch_size, num_verts, 3), device=rot_mats.device)
        # d_skin_raw = torch.zeros((batch_size, num_verts, 3), device=rot_mats.device)

        # ===========

        # 2. Separate Components
        # d_resid_raw = Total - Muscle
        d_resid_raw = d_skin_raw - d_muscle_raw

        # 3. [CRITICAL] Apply Stability Mask (Pinning)
        # If mask is 1.0 (Body), we keep the deformation.
        # If mask is 0.0 (Face/Hands), we force deformation to 0.
        if self.stability_mask is not None:
            # A. Pin the Residual (Fixes spikes/volume explosion)
            d_resid = d_resid_raw * self.stability_mask
            
            # B. Pin the Muscle (Fixes structural drifting/wrinkling in degenerate areas)
            # This forces the muscle layer in face/hands to be perfectly rigid (Rest Pose).
            d_muscle = d_muscle_raw * self.stability_mask
        else:
            d_resid = d_resid_raw
            d_muscle = d_muscle_raw

        # 4. Apply Active Muscle Mask
        # If we are training a specific muscle, we zero out the deformation 
        # for all other vertices.
        if self.active_muscle_mask is not None:
             d_muscle = d_muscle * self.active_muscle_mask # Broadcasting (B,V,3) * (V,1)
             d_resid = d_resid * self.active_muscle_mask # Mask residual too if isolating

        # 5. Re-combine for Final Skin
        # In Face/Hands: d_skin = 0 + 0 = 0 (Pure LBS)
        # In Body: d_skin = Learned Muscle + Learned Residual
        d_skin = d_muscle + d_resid

        # 6. Deform the muscle mesh in the rest pose
        m_final = self.m_bind.unsqueeze(0) + d_muscle  # (1,V,3) + (B,V,3)

        # 7. Deform skin and apply LBS
        s_final = self.s_bind.unsqueeze(0) + d_skin

        # 8. LBS on skin (optimized) and muscle (optimized)
        m_final_theta = lbs_working_batch_rotmat(m_final, rot_mats, self.skin_weights, self.j_rest, self.parents, root_positions)
        s_final_theta = lbs_working_batch_rotmat(s_final, rot_mats, self.skin_weights, self.j_rest, self.parents, root_positions)

        # (Assuming self.b_bind and self.bone_weights exist)
        b_bind_batch = self.b_bind.unsqueeze(0).expand(batch_size, -1, -1)
        b_final_theta = lbs_working_batch_rotmat(b_bind_batch, rot_mats, self.skin_weights, self.j_rest, self.parents, root_positions)

        # 9. Barycentric interpolation to get predicted marker positions
        p_pred_theta = barycentric_interpolation_batch(s_final_theta, self.bary_verts, self.bary_weights)

        # 10. LBS on skin (bind, non-optimized)
        s_bind_batch = self.s_bind.unsqueeze(0).repeat(rot_mats.shape[0], 1, 1)        
        s_bind_theta = lbs_working_batch_rotmat(s_bind_batch, rot_mats, self.skin_weights, self.j_rest, self.parents, root_positions)
        p_bind_theta = barycentric_interpolation_batch(s_bind_theta, self.bary_verts, self.bary_weights)

        # 11. LBS on muscle (bind, non-optimized)
        m_bind_batch = self.m_bind.unsqueeze(0).repeat(rot_mats.shape[0], 1, 1)
        m_bind_theta = lbs_working_batch_rotmat(m_bind_batch, rot_mats, self.skin_weights, self.j_rest, self.parents, root_positions)

        # 12. T-Pose results

        # OLD
        # s_final_tpose = lbs_working_batch_rotmat(s_final, rot_identity_input, self.skin_weights, self.j_rest, self.parents, root_positions)
        
        # NEW
        zero_root = torch.zeros_like(root_positions)
        s_final_tpose = lbs_working_batch_rotmat(s_final, rot_identity_input, self.skin_weights, self.j_rest, self.parents, zero_root)

        p_pred_tpose = barycentric_interpolation_batch(s_final_tpose, self.bary_verts, self.bary_weights)

        return {
            # ---- DEFORMATIONS IN CANONICAL SPACE (T-POSE) ----
            'm_final': m_final,                         # (B, V, 3) | Muscle after blendshapes, before LBS
            's_final': s_final,                         # (B, V, 3) | Skin after blendshapes, before LBS
            
            # ---- DEFORMATIONS IN POSED SPACE (AFTER LBS) ----
            'm_final_theta': m_final_theta,             # (B, V, 3) | Muscle corrected after LBS
            's_final_theta': s_final_theta,             # (B, V, 3) | Skin corrected after LBS
            'b_final_theta': b_final_theta,             # (B, V, 3) | Bone positions corrected after LBS

            # ---- MARKER PREDICTIONS (POSED SPACE) ----
            'p_pred_theta': p_pred_theta,               # (B, M, 3) | Predicted markers from corrected skin (pose space)
            
            # ---- REFERENCE POSES (BIND POSE) ----
            's_bind_theta': s_bind_theta,               # (B, V, 3) | Bind skin after LBS (no blendshape correction)
            'p_bind_theta': p_bind_theta,               # (B, M, 3) | Bind markers from uncorrected skin
            'm_bind_theta': m_bind_theta,               # (B, V, 3) | Bind muscle after LBS
            
            # ---- CANONICAL SPACE (T-POSE) REFERENCES ----
            's_final_tpose': s_final_tpose,             # (B, V, 3) | Skin corrected in t-pose (identity rotation)
            'p_pred_tpose': p_pred_tpose,               # (B, M, 3) | Markers in t-pose space
            
            # ---- DISPLACEMENT FIELDS ----
            'd_muscle': d_muscle,                       # (B, V, 3) | Muscle displacement field (after masking)
            'd_skin': d_skin,                           # (B, V, 3) | Skin displacement field (after masking)
            
            # ---- GEOMETRIC PROPERTIES ----
            'm_normals': self.m_normals,                # (V, 3) | Muscle vertex normals (normalized)
            's_normals': self.s_normals,                # (V, 3) | Skin vertex normals (normalized)
        }

# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================

def main():
    logger.info("--- STARTING INTERACTIVE VALIDATION ---")

    """
    Initializes model, optimizer, AND loads Meshes + Canonical JSON Data.
    """
    device = torch.device(CONFIG["device"])
    logger.info(f"[TRAIN] Running on device: {device}")

    SUBJECT = CONFIG['subject']
    SHOT = CONFIG['shot']
    device = CONFIG['device']

    # ========================================================================================================================================================
    # 1. Configure paths
    logger.info("[INFO] 1. CONFIGURING PATHS...")
    # ========================================================================================================================================================
    
    # ---------------------------------------------------------
    # A. BVH file

    logger.info("[BVH] FINDING SAMPLE BVH...")

    bvh_path = glob.glob(os.path.join(PATHS['raw'], f"{SHOT}_captury", f"{SUBJECT}_{SHOT}.bvh"))[0]

    # EXPERIMENTAL
    # ------------
    # bvh_path = os.path.join(BASE_DIR, "validation", "S2_shot_002.bvh")
    bvh_path = os.path.join(BASE_DIR, "0252.bvh")
    # ------------

    if not os.path.exists(bvh_path):
        raise FileNotFoundError(f"Sample BVH file not found at {bvh_path}")

    logger.success(f"[BVH] Sample BVH file found! {bvh_path}")
    logger.info("============================================")

    # ----------------

    # ----------------
    # B. Static Meshes

    logger.info("[MESH] FINDING MESHES...")

    # Load all static assets
    muscle_mesh_path = os.path.join(PATHS['layers_tpose'], f"musc_layer-{SUBJECT}-TPose.obj")
    skin_mesh_path = os.path.join(PATHS['layers_tpose'], f"skin_layer-{SUBJECT}-TPose.obj")
    bone_mesh_path = os.path.join(PATHS['layers_tpose'], f"skel_layer-{SUBJECT}-TPose.obj")

    # EXPERIMENTAL
    # ------------
    # NEW_SUBJECT = "S5"
    # NEW_BASE_DIR = rf"/CT/SOMA/{CONFIG['base_path_suffix']}"
    # muscle_mesh_path = os.path.join(NEW_BASE_DIR, f"{NEW_SUBJECT}", "layers", "tpose", f"musc_layer-{NEW_SUBJECT}-TPose.obj")
    # skin_mesh_path = os.path.join(NEW_BASE_DIR, f"{NEW_SUBJECT}", "layers", "tpose", f"skin_layer-{NEW_SUBJECT}-TPose.obj")
    # bone_mesh_path = os.path.join(NEW_BASE_DIR, f"{NEW_SUBJECT}", "layers", "tpose", f"skel_layer-{NEW_SUBJECT}-TPose.obj")
    # ------------

    if not os.path.exists(muscle_mesh_path):
        raise FileNotFoundError(f"Muscle Layer file not found at {muscle_mesh_path}")
    else:
        logger.debug(f"[MESH] Muscle mesh path: {muscle_mesh_path}")

    if not os.path.exists(skin_mesh_path):
        raise FileNotFoundError(f"Skin Layer file not found at {skin_mesh_path}")
    else:
        logger.debug(f"[MESH] Skin mesh path: {skin_mesh_path}")   

    if not os.path.exists(bone_mesh_path):
        raise FileNotFoundError(f"Bone Layer file not found at {bone_mesh_path}")
    else:
        logger.debug(f"[MESH] Bone mesh path: {bone_mesh_path}")

    # Load all muscle meshes
    muscle_obj_dir = os.path.join(PATHS['layers_tpose'], "muscle_meshes_tpose")

    if not os.path.exists(muscle_obj_dir):
        raise FileNotFoundError(f"Muscle Meshes not found at {muscle_obj_dir}")
    else:
        logger.debug(f"[MESH] Muscle meshes path: {muscle_obj_dir}")

    logger.success("[MESH] Static meshes defined!")
    logger.info("============================================")

    # Loading Individual Muscles Meshes
    try:
        single_m_vertex_np, single_m_face_np, single_l_muscle_sp, single_m_vertex_ranges, single_m_face_ranges = load_and_stack_muscles(muscle_obj_dir)

        # Create Tensors
        single_m_bind_vertices = torch.from_numpy(single_m_vertex_np).float()
        
        # Process Block-Diagonal Laplacian (Sparse Tensor)
        indices = np.vstack((single_l_muscle_sp.row, single_l_muscle_sp.col))
        values = single_l_muscle_sp.data
        single_l_muscle_torch = torch.sparse_coo_tensor(
            torch.from_numpy(indices).long(),
            torch.from_numpy(values).float(),
            torch.Size(single_l_muscle_sp.shape)
        ).coalesce()

        # ----------------
        # Define Active Muscle Mask (OPTIONAL)
        # If set, we will zero out deformation for all muscles EXCEPT this one.
        # TARGET_MUSCLE_NAME = "r-rectus-femoris"
        TARGET_MUSCLE_NAME = None
        active_muscle_mask = None
        target_marker_mask = None # We will need this for the loss function
        target_id = None
        # ----------------

        logger.success("[MESH] All individual muscles stacked!")
        logger.info("============================================")

    except Exception as e:
        logger.error(f"[ERROR] Loading independent muscles: {e}")
        raise e

    # ----------------

    # ----------------
    # C. Canonical Data

    logger.info("[CANONICAL] FINDING CANONICAL DATA...")

    canonical_markers_path = os.path.join(PATHS['canonical'], f"{SUBJECT}_canonical_data_tpose.json")

    if not os.path.exists(canonical_markers_path):
        raise FileNotFoundError(f"Canonical Markers file not found at {canonical_markers_path}")
    else:
        logger.debug(f"[CANONICAL] Canonical markers path: {canonical_markers_path}")
        logger.info("============================================")

    # ----------------

    # ----------------
    # D. Observed Residuals + Masks

    logger.info("[RESIDUALS] FINDING RESIDUALS AND MASKS DATA...")

    local_residuals_path = os.path.join(PATHS['raw'], f"{SHOT}_captury", f"{SUBJECT}_residuals_{SHOT}_world_lbs_scaled_tpose.json")
    mask_path = os.path.join(PATHS['raw'], f"{SHOT}_captury", f"{SUBJECT}_masked_residuals_{SHOT}_world_tpose.json")

    if not os.path.exists(local_residuals_path):
        raise FileNotFoundError(f"Local Residuals file not found at {local_residuals_path}")
    else:
        logger.debug(f"[RESIDUALS] Local residuals path: {local_residuals_path}")

    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found at {mask_path}")
    else:
        logger.debug(f"[MASK] Mask path: {mask_path}")
        logger.info("============================================")

    # ----------------

    # ----------------
    # E. LBS

    logger.info("[LBS] FINDING LBS DATA...")

    skin_weights_npy_path = os.path.join(PATHS['canonical'], f"lbs_skin/{SUBJECT}_skin_lbs_weights_exported.npy")

    if not os.path.exists(skin_weights_npy_path):
        raise FileNotFoundError(f"Skin Weights file not found at {skin_weights_npy_path}")
    else:
        logger.debug(f"[LBS] Skin weights path: {skin_weights_npy_path}")
        logger.info("============================================")

    # ----------------

    # ----------------
    # F. Barycentrinc Interpolation

    logger.info("[BARYCENTRIC] FINDING BARYCENTRIC MAP DATA...")

    barycentric_map_path = os.path.join(PATHS['canonical'], "generated_marker_barycentric_map.json")

    if not os.path.exists(barycentric_map_path):
        raise FileNotFoundError(f"Barycentric Map file not found at {barycentric_map_path}")
    else:
        logger.debug(f"[BARYCENTRIC] Barycentric map path: {barycentric_map_path}")
        logger.info("============================================")

    # ----------------

    # ----------------
    # H. Muscle Vertex Mapping

    logger.info("[MUSCLE VERTEX MAPPING] FINDING MUSCLE VERTEX MAPPING DATA...")

    # Path to the binding JSON
    muscle_vertex_mapping_path = os.path.join(PATHS['canonical'], "individual_muscle_to_skin_binding.json")

    if not os.path.exists(muscle_vertex_mapping_path):
        raise FileNotFoundError(f"Muscle Vertex Mapping file not found at {muscle_vertex_mapping_path}")
    else:
        logger.debug(f"[MUSCLE VERTEX MAPPING] Muscle vertex mapping path: {muscle_vertex_mapping_path}")
        logger.info("============================================")

    # ----------------

    # ---------------------------------------
    # [OPTIMIZATION] PRE-LOAD STATIC GEOMETRY
    # ---------------------------------------
    logger.info("[OPTIMIZATION] Pre-loading static muscle and skin meshes for fast processing...")
    
    # 1. Pre-load Raw Skin (Rest Pose)
    # We need this to calculate the delta (Deformed - Rest)
    try:
        skin_verts_raw, skin_faces_raw = load_obj_simple(skin_mesh_path)
        logger.success(f"[PRE-LOAD] Loaded Raw Skin: {len(skin_verts_raw)} verts")
    except Exception as e:
        logger.error(f"[PRE-LOAD] Failed to load skin: {e}")
        skin_verts_raw, skin_faces_raw = None, None

    # 2. Pre-load All Muscles
    # Dictionary: { "muscle_name": {'verts': np.array, 'faces': np.array} }
    preloaded_muscles = {}
    if os.path.exists(muscle_obj_dir):
        muscle_files = sorted(glob.glob(os.path.join(muscle_obj_dir, "*.obj")))
        for fpath in muscle_files:
            m_name = os.path.basename(fpath).split('.')[0]
            try:
                v, f = load_obj_simple(fpath)
                preloaded_muscles[m_name] = {'verts': v, 'faces': f}
            except Exception as e:
                logger.warning(f"Failed to load muscle {m_name}: {e}")
        logger.success(f"[PRE-LOAD] Cached {len(preloaded_muscles)} muscles in memory.")
        logger.info("============================================")
    else:
        logger.error(f"[PRE-LOAD] Muscle directory not found: {muscle_obj_dir}")
    # ---------------------------------------------------------

    # ========================================================================================================================================================
    # 2. Prepare Dataset and DataLoader
    logger.info("[INFO] 2. PREPARING DATA...")
    # ========================================================================================================================================================

    # --- Load the list of unseen validation samples ---
    validation_filepaths_path = f"{SUBJECT}_validation_filepaths.json"
    if not os.path.exists(validation_filepaths_path):
        logger.error(f"Validation file not found at {validation_filepaths_path}. Please run the training script first.")
        return
    with open(validation_filepaths_path, "r") as f:
        validation_files = json.load(f)
    logger.success(f"Loaded {len(validation_files)} unseen validation samples.")
    logger.info("============================================")

    # ----------------
    # A1. BVH file (Global - RED)

    logger.info("[BVH] SETTING BVH...")

    bvh_global = BVH()
    bvh_global.load(bvh_path)

     # Convert joint names to standard Python strings
    joint_names_global = [str(name) for name in bvh_global.data['names']]
    num_joints_global = len(joint_names_global)

    # Print bones with indexes
    logger.debug("[BVH] Reading BVH file...")
    for i, name in enumerate(joint_names_global):
        logger.debug(f"[BVH] Preparing BVH (Global) with Joint {i}: {name}")
    logger.debug(f"[BVH] Loaded BVH (Global) with {num_joints_global} joints.")

    #############################
    if SUBJECT == "S1":
        SCALE = 1.0
    else:
        SCALE = 0.001

    # EXPERIMENTAL
    # SCALE = 0.001

    # Get the rest joint locations and convert to torch tensors
    j_rest_global, parents_global, offsets_global = get_rest_joint_locations(bvh_global, scale=SCALE) # Scale is always 1.0 / offset is NOT set to zero in the hips
    j_rest_tensor_global = torch.from_numpy(j_rest_global).float() # (24, 3)    <------------------------------------ TENSOR
    parents_tensor_global = torch.from_numpy(parents_global).long() # (24,)     <------------------------------------ TENSOR
    offsets_tensor_global = torch.from_numpy(offsets_global).float() # (24, 3)  <------------------------------------ TENSOR

    # ---

    # Extract motion data
    local_rotations_global, local_positions_global, _, _, _, _ = bvh_global.get_data()

    # Separate global position of the root joint for visualization
    global_positions_global = local_positions_global[:, 0, :]

    # Scale the skeleton
    local_positions_global[:, 0, :] *= SCALE # Same way we scale the offsets for the rest pose, we need to scale the root position

    num_frames = local_rotations_global.shape[0]

    # Print first 5 joints of rotations and positions for the first frame
    logger.debug(f"[BVH] First frame local rotations (Global): {local_rotations_global[0, :5]}")
    logger.debug(f"[BVH] First frame local positions (Global): {local_positions_global[0, :5]}")

    logger.success(f"[BVH] Loaded BVH (Global) with {num_frames} frames.")
    logger.info("============================================")

    # ----------------
    # A2. 6D Representation

    bvh_6d = BVH()
    bvh_6d.load(bvh_path)

    # Convert joint names to standard Python strings
    joint_names_6d = [str(name) for name in bvh_6d.data['names']]
    num_joints_6d = len(joint_names_6d)

    # Print bones with indexes
    for i, name in enumerate(joint_names_6d):
        logger.debug(f"[BVH] Preparing BVH (6D) with Joint {i}: {name}")
    logger.debug(f"[BVH] Loaded BVH (6D) with {num_joints_6d} joints.")

    #############################
    if SUBJECT == "S1":
        SCALE = 1.0
    else:
        SCALE = 0.001

    # EXPERIMENTAL
    # SCALE = 0.001

    # Get the rest joint locations and convert to torch tensors
    j_rest_6d, parents_6d, offsets_6d = get_rest_joint_locations_zero_offset(bvh_6d, scale=SCALE) # Scale is always 1.0 / offset is NOT set to zero in the hips
    j_rest_tensor_6d = torch.from_numpy(j_rest_6d).float() # (24, 3)    <------------------------------------ TENSOR
    parents_tensor_6d = torch.from_numpy(parents_6d).long() # (24,)     <------------------------------------ TENSOR
    offsets_tensor_6d = torch.from_numpy(offsets_6d).float() # (24, 3)  <------------------------------------ TENSOR

    # Extract motion data
    local_rotations_6d, local_positions_6d, _, offset_6d_new, _, _ = bvh_6d.get_data()
    
    # Separate global position of the root joint for visualization
    global_positions_6d = local_positions_6d[:, 0, :]

    # ===== PROCESS MOTION DATA ===== 

    # 1. SET ROOT POSITION IN THE WORLD ORIGIN (0,0,0) (zero translation only; keep original root orientation)
    local_positions_6d[:, 0, :] = np.zeros((local_positions_6d.shape[0], 3)) # root joint position <-- We center the root (which now is the hips, like it should be) in the global origin (0,0,0)

    # 2. SET ROTATION

    # 2.1 Do NOT zero the root rotation; preserve original orientation
    local_rotations_6d_rest = local_rotations_6d.copy()
    local_rotations_6d_rest[:, 0, :] = np.zeros((local_rotations_6d.shape[0], 4))

    # 2.2 Fix Rotation
    # Define rotation in axis angle and convert to quaternion, then to rotation matrix

    z_rot = np.pi / 2
    angles_6d = np.array([0, z_rot, z_rot])[..., np.newaxis] # angles.shape = [3, 1]

    axes_6d = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # axis.shape = [3, 3]
    q_6d = quat.from_angle_axis(angles_6d, axes_6d)
    rotmats_6d = quat.to_matrix(q_6d)

    # Extract current root orientation and convert to [x, y, z, w] for scipy
    q_root_6d = local_rotations_6d[:, 0, :]
    q_root_xyzw_6d = np.concatenate([q_root_6d[:, 1:], q_root_6d[:, 0:1]], axis=1)  # (num_frames, 4)
    
    # Convert to rotation matrices
    rotmats_root_6d = Rotation.from_quat(q_root_xyzw_6d).as_matrix()  # (num_frames, 3, 3)

    # Compose rotations
    R_x_6d = rotmats_6d[0]  # (3, 3)
    R_y_6d = rotmats_6d[1]  # (3, 3)
    R_z_6d = rotmats_6d[2]  # (3, 3)
    R_total_6d = R_z_6d @ R_y_6d @ R_x_6d  # (3, 3)

    # Apply rotations
    R_new_6d = R_total_6d @ rotmats_root_6d  # (num_frames, 3, 3)

    # Convert back to quaternion [x, y, z, w]
    R_new_quat_xyzw_6d = Rotation.from_matrix(R_new_6d).as_quat()  # (num_frames, 4)

    # Convert back to [w, x, y, z]
    R_new_quat_wxyz_6d = np.concatenate([R_new_quat_xyzw_6d[:, 3:4], R_new_quat_xyzw_6d[:, :3]], axis=1)
    local_rotations_6d[:, 0, :] = R_new_quat_wxyz_6d

    # 3. CONVERT TO 6D CONTINUOUS REPRESENTATION AND BACK TO VERIFY
    continuous = sixd.from_quat(local_rotations_6d)
    logger.debug("[BVH] 6D continuous representation shape:", continuous.shape) # (2348, 24, 3, 2)
    reconstructed_6d_local_rotations = sixd.to_quat(continuous)
    logger.debug("[BVH] 6D reconstructed rotations shape:", reconstructed_6d_local_rotations.shape) # (2348, 24, 4)

    # 4. UPDATE BVH DATA WITH PROCESSED MOTION
    bvh_6d.set_data(local_rotations_6d, local_positions_6d)

    # ==============================

    # Scale the skeleton
    local_positions_6d[:, 0, :] *= SCALE # Same way we scale the offsets for the rest pose, we need to scale the root position

    num_frames = reconstructed_6d_local_rotations.shape[0]

    logger.success(f"[BVH] Loaded BVH (6D) with {num_frames} frames.")
    logger.info("============================================")

    # ----------------

    # ----------------
    # B. Static Meshes

    logger.info("[MESH] SETTING MESHES...")

    try:
        logger.info("[MESH] Loading muscle layer...")
        musc_layer_vertices_np, musc_layer_faces_np = load_obj_simple(muscle_mesh_path)
        musc_layer = trimesh.Trimesh(vertices=musc_layer_vertices_np, faces=musc_layer_faces_np, process=False)

        if not isinstance(musc_layer, trimesh.Trimesh):
            raise ValueError(f"Loaded object is not a valid Trimesh: {type(musc_layer)}")
        else:
            logger.debug(f"[MESH] Loaded muscle layer with {musc_layer_vertices_np.shape[0]} vertices and {musc_layer_faces_np.shape[0]} faces.")

        logger.info("[MESH] Loading skin layer...")
        skin_layer_vertices_np, skin_layer_faces_np = load_obj_simple(skin_mesh_path)
        skin_layer = trimesh.Trimesh(vertices=skin_layer_vertices_np, faces=skin_layer_faces_np, process=False)

        if not isinstance(skin_layer, trimesh.Trimesh):
            raise ValueError(f"Loaded object is not a valid Trimesh: {type(skin_layer)}")
        else:
            logger.debug(f"[MESH] Loaded skin layer with {skin_layer_vertices_np.shape[0]} vertices and {skin_layer_faces_np.shape[0]} faces.")

        logger.info("[MESH] Loading bone layer...")
        bone_layer_vertices_np, bone_layer_faces_np = load_obj_simple(bone_mesh_path)
        bone_layer = trimesh.Trimesh(vertices=bone_layer_vertices_np, faces=bone_layer_faces_np, process=False)

        if not isinstance(bone_layer, trimesh.Trimesh):
            raise ValueError(f"Loaded object is not a valid Trimesh: {type(bone_layer)}")
        else:
            logger.debug(f"[MESH] Loaded bone layer with {bone_layer_vertices_np.shape[0]} vertices and {bone_layer_faces_np.shape[0]} faces.")

        logger.success(f"[MESH] Loaded mesh '{os.path.basename(muscle_mesh_path)}' successfully.")
        logger.success(f"[MESH] Loaded mesh '{os.path.basename(skin_mesh_path)}' successfully.")
        logger.success(f"[MESH] Loaded mesh '{os.path.basename(bone_mesh_path)}' successfully.")
        logger.info("============================================")

        # Compute Properties
        logger.info("[MESH] Computing muscle layer properties...")
        musc_props = compute_mesh_properties(musc_layer, device)
        logger.debug(f"[MESH] Muscle: {len(musc_props['edges'])} edges, {len(musc_props['face_rest_areas'])} faces.")

        logger.info("[MESH] Computing skin layer properties...")
        skin_props = compute_mesh_properties(skin_layer, device)
        logger.debug(f"[MESH] Skin: {len(skin_props['edges'])} edges, {len(skin_props['face_rest_areas'])} faces.")

        logger.info("[MESH] Computing bone layer properties...")
        bone_props = compute_mesh_properties(bone_layer, device)
        logger.debug(f"[MESH] Bone: {len(bone_props['edges'])} edges, {len(bone_props['face_rest_areas'])} faces.")

        # Extract Muscle Properties
        m_bind_vertices = musc_props['vertices']
        m_bind_vertices_np = musc_props['vertices_np']
        m_bind_faces = musc_props['faces']
        m_bind_num_vertices = len(m_bind_vertices)
        
        musc_edges_np = musc_props['edges']
        musc_rest_lengths_np = musc_props['edge_rest_lengths']
        musc_edge_weights = musc_props['edge_weights']
        m_rest_areas = musc_props['face_rest_areas']
        m_normals_tensor = musc_props['normals']
        m_vertex_mass = musc_props['vertex_mass']
        l_muscle_sp = musc_props['laplacian_sp']
        l_muscle_torch = musc_props['laplacian']
        muscle_degree_tensor = musc_props['laplacian_degree']

        # Extract Skin Properties
        s_bind_vertices = skin_props['vertices']
        s_bind_vertices_np = skin_props['vertices_np']
        s_bind_faces = skin_props['faces']
        s_bind_num_vertices = len(s_bind_vertices)
        
        skin_edges_np = skin_props['edges']
        skin_rest_lengths_np = skin_props['edge_rest_lengths']
        skin_edge_weights = skin_props['edge_weights']
        s_rest_areas = skin_props['face_rest_areas']
        s_normals_tensor = skin_props['normals']
        s_vertex_mass = skin_props['vertex_mass']
        l_skin_sp = skin_props['laplacian_sp']
        l_skin_torch = skin_props['laplacian']

        # Extract Bone Properties
        b_bind_vertices = bone_props['vertices']
        b_bind_vertices_np = bone_props['vertices_np']
        b_bind_faces = bone_props['faces']
        b_bind_num_vertices = len(b_bind_vertices)
        
        bone_edges_np = bone_props['edges']
        bone_rest_lengths_np = bone_props['edge_rest_lengths']
        bone_edge_weights = bone_props['edge_weights']
        b_rest_areas = bone_props['face_rest_areas']
        b_normals_tensor = bone_props['normals']
        b_vertex_mass = bone_props['vertex_mass']
        l_bone_sp = bone_props['laplacian_sp']
        l_bone_torch = bone_props['laplacian']

        # Face adjacency
        logger.info("[MESH] Calculating face adjacency pairs...")
        face_adjacency_np = skin_layer.face_adjacency
        logger.debug(f"[MESH] Extracted {len(face_adjacency_np)} face adjacency pairs.")

        logger.success("[MESH] All mesh properties computed successfully.")
        logger.success(f"[MESH] Vertices: skin={s_bind_vertices_np.shape}, muscle={m_bind_vertices_np.shape}")
        logger.success(f"[MESH] Faces: skin={len(skin_layer.faces)}, muscle={len(musc_layer.faces)}")
        logger.info("============================================")

    except Exception as e:
        logger.error(f"Error loading mesh file: {e}")
        return

    # ----------------

    # ----------------
    # C. Canonical Data

    # Load and process the canonical data
    logger.info("[CANONICAL] Loading and processing canonical data...")
    
    with open(canonical_markers_path, 'r') as f:
        canonical_data = json.load(f).get("0", {})

    # Get a canonical, sorted list of marker IDs and their base positions
    canonical_marker_ids = sorted(canonical_data.keys())
    marker_id_to_index = {marker_id: i for i, marker_id in enumerate(canonical_marker_ids)}
    num_markers = len(canonical_marker_ids)

    # This is because in the barycentric map, there are 4 markers that do not map to the skin mesh
    # We will remove these markers from the canonical data when calculating the loss
    logger.debug(f"[CANONICAL] Found {num_markers} canonical markers.")

    # Load barycentric map and extract bary_verts and bary_weights
    with open(barycentric_map_path, 'r') as f:
        barycentric_map = json.load(f)

    # We need to create p_bind in the order of barycentric_map keys, to ensure consistency
    # --------------------------------------------------------------------------------------------
    
    # p_bind = np.zeros((len(canonical_marker_ids), 3), dtype=np.float32)
    # for marker_id, pos_list in canonical_data.items():
    #     if marker_id in marker_id_to_index:
    #         idx = marker_id_to_index[marker_id]
    #         p_bind[idx] = np.array(pos_list[0])  # Use the first position in the list

    bary_marker_ids = sorted(barycentric_map.keys())
    p_bind = np.zeros((len(bary_marker_ids), 3), dtype=np.float32)
    for i, marker_id in enumerate(bary_marker_ids):
        pos_list = canonical_data[marker_id][0]
        p_bind[i] = np.array(pos_list)

    # --------------------------------------------------------------------------------------------

    # Rotate canonical markers to align with BVH visualization
    # ====================================================== #
    DEGREE_BVH_X = -90.0
    p_bind = _rot_x(p_bind, deg=DEGREE_BVH_X)
    # ====================================================== #

    p_bind_tensor = torch.from_numpy(p_bind).float() # <------------------------------------ TENSOR

    logger.success(f"[CANONICAL] Loaded p_bind '{os.path.basename(canonical_markers_path)}' successfully.")
    logger.info("============================================")

    # ----------------

    # ----------------
    # E. LBS

    logger.info(f"[LBS] Loaded skin mesh with {s_bind_num_vertices} vertices and {len(skin_layer.faces)} faces.")

    weights_data = np.load(skin_weights_npy_path) # Instead of generating it, we just load it now
    skin_weights_tensor = torch.from_numpy(weights_data).float() # <------------------------------------ TENSOR
 
    logger.debug(f"[LBS] Loaded skin_weights matrix with shape: {skin_weights_tensor.shape}")  # (23752, 24)
    logger.info("============================================")

    # ----------------

    # ----------------
    # F. Barycentrinc Interpolation

    # Load and process the barycentric map
    logger.info("[BARYCENTRIC] Loading and processing barycentric map...")

    # Load the saved barycentric map for further use
    with open(barycentric_map_path, 'r') as f:
        barycentric_map = json.load(f)

    logger.info(f"[BARYCENTRIC] Barycentric map loaded from {barycentric_map_path}")

    # We need to ensure the markers are in a consistent order
    canonical_marker_ids = sorted(barycentric_map.keys())
    num_markers = len(canonical_marker_ids)

    logger.success(f"[BARYCENTRIC] Number of markers found in barycentric map: {num_markers}") # 2306
    logger.info("============================================")

    # Create tensors for bary_verts and bary_weights
    bary_verts_np = np.zeros((num_markers, 3), dtype=np.int64)
    bary_weights_np = np.zeros((num_markers, 3), dtype=np.float32)

    for i, marker_id in enumerate(canonical_marker_ids):
        data = barycentric_map[marker_id]
        bary_verts_np[i] = data['vertex_indices']
        bary_weights_np[i] = data['bary_coords'][0]

    bary_verts_tensor = torch.from_numpy(bary_verts_np).long() #  <------------------------------------ TENSOR
    bary_weights_tensor = torch.from_numpy(bary_weights_np).float() #  <-------------------------------- TENSOR

    # ----------------

    # 1. Prepare Skinning Static Data
    skin_weights_np = skin_weights_tensor.cpu().numpy() # (V, J)
    joint_names_list = joint_names_global              # List of strings
    parents_list = parents_global.tolist()             # List of ints
    
    # 2. Calculate Inverse Bind Matrices (using Global Rest Positions)
    inv_bind_matrices = get_inverse_bind_matrices(j_rest_tensor_global)

    # ----------------
    # H. Muscle Vertex Mapping -> [[DONE]]

    muscle_bary_map = None
    muscle_name_mapping = {}  # Map new filenames to old JSON keys

    if os.path.exists(muscle_vertex_mapping_path):
        with open(muscle_vertex_mapping_path, 'r') as f:
            muscle_bary_map = json.load(f)
        logger.debug(f"[MUSCLE VERTEX MAPPING] Loaded Muscle-Skin Map from {os.path.basename(muscle_vertex_mapping_path)}")

        # Create mapping from actual filenames to JSON keys
        all_muscle_files = sorted(glob.glob(os.path.join(muscle_obj_dir, "*.obj")))
        json_keys = list(muscle_bary_map.keys()) # Get all keys from the JSON map
        
        if len(all_muscle_files) > 0:
            for actual_file in all_muscle_files:
                actual_name = os.path.basename(actual_file).replace(".obj", "") # e.g., "Abs_t-pose"
                json_key_to_use = None # Initialize to None for each muscle

                # --- Mapping Strategy ---
                # 1. Attempt: Exact match (e.g., "Abs_t-pose" == "Abs_t-pose")
                if actual_name in muscle_bary_map:
                    json_key_to_use = actual_name
                
                # 2. Attempt: Specific transformation for common patterns
                #    e.g., "Abs_t-pose" (from OBJ) -> "Abs_a-pose-ID" (for JSON)
                if json_key_to_use is None and "_t-pose" in actual_name:
                    base_muscle_name = actual_name.replace("_t-pose", "") # Extracts "Abs"
                    candidate_json_key = f"{base_muscle_name}_a-pose-ID" # Constructs "Abs_a-pose-ID"
                    if candidate_json_key in muscle_bary_map:
                        json_key_to_use = candidate_json_key
                
                # 3. Attempt: General suffix removal and then check for direct match
                #    e.g., "Abs_t-pose" -> "Abs" (then check if "Abs" is a JSON key)
                if json_key_to_use is None:
                    simplified_actual_name = actual_name.replace("_t-pose", "").replace("_FIX", "").replace("-FIX", "").replace("_a-pose", "")
                    if simplified_actual_name in muscle_bary_map:
                        json_key_to_use = simplified_actual_name
                    else:
                        # 4. Attempt: Fuzzy match (case-insensitive comparison of simplified names)
                        #    e.g., "Abs" (from OBJ) vs "Abs" (from JSON key "Abs_a-pose-ID")
                        for json_key in json_keys:
                            simplified_json_key = json_key.replace("_a-pose-ID", "").replace("_t-pose", "").replace("_FIX", "").replace("-FIX", "").replace("_a-pose", "")
                            if simplified_actual_name.lower() == simplified_json_key.lower():
                                json_key_to_use = json_key # Use the original JSON key
                                break
                # --- End Mapping Strategy ---
                
                if json_key_to_use is not None:
                    muscle_name_mapping[actual_name] = json_key_to_use
                else:
                    logger.warning(f"No mapping found for OBJ '{actual_name}'. Please check naming conventions.")
            
            logger.success(f"[MUSCLE VERTEX MAPPING] Created muscle name mapping: {len(muscle_name_mapping)} files matched to JSON keys")
            logger.info("============================================")
            
            if len(muscle_name_mapping) < len(all_muscle_files):
                logger.warning(f"[MUSCLE VERTEX MAPPING] Only {len(muscle_name_mapping)}/{len(all_muscle_files)} muscles mapped. Missing:")
                for actual_file in all_muscle_files:
                    actual_name = os.path.basename(actual_file).replace(".obj", "")
                    if actual_name not in muscle_name_mapping:
                        logger.warning(f"  - {actual_name}")
    else:
        logger.warning(f"[MUSCLE VERTEX MAPPING] Map not found at {muscle_vertex_mapping_path}")

    # ----------------

    # ========================================================================================================================================================
    # 3. Prepare the Model
    logger.info("[INFO] 3. PREPARING THE MODEL...")
    # ========================================================================================================================================================

    try:            
        # MESH GEOMETRY: Vertex positions and face indices
        mesh_geometry_tensors = {
            'm_bind': m_bind_vertices,           # Muscle bind pose vertices (23752, 3)
            's_bind': s_bind_vertices,           # Skin bind pose vertices (23752, 3)
            'b_bind': b_bind_vertices,           # Bone bind pose vertices (n_bone_vertices, 3)
            'm_faces': m_bind_faces,             # Muscle face indices (n_faces, 3)
            's_faces': s_bind_faces,             # Skin face indices (n_faces, 3)
            'b_faces': b_bind_faces,             # Bone face indices (n_bone_faces, 3)
        }
        
        # MESH PROPERTIES: Areas, normals, masses
        mesh_properties_tensors = {
            'm_rest_areas': m_rest_areas,        # Muscle face rest areas (n_faces,)
            's_rest_areas': s_rest_areas,        # Skin face rest areas (n_faces,)
            'b_rest_areas': b_rest_areas,        # Bone face rest areas (n_bone_faces,)
            'm_vertex_mass': m_vertex_mass,      # Muscle vertex mass diagonal (23752,)
            's_vertex_mass': s_vertex_mass,      # Skin vertex mass diagonal (23752,)
            'b_vertex_mass': b_vertex_mass,      # Bone vertex mass diagonal (n_bone_vertices,)
            'm_normals': m_normals_tensor,       # Muscle vertex normals (23752, 3)
            's_normals': s_normals_tensor,       # Skin vertex normals (23752, 3)
            'b_normals': b_normals_tensor,       # Bone vertex normals (n_bone_vertices, 3)
        }
        
        # MESH REGULARIZATION: Laplacians and edge weights
        mesh_regularization_tensors = {
            'l_muscle': l_muscle_torch,          # Muscle Laplacian (23752, 23752) sparse
            'l_muscle_degree': muscle_degree_tensor,  # Muscle Laplacian degree (23752,)
            'l_skin': l_skin_torch,              # Skin Laplacian (23752, 23752) sparse
            'l_bone': l_bone_torch,              # Bone Laplacian (n_bone_vertices, n_bone_vertices) sparse
            'musc_edge_weights': musc_edge_weights,   # Muscle edge weights (n_edges,)
            'skin_edge_weights': skin_edge_weights,   # Skin edge weights (n_edges,)
            'bone_edge_weights': bone_edge_weights,   # Bone edge weights (n_bone_edges,)
        }
        
        # SKELETON: Joint positions and kinematic chain
        skeleton_tensors = {
            'j_rest': j_rest_tensor_global,      # Joint rest positions (24, 3)
            'parents': parents_tensor_global,    # Kinematic parent indices (24,)
            'offsets': offsets_tensor_global,    # Joint offsets (24, 3)
        }
        
        # MARKERS: Marker positions and interpolation
        marker_tensors = {
            'p_bind': p_bind_tensor,             # Canonical marker positions (2306, 3)
            'bary_verts': bary_verts_tensor,     # Marker barycentric vertex indices (2306, 3)
            'bary_weights': bary_weights_tensor, # Marker barycentric weights (2306, 3)
        }
        
        # LBS: Skinning weights for skeleton-driven deformation
        lbs_tensors = {
            'skin_weights': skin_weights_tensor, # LBS weights (23752, 24)
        }
        
        # ADAPTIVE: Model-specific weights and masks
        adaptive_tensors = {
            'adaptive_weights': torch.ones(m_bind_vertices.shape[0], 1),  # Initialize to 1s
            'active_muscle_mask': active_muscle_mask,   # Optional: active muscles
            'target_marker_mask': target_marker_mask,   # Optional: target markers
        }
        
        # Combine all tensor dictionaries
        all_tensors = {}
        all_tensors.update(mesh_geometry_tensors)
        all_tensors.update(mesh_properties_tensors)
        all_tensors.update(mesh_regularization_tensors)
        all_tensors.update(skeleton_tensors)
        all_tensors.update(marker_tensors)
        all_tensors.update(lbs_tensors)
        all_tensors.update(adaptive_tensors)
        
        # Move all tensors to device in one operation
        device_tensors = move_tensors_to_device(all_tensors, device)
        
        # Unpack from moved dictionary for backward compatibility
        m_bind = device_tensors['m_bind']
        s_bind = device_tensors['s_bind']
        b_bind = device_tensors['b_bind']
        m_faces = device_tensors['m_faces']
        s_faces = device_tensors['s_faces']
        b_faces = device_tensors['b_faces']
        m_rest_areas = device_tensors['m_rest_areas']
        s_rest_areas = device_tensors['s_rest_areas']
        b_rest_areas = device_tensors['b_rest_areas']
        m_vertex_mass = device_tensors['m_vertex_mass']
        s_vertex_mass = device_tensors['s_vertex_mass']
        b_vertex_mass = device_tensors['b_vertex_mass']
        m_normals_tensor = device_tensors['m_normals']
        s_normals_tensor = device_tensors['s_normals']
        b_normals_tensor = device_tensors['b_normals']
        l_muscle = device_tensors['l_muscle']
        muscle_degree = device_tensors['l_muscle_degree']
        l_skin = device_tensors['l_skin']
        l_bone = device_tensors['l_bone']
        bone_edge_weights = device_tensors['bone_edge_weights']
        musc_edge_weights = device_tensors['musc_edge_weights']
        skin_edge_weights = device_tensors['skin_edge_weights']
        j_rest = device_tensors['j_rest']
        parents = device_tensors['parents']
        offsets = device_tensors['offsets']
        p_bind_tensor = device_tensors['p_bind']
        bary_verts = device_tensors['bary_verts']
        bary_weights = device_tensors['bary_weights']
        weights = device_tensors['skin_weights']
        adaptive_weights = device_tensors['adaptive_weights']
        active_muscle_mask = device_tensors['active_muscle_mask']
        target_marker_mask = device_tensors['target_marker_mask']
        
        logger.info("[INFO] ✓ Mesh geometry, properties, regularization moved to device")
        logger.info("[INFO] ✓ Skeleton and kinematic chain moved to device")
        logger.info("[INFO] ✓ Marker positions and interpolation moved to device")
        logger.info("[INFO] ✓ LBS weights and adaptive parameters moved to device")
        logger.info("[INFO] ✓ Static assets loaded and moved to device successfully.")

    except Exception as e:
        logger.error("[ERROR] Loading static assets. Please check the mesh files and weights.")
        raise e

    try:
        logger.info("[MESH] Computing Stability Mask (Pinning Face/Hands)...")
        stability_mask = compute_vertex_stability_mask(m_bind, s_bind, m_faces)
        logger.success(f"[MESH] Stability Mask created. Pinned {int((1-stability_mask).sum())} vertices.")

        # Instantiate the SOMAModel with all required tensors
        # Note: SOMAModel constructor expects all tensors already on device
        model = SOMAModel(
            # ---- MESH GEOMETRY ----
            m_bind=m_bind,
            s_bind=s_bind,
            b_bind=b_bind,
            m_faces=m_faces,
            s_faces=s_faces,
            b_faces=b_faces,
            # ---- MESH PROPERTIES ----
            m_rest_areas=m_rest_areas,
            s_rest_areas=s_rest_areas,
            b_rest_areas=b_rest_areas,
            m_vertex_mass=m_vertex_mass,
            s_vertex_mass=s_vertex_mass,
            b_vertex_mass=b_vertex_mass,
            m_normals=m_normals_tensor,
            s_normals=s_normals_tensor,
            b_normals=b_normals_tensor,
            # ---- MESH REGULARIZATION ----
            l_muscle=l_muscle,
            l_muscle_degree=muscle_degree,
            l_skin=l_skin,
            l_bone=l_bone,
            musc_edge_weights=musc_edge_weights,
            skin_edge_weights=skin_edge_weights,
            bone_edge_weights=bone_edge_weights,
            # ---- SKELETON (LBS) ----
            skin_weights=weights,
            j_rest=j_rest,
            parents=parents,
            offsets=offsets,
            # ---- MARKERS (Barycentric) ----
            p_bind=p_bind_tensor,
            bary_verts=bary_verts,
            bary_weights=bary_weights,
            # ---- OPTIONAL ----
            active_muscle_mask=active_muscle_mask,
            target_marker_mask=target_marker_mask,
            stability_mask=stability_mask
        )

        logger.success(f"[MODEL] SOMAModel instantiated successfully.")
        logger.success(f"[MODEL] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        model = model.to(device)

        logger.success(f"[MODEL] Model moved to device: {device}")

        blendshape_weights = torch.load(CHECKPOINT_PATH, map_location=device)
        model.deformation_model.load_state_dict(blendshape_weights)

        logger.success(f"[MODEL] Weights {CHECKPOINT_PATH} loaded.")

        model.eval()

    except Exception as e:
        logger.error("[ERROR] Model instantiation failed.")
        raise e

    # ========================================================================================================================================================
    # 4. Visualization
    logger.info("[INFO] 4. SETTING VISUALIZATION...")
    # ========================================================================================================================================================

    server = viser.ViserServer()
    server.scene.add_grid(name="grid", width=8, height=8, plane="xz", section_color=(0, 0, 0), cell_color=(230, 230, 230), cell_thickness=1, cell_size=0.25, section_thickness=0.1, section_size=2)
    server.scene.set_up_direction("+y")

    # ==========================================
    # NEW: State dictionary to hold data for exporting
    # ==========================================
    export_state = {
        "m_final_tpose": None,
        "s_final_tpose": None,
        "frame_idx": 0
    }

    VALIDATION_CLIPS = False

    if VALIDATION_CLIPS:
        frame_slider = server.gui.add_slider("Frame", min=0, max=len(validation_files) - 1, step=1, initial_value=0)
        current_file_display = server.gui.add_markdown("Current File: **N/A**")     
    else:
        frame_slider = server.gui.add_slider("Frame", min=0, max=max(0, num_frames - 2), step=1, initial_value=0)
        current_file_display = server.gui.add_markdown(f"Current File: **{SHOT}_frame_0000.npy**")

    # BVH
    bvh_global_bool = server.gui.add_checkbox("BVH (Global)", initial_value=False)
    bvh_6d_bool = server.gui.add_checkbox("BVH (6D)", initial_value=False)

    # Hips axes UI
    hips_axes_bool = server.gui.add_checkbox("Hips Axes (root)", initial_value=True)
    hips_axes_len = server.gui.add_slider("Hips Axes Length", min=0.01, max=0.5, step=0.005, initial_value=0.1)
    
    # Play Controls
    play_bool = server.gui.add_checkbox("Play Animation", initial_value=False)
    playback_speed = server.gui.add_slider("Playback Speed (ms)", min=10, max=1000, step=10, initial_value=100)

    # --- ANIMATION EXPORT UI ---
    with server.gui.add_folder("Animation Sequence Exporter"):

        directory_export = f"./exports_multiple/{SUBJECT}/{SHOT}"

        export_dir = server.gui.add_text("Export Directory", initial_value=directory_export)
        export_skin = server.gui.add_checkbox("Seq: Export Skin", initial_value=True)
        export_muscle = server.gui.add_checkbox("Seq: Export Muscle", initial_value=True)
        # export_bone = server.gui.add_checkbox("Seq: Export Bone", initial_value=False)
        export_unified_indiv = server.gui.add_checkbox("Seq: Export Unified Indiv. Muscles", initial_value=False)
        record_bool = server.gui.add_checkbox("🔴 RECORD (Saves during Playback)", initial_value=False)


    # ==========================================
    # Export Button and Logic
    # ==========================================
    with server.gui.add_folder("Export"):
        export_btn = server.gui.add_button("Export T-Pose Meshes (Current Frame)")

        @export_btn.on_click
        def _(_):
            if export_state["m_final_tpose"] is None:
                logger.warning("No meshes generated yet. Please wait for the frame to load.")
                return
            
            frame_idx = export_state["frame_idx"]
            export_dir = os.path.join("exports_single", f"frame_{frame_idx:06d}_tpose")
            os.makedirs(export_dir, exist_ok=True)
            
            logger.info(f"Exporting T-Pose meshes to {export_dir}...")
            
            # 1. Export Full Skin
            skin_mesh = trimesh.Trimesh(vertices=export_state["s_final_tpose"], faces=skin_layer.faces, process=False)
            skin_mesh.export(os.path.join(export_dir, "s_final_tpose.obj"))
            
            # 2. Export Full Muscle Layer
            musc_mesh = trimesh.Trimesh(vertices=export_state["m_final_tpose"], faces=musc_layer.faces, process=False)
            musc_mesh.export(os.path.join(export_dir, "m_final_tpose.obj"))
            
            # 3. Export Individual Muscles
            indiv_dir = os.path.join(export_dir, "individual_muscles")
            os.makedirs(indiv_dir, exist_ok=True)
            
            # Loop through the ranges we captured during `load_and_stack_muscles`
            for m_name, v_range in single_m_vertex_ranges.items():
                v_start, v_end = v_range
                f_start, f_end = single_m_face_ranges[m_name]
                
                # Extract this specific muscle's vertices
                m_verts = export_state["m_final_tpose"][v_start:v_end]
                
                # Extract faces and re-zero them so they start at index 0 for the individual OBJ
                m_faces = single_m_face_np[f_start:f_end] - v_start
                
                m_indiv_mesh = trimesh.Trimesh(vertices=m_verts, faces=m_faces, process=False)
                m_indiv_mesh.export(os.path.join(indiv_dir, f"{m_name}.obj"))
                
            logger.success(f"✅ Export completed for Frame {frame_idx}!")
    # ==========================================

    # Residuals
    residuals_gt_bool = server.gui.add_checkbox("Ground Truth Residuals (T-Pose)", initial_value=False)

    # Model
    inference_bool = server.gui.add_checkbox("Model Inference", initial_value=False)

    # Outputs
    output_final_theta_bool = server.gui.add_checkbox("Model Output (Theta)", initial_value=False)
    output_final_tpose_bool = server.gui.add_checkbox("Model Output (T-Pose)", initial_value=False)

    # Individual muscless
    with server.gui.add_folder("Individual Muscles"):
        show_individual_muscles_bool = server.gui.add_checkbox("Show Deformed Muscles", initial_value=False)
        export_individual_muscles_bool = server.gui.add_checkbox("Export Deformed Muscles", initial_value=False)
        target_muscle_text = server.gui.add_text("Target Muscle (leave empty for all)", initial_value="")
        
    export_dir_ind_muscle = os.path.join(BASE_DIR, "exported_ind_musc_meshes", f"{SHOT}_captury")
    os.makedirs(export_dir_ind_muscle, exist_ok=True)

    # ----------------
    # A2. 6D Representation (REST)

    # Add the static skeleton 
    bone_points_6d = []
    for i, p_idx in enumerate(parents_6d):
        if p_idx != -1:
            # Each bone is a pair of [start_point, end_point]
            # Convert to CPU and numpy if tensor
            start = j_rest_6d[p_idx]
            end = j_rest_6d[i]
            if torch.is_tensor(start):
                start = start.cpu().numpy()
            if torch.is_tensor(end):
                end = end.cpu().numpy()
            bone_points_6d.append([start, end])

    # Convert to a single NumPy array of shape (num_bones, 2, 3)
    bone_points_6d = np.array(bone_points_6d)
    
    # Update bone segments
    server.scene.add_line_segments(
        name="/theta_rest/bones",
        points=bone_points_6d,
        line_width=3.0,
        colors=(0, 255, 0), # Green
    )

    # Update joint positions
    server.scene.add_point_cloud(
        name="/theta_rest/joints",
        points=j_rest_6d.cpu().numpy() if torch.is_tensor(j_rest_6d) else j_rest_6d,
        colors=(255, 255, 0), # Yellow
        point_size=0.015,
    )

    # Draw hips local coordinate axes (Y=green, Z=blue)
    if hips_axes_bool.value:
        axis_len = float(hips_axes_len.value)

        # Root rotation matrix from BVH quaternions
        pose_rotations_tensor_axes_rest = torch.from_numpy(local_rotations_6d_rest[0]).float() # < ------------------------------------ TENSOR
        rot_mats_axes_rest = quat_torch.to_matrix(pose_rotations_tensor_axes_rest)  # (J, 3, 3)
        R0_global_rest = rot_mats_axes_rest[0].cpu().numpy() # hips/global root

        origin_world = j_rest_6d[0]
        x_end_rest = origin_world + axis_len * R0_global_rest[:, 0]  # +X
        y_end_rest = origin_world + axis_len * R0_global_rest[:, 1]  # +Y
        z_end_rest = origin_world + axis_len * R0_global_rest[:, 2]  # +Z

        x_seg_rest = np.stack([origin_world, x_end_rest], axis=0)  # (2,3)
        y_seg_rest = np.stack([origin_world, y_end_rest], axis=0)  # (2,3)
        z_seg_rest = np.stack([origin_world, z_end_rest], axis=0)  # (2,3)

        server.scene.add_line_segments(
            name="/theta_rest/hips_axis_global/x",
            points=x_seg_rest[None, ...],  # (1,2,3)
            line_width=4.0,
            colors=(255, 0, 0),  # Red
        )
        server.scene.add_line_segments(
            name="/theta_rest/hips_axis_global/y",
            points=y_seg_rest[None, ...],  # (1,2,3)
            line_width=4.0,
            colors=(0, 255, 0),  # Green
        )
        server.scene.add_line_segments(
            name="/theta_rest/hips_axis_global/z",
            points=z_seg_rest[None, ...],  # (1,2,3)
            line_width=4.0,
            colors=(0, 0, 255),  # Blue
        )

    # ----------------

    # ----------------
    # B. Static Meshes

    server.scene.add_mesh_simple(
        name="/static/m_bind",
        vertices=musc_layer.vertices,
        faces=musc_layer.faces,
        color=(200, 0, 0), # Red
        wireframe = False
    )

    server.scene.add_mesh_simple(
        name="/static/s_bind",
        vertices=skin_layer.vertices,
        faces=skin_layer.faces,
        color=(120, 158, 240), # Blue
        wireframe = False
    )

    server.scene.add_mesh_simple(
        name="/static/b_bind",
        vertices=bone_layer.vertices,
        faces=bone_layer.faces,
        color=(220, 220, 220), # Beige
        wireframe = False
    )

    # ----------------
    # C. Canonical Data

    # Add canonical pointcloud
    server.scene.add_point_cloud(
        name="/static/p_bind",
        points=p_bind,
        colors=(180, 180, 0), # Yellow
        point_size=0.0025
    )

    # ----------------

    # ----------------

    # ========================================================================================================================================================
    # 5. Animation
    logger.info("[INFO] 5. SETTING ANIMATION...")
    # ========================================================================================================================================================

    print("\nOpen your browser to http://localhost:8080")
    print("Press Ctrl+C in the terminal to exit.")

    # ====================================================================================
    # [FIX] Calculate Global Offset Correction (Frame 0 Bias Removal)
    # ====================================================================================
    offset_correction = None
    try:
        res_path_0 = os.path.join(PATHS['processed'], 'residuals', f"{SHOT}_frame_0000.npy")

        if os.path.exists(res_path_0):
            res_0 = np.load(res_path_0).astype(np.float32).reshape(-1, 3)
            offset_correction = res_0
            logger.success(f"[OFFSET FIX] Loaded GLOBAL offset bias from {os.path.basename(res_path_0)}.")
        else:
            logger.warning("[OFFSET FIX] Could not find Frame 0 residuals. Skipping offset correction.")
    except Exception as e:
        logger.error(f"[OFFSET FIX] Error calculating offset: {e}")
    # ====================================================================================

    def update_scene(camera_handle: Optional[viser.CameraHandle] = None): # ADDED: camera_handle parameter
        
        # Get the current frame index and scale factor from the sliders
        current_frame_idx = frame_slider.value

        # --- Load Ground Truth Residuals for the current frame ---
        residuals_gt_for_frame = None
        try:
            if VALIDATION_CLIPS:
                # Extract base filename from validation_files path
                pose_filepath = validation_files[current_frame_idx]
                base_filename = os.path.basename(pose_filepath)
                residuals_gt_path = os.path.join(PATHS['processed'], 'residuals', base_filename)
                current_file_display.content = f"Current File: **{base_filename}**"
            else:
                # Construct filename for non-validation clips
                # Assuming shot_name is defined globally or passed
                filename = f"{SHOT}_frame_{current_frame_idx:04d}.npy"
                residuals_gt_path = os.path.join(PATHS['processed'], 'residuals', filename)
                current_file_display.content = f"Current File: **{filename}**"

            if os.path.exists(residuals_gt_path):
                residuals_gt_for_frame = np.load(residuals_gt_path).astype(np.float32)
                
                # Reshape from (N*3,) to (N, 3)
                residuals_gt_for_frame = residuals_gt_for_frame.reshape(-1, 3)

                # [NEW] Load the Mask for Visualization Colors
                # ------------------------------------------------------------------------------------
                mask_filename = os.path.basename(residuals_gt_path)
                mask_gt_path = os.path.join(PATHS['processed'], 'masks', mask_filename)
                
                if os.path.exists(mask_gt_path):
                    current_mask = np.load(mask_gt_path).reshape(-1)
                else:
                    logger.warning(f"Mask file not found: {mask_gt_path}")
                    current_mask = None
                # ------------------------------------------------------------------------------------
            
                # ====================================================================================
                # [FIX] Apply the offset correction
                # ====================================================================================
                if offset_correction is not None:
                    # residuals_gt_for_frame -= offset_correction
                    residuals_gt_for_frame = residuals_gt_for_frame

                # [CRITICAL FIX] Zero out phantom offsets for invalid markers
                if current_mask is not None:
                    residuals_gt_for_frame[current_mask == 0] = 0.0
                # ====================================================================================

            else:
                logger.warning(f"Ground truth residuals file not found for frame {current_frame_idx}: {residuals_gt_path}")
        except Exception as e:
            logger.error(f"Error loading ground truth residuals for frame {current_frame_idx}: {e}")
        # ---------------------------------------------------------

        # ----------------
        # A1. BVH file (Global - RED)

        if bvh_global_bool.value == True:
            # 1. Calculate the skeleton's pose relative to its own origin (0,0,0)
            posed_joints_local_global, _ = fk(local_rotations_global[current_frame_idx], np.zeros(3), offsets_global, parents_global)

            # 2. Convert to world space by adding the global root position (brings to original world position)
            posed_joints_world_global = posed_joints_local_global + global_positions_global[current_frame_idx, :] # global_positions_global is non-zero

            # Add the dynamic skeleton 
            bone_points_global = []
            for i, p_idx in enumerate(parents_global):
                if p_idx != -1:
                    # Each bone is a pair of [start_point, end_point]
                    bone_points_global.append([posed_joints_world_global[p_idx], posed_joints_world_global[i]])

            # Convert to a single NumPy array of shape (num_bones, 2, 3)
            bone_points_global = np.array(bone_points_global)

            # Update bone segments
            server.scene.add_line_segments(
                name="/theta_global/bones",
                points=bone_points_global,
                line_width=3.0,
                colors=(255, 0, 0), # Red
            )

            # Update joint positions
            server.scene.add_point_cloud(
                name="/theta_global/joints",
                points=posed_joints_world_global,
                colors=(255, 255, 0), # Yellow
                point_size=0.015,
            )

            # Draw hips local coordinate axes (Y=green, Z=blue)
            if hips_axes_bool.value:
                axis_len = float(hips_axes_len.value)

                # Root rotation matrix from BVH quaternions
                pose_rotations_tensor_axes_global = torch.from_numpy(local_rotations_global[current_frame_idx]).float() # < ------------------------------------ TENSOR
                rot_mats_axes = quat_torch.to_matrix(pose_rotations_tensor_axes_global)  # (J, 3, 3)
                R0_global = rot_mats_axes[0].cpu().numpy()  # hips/global root

                origin_world_global = posed_joints_world_global[0]
                x_end_world = origin_world_global + axis_len * R0_global[:, 0]  # +X
                y_end_world = origin_world_global + axis_len * R0_global[:, 1]  # +Y
                z_end_world = origin_world_global + axis_len * R0_global[:, 2]  # +Z

                x_seg_global = np.stack([origin_world_global, x_end_world], axis=0)  # (2,3)
                y_seg_global = np.stack([origin_world_global, y_end_world], axis=0)  # (2,3)
                z_seg_global = np.stack([origin_world_global, z_end_world], axis=0)  # (2,3)

                server.scene.add_line_segments(
                    name="/theta_global/hips_axis_global/x",
                    points=x_seg_global[None, ...],  # (1,2,3)
                    line_width=4.0,
                    colors=(255, 0, 0),  # Red
                )
                server.scene.add_line_segments(
                    name="/theta_global/hips_axis_global/y",
                    points=y_seg_global[None, ...],  # (1,2,3)
                    line_width=4.0,
                    colors=(0, 255, 0),  # Green
                )
                server.scene.add_line_segments(
                    name="/theta_global/hips_axis_global/z",
                    points=z_seg_global[None, ...],  # (1,2,3)
                    line_width=4.0,
                    colors=(0, 0, 255),  # Blue
                )
        else:
            # If not selected, remove the global skeleton from the scene
            server.scene.remove_by_name("/theta_global/bones")
            server.scene.remove_by_name("/theta_global/joints")
            server.scene.remove_by_name("/theta_global/hips_axis_global/x")
            server.scene.remove_by_name("/theta_global/hips_axis_global/y")
            server.scene.remove_by_name("/theta_global/hips_axis_global/z")
            
        # ----------------

        # ----------------
        # A2. 6D Representation

        if bvh_6d_bool.value == True:
            # ====================================================================
            # [FIX] Determine the correct pose based on VALIDATION_CLIPS
            # ====================================================================
            if VALIDATION_CLIPS:
                pose_filepath = validation_files[current_frame_idx]
                rotation_vector = np.load(pose_filepath) # Shape: (144,)
                num_j = len(rotation_vector) // 6
                
                # 1. Load exactly like the model does
                rot6d_tensor = torch.from_numpy(rotation_vector).float().view(num_j, 6)
                
                # 2. Convert to 3x3 using the same function as the neural network
                rot_mats = sixd_to_rotmat(rot6d_tensor).numpy() # Shape: (24, 3, 3)
                
                # 3. Convert 3x3 directly to Quaternions using SciPy
                # SciPy uses [x, y, z, w], so we must shift 'w' back to the front for PyMotion
                quats_xyzw = Rotation.from_matrix(rot_mats).as_quat()
                current_quats = np.concatenate([quats_xyzw[:, 3:4], quats_xyzw[:, :3]], axis=1) # [w, x, y, z]
                
                current_root_pos = np.zeros(3) # Root is centered in processed clips
            else:
                current_quats = reconstructed_6d_local_rotations[current_frame_idx]
                current_root_pos = global_positions_6d[current_frame_idx, :]
            # ====================================================================

            # 1. Calculate the skeleton's pose relative to its own origin (0,0,0)
            # posed_joints_local_6d, _ = fk(reconstructed_6d_local_rotations[current_frame_idx], np.zeros(3), offsets_6d, parents_6d)
            posed_joints_local_6d, _ = fk(current_quats, np.zeros(3), offsets_6d, parents_6d)

            # 2. Convert to world space by adding the global root position (brings to original world position)
            # posed_joints_world_6d = posed_joints_local_6d + global_positions_6d[current_frame_idx, :]
            posed_joints_world_6d = posed_joints_local_6d + current_root_pos

            # Add the dynamic skeleton 
            bone_points_6d = []
            for i, p_idx in enumerate(parents_6d):
                if p_idx != -1:
                    # Each bone is a pair of [start_point, end_point]
                    bone_points_6d.append([posed_joints_world_6d[p_idx], posed_joints_world_6d[i]])

            # Convert to a single NumPy array of shape (num_bones, 2, 3)
            bone_points_6d = np.array(bone_points_6d)

            # Update bone segments
            server.scene.add_line_segments(
                name="/theta_6d/bones",
                points=bone_points_6d,
                line_width=3.0,
                colors=(0, 0, 255), # Blue  
            )

            # Update joint positions
            server.scene.add_point_cloud(
                name="/theta_6d/joints",
                points=posed_joints_world_6d,
                colors=(255, 255, 0), # Yellow
                point_size=0.015,
            )

            # Draw hips local coordinate axes (Y=green, Z=blue)
            if hips_axes_bool.value:
                axis_len = float(hips_axes_len.value)

                # Root rotation matrix from BVH quaternions
                pose_rotations_tensor_axes_6d = torch.from_numpy(reconstructed_6d_local_rotations[current_frame_idx]).float() # < ------------------------------------ TENSOR
                rot_mats_axes = quat_torch.to_matrix(pose_rotations_tensor_axes_6d)  # (J, 3, 3)
                R0_6d = rot_mats_axes[0].cpu().numpy()  # hips/global root

                origin_world_6d = posed_joints_world_6d[0]
                x_end_world_6d = origin_world_6d + axis_len * R0_6d[:, 0]  # +X
                y_end_world_6d = origin_world_6d + axis_len * R0_6d[:, 1]  # +Y
                z_end_world_6d = origin_world_6d + axis_len * R0_6d[:, 2]  # +Z

                x_seg_6d = np.stack([origin_world_6d, x_end_world_6d], axis=0)  # (2,3)
                y_seg_6d = np.stack([origin_world_6d, y_end_world_6d], axis=0)  # (2,3)
                z_seg_6d = np.stack([origin_world_6d, z_end_world_6d], axis=0)  # (2,3)

                server.scene.add_line_segments(
                    name="/theta_6d/hips_axis_6d/x",
                    points=x_seg_6d[None, ...],  # (1,2,3)
                    line_width=4.0,
                    colors=(255, 0, 0),  # Red
                )
                server.scene.add_line_segments(
                    name="/theta_6d/hips_axis_6d/y",
                    points=y_seg_6d[None, ...],  # (1,2,3)
                    line_width=4.0,
                    colors=(0, 255, 0),  # Green
                )
                server.scene.add_line_segments(
                    name="/theta_6d/hips_axis_6d/z",
                    points=z_seg_6d[None, ...],  # (1,2,3)
                    line_width=4.0,
                    colors=(0, 0, 255),  # Blue
                )
        else:
            # If not selected, remove the 6D skeleton from the scene
            server.scene.remove_by_name("/theta_6d/bones")
            server.scene.remove_by_name("/theta_6d/joints")
            server.scene.remove_by_name("/theta_6d/hips_axis_6d/x")
            server.scene.remove_by_name("/theta_6d/hips_axis_6d/y")
            server.scene.remove_by_name("/theta_6d/hips_axis_6d/z")

        # ----------------

        # ----------------
        # B. Residuals

        if residuals_gt_bool.value and residuals_gt_for_frame is not None:
            # p_bind is already rotated by DEGREE_BVH_X
            # residuals_gt_for_frame is the unrotated displacement from the .npy file.
            # To match the training script's p_gt = p_bind + residuals_gt, we add the unrotated
            # residuals to the already rotated p_bind.
            p_gt_tpose_rotated = p_bind + residuals_gt_for_frame

            server.scene.add_point_cloud(
                name="/GT_residuals/p_bind_gt",
                points=p_bind,
                colors=(0, 200, 0), # Green
                point_size=0.0025
            )

            # Color coding based on Mask
            gt_colors = np.zeros((p_gt_tpose_rotated.shape[0], 3), dtype=np.uint8)
            if current_mask is not None:
                gt_colors[current_mask == 1] = [0, 220, 0]  # Green for VALID
                gt_colors[current_mask == 0] = [220, 0, 0]  # Red for INVALID
            else:
                gt_colors[:] = [0, 220, 0] # Default green
                
            server.scene.add_point_cloud(
                name="/GT_residuals/p_bind_gt + residuals_gt_for_frame",
                points=p_gt_tpose_rotated,
                colors=gt_colors,
                point_size=0.0025
            )

            # Both points and p_bind_filtered should have shape (N, 3)
            # Stack them into (N, 2, 3) for line segments
            line_segments_tpose = np.stack([p_bind, p_gt_tpose_rotated], axis=1)  # shape: (N, 2, 3)
            server.scene.add_line_segments(
                name="/GT_residuals/correspondence_gt_changes_tpose_space",
                points=line_segments_tpose,
                line_width=2.0,
                colors=(255, 0, 255),  # Magenta
            )
        else:
            # If residuals are not enabled, remove the corresponding scene elements
            server.scene.remove_by_name("/GT_residuals/p_bind_gt")
            server.scene.remove_by_name("/GT_residuals/p_bind_gt + residuals_gt_for_frame")
            server.scene.remove_by_name("/GT_residuals/correspondence_gt_changes_tpose_space")
        # ----------------

        # ----------------
        # C. Inference

        if inference_bool.value == True:
            # A. Prepare constant identity pose (T-pose)
            identity_quat = np.zeros((j_rest_6d.shape[0], 4), dtype=np.float32)
            identity_quat[:, 0] = 1.0  # w=1, x=y=z=0
            rot_identity = quat_torch.to_matrix(torch.from_numpy(identity_quat).float().to(device)) 
            rot_identity_tensor = rot_identity.float().unsqueeze(0).to(device)  # (1, 24, 3, 3)
            
            # B. Prepare current pose from BVH (6D continuous representation)
            if VALIDATION_CLIPS:
                pose_filepath = validation_files[current_frame_idx]
                rotation_vector = np.load(pose_filepath) # Shape: (144,)
                num_joints_from_file = len(rotation_vector) // 6
                
                # Use a unique variable name (rot6d_tensor_inf) and ensure it has a batch dimension (1, 24, 6)
                rot6d_tensor_inf = torch.from_numpy(rotation_vector).float().view(1, num_joints_from_file, 6).to(device)
                
                root_position = torch.zeros(1, 3, device=device, dtype=s_bind.dtype)
            else:
                rot6d = continuous[current_frame_idx]  # (24, 3, 2)
                rot6d_flat = np.concatenate([rot6d[..., 0], rot6d[..., 1]], axis=-1)  # (24, 6)

                # Use a unique variable name (rot6d_tensor_inf)
                rot6d_tensor_inf = torch.from_numpy(rot6d_flat).float().unsqueeze(0).to(device)  # (1, 24, 6)
                
                root_position = torch.from_numpy(local_positions_6d[current_frame_idx, 0, :]).float().to(device).unsqueeze(0)  # (1, 3)

            rot_mats = sixd_to_rotmat(rot6d_tensor_inf) # (B, J, 3, 3)
            
            # C. Root position
            if VALIDATION_CLIPS:
                root_position = torch.zeros(1, 3, device=device, dtype=s_bind.dtype)
            else: 
                root_position = torch.from_numpy(local_positions_6d[current_frame_idx, 0, :]).float().to(device).unsqueeze(0)  # (1, 3)

            # Forward pass - returns dictionary
            with torch.no_grad():
                forward_output = model(rot6d_tensor_inf, rot_identity_tensor, root_position)

                p_pred_theta = forward_output['p_pred_theta']        # (B, M, 3) - predicted markers in pose space
                p_pred_tpose = forward_output['p_pred_tpose']        # (B, M, 3) - predicted markers in t-pose
                s_final_theta = forward_output['s_final_theta']      # (B, V, 3) - skin vertices in pose space
                m_final_theta = forward_output['m_final_theta']      # (B, V, 3) - muscle vertices in pose space
                s_final = forward_output['s_final']      # (B, V, 3) - skin vertices in pose space
                m_final = forward_output['m_final']      # (B, V, 3) - muscle vertices in pose space
                d_skin = forward_output['d_skin']                    # (B, V, 3) - skin displacement
                d_muscle = forward_output['d_muscle']                # (B, V, 3) - muscle displacement
                p_bind_theta = forward_output['p_bind_theta']        # (B, M, 3) - bind markers in pose space
                
                # ====================================================================================
                # --- ANIMATION RECORDING LOGIC ---
                # ====================================================================================
                if record_bool.value:
                    # Grab the current frame from the slider
                    frame_idx = frame_slider.value
                    out_dir = export_dir.value
                    os.makedirs(out_dir, exist_ok=True)
                    
                    # Use the clean helper function!
                    if export_skin.value:
                        skin_path = os.path.join(out_dir, f"skin_frame_{frame_idx:04d}.obj")
                        save_obj_fast(skin_path, s_final[0].cpu().numpy(), s_bind_faces.cpu().numpy())
                    
                    if export_muscle.value:
                        muscle_path = os.path.join(out_dir, f"muscle_frame_{frame_idx:04d}.obj")
                        save_obj_fast(muscle_path, m_final[0].cpu().numpy(), m_bind_faces.cpu().numpy())
                        
                    # if export_bone.value:
                    #     bone_path = os.path.join(out_dir, f"bone_frame_{frame_idx:04d}.obj")
                    #     save_obj_fast(bone_path, b_bind[0].cpu().numpy(), b_bind_faces.cpu().numpy())  

                    # --------------------------------------------------------------------------------
                    # NEW: Export Unified Individual Muscles
                    # --------------------------------------------------------------------------------
                    if export_unified_indiv.value and skin_verts_raw is not None and muscle_bary_map is not None:
                        skin_verts_deformed = s_final.detach().cpu().numpy()[0]
                        all_unified_verts = []
                        all_unified_faces = []
                        vertex_offset = 0

                        unified_ranges = {}
                        
                        for m_name, m_data in preloaded_muscles.items():
                            json_key = muscle_name_mapping.get(m_name, m_name)
                            muscle_verts_raw = m_data['verts']
                            muscle_faces_raw = m_data['faces']
                            muscle_verts_deformed = muscle_verts_raw.copy()
                            
                            # Apply Deformation
                            if json_key in muscle_bary_map:
                                target_data = muscle_bary_map[json_key]
                                for k, data in target_data.items():
                                    try:
                                        local_idx = int(k)
                                        if local_idx < len(muscle_verts_raw):
                                            tri_idx = data['tri_idx']
                                            bary = data['bary']
                                            if tri_idx < len(skin_faces_raw):
                                                f_indices = skin_faces_raw[tri_idx]
                                                
                                                # Rest vs Deformed
                                                v1_r, v2_r, v3_r = skin_verts_raw[f_indices]
                                                p_skin_rest = (bary[0]*v1_r) + (bary[1]*v2_r) + (bary[2]*v3_r)
                                                
                                                v1_d, v2_d, v3_d = skin_verts_deformed[f_indices]
                                                p_skin_deformed = (bary[0]*v1_d) + (bary[1]*v2_d) + (bary[2]*v3_d)
                                                
                                                # Transfer Delta
                                                muscle_verts_deformed[local_idx] += (p_skin_deformed - p_skin_rest)
                                    except: pass
                            
                            # Track ranges before appending
                            v_start = vertex_offset
                            v_end = vertex_offset + len(muscle_verts_deformed)
                            unified_ranges[m_name] = [v_start, v_end] # <--- NEW: Store the range

                            # Append to our master list, shifting the face indices to prevent tearing
                            all_unified_verts.append(muscle_verts_deformed)
                            all_unified_faces.append(muscle_faces_raw + vertex_offset)
                            vertex_offset += len(muscle_verts_deformed)
                            
                        # Stack and Export
                        if all_unified_verts:
                            unified_verts_np = np.vstack(all_unified_verts)
                            unified_faces_np = np.vstack(all_unified_faces)
                            unified_path = os.path.join(out_dir, f"indiv_muscles_unified_frame_{frame_idx:04d}.obj")
                            save_obj_fast(unified_path, unified_verts_np, unified_faces_np)

                            if frame_idx == 1: 
                                import json
                                with open(os.path.join(out_dir, "unified_muscle_ranges.json"), "w") as f:
                                    json.dump(unified_ranges, f)
                    # --------------------------------------------------------------------------------                      
                    
                    logger.info(f"[RECORDING] Saved frame {frame_idx:04d}")
                    
                # ==========================================
                # NEW: Capture current T-Pose meshes for exporter
                # ==========================================
                export_state["m_final_tpose"] = m_final[0].detach().cpu().numpy()
                export_state["s_final_tpose"] = s_final[0].detach().cpu().numpy()
                export_state["frame_idx"] = frame_slider.value

                # ====================================================================================
                # [NEW] INDIVIDUAL MUSCLE DEFORMATION TRANSFER & VISUALIZATION
                # ====================================================================================
                if show_individual_muscles_bool.value or export_individual_muscles_bool.value:
                    
                    # 1. Prepare Deformed Skin
                    skin_verts_deformed = s_final.detach().cpu().numpy()[0]
                    
                    if skin_verts_raw is not None and muscle_bary_map is not None:
                        target_name = target_muscle_text.value.strip()
                        
                        if export_individual_muscles_bool.value:
                            frame_export_dir = os.path.join(export_dir_ind_muscle, f"frame_{current_frame_idx:04d}")
                            os.makedirs(frame_export_dir, exist_ok=True)
                        
                        processed_count = 0

                        for m_name, m_data in preloaded_muscles.items():
                            json_key = muscle_name_mapping.get(m_name, m_name)
                            
                            # Filter logic
                            should_show = (not target_name) or (target_name in m_name)
                            
                            if not should_show and not export_individual_muscles_bool.value:
                                server.scene.remove_by_name(f"/model/tpose/deformed_muscles/{m_name}")
                                server.scene.remove_by_name(f"/model/tpose/deformed_muscles_wireframe/{m_name}")
                                continue
                                
                            try:
                                # A. Rest Pose Data
                                muscle_verts_raw = m_data['verts']
                                muscle_faces_raw = m_data['faces']
                                muscle_verts_deformed = muscle_verts_raw.copy()
                                
                                # B. Apply Deltas via Barycentric JSON Map
                                if json_key in muscle_bary_map:
                                    target_data = muscle_bary_map[json_key]
                                    
                                    for k, data in target_data.items():
                                        try:
                                            local_idx = int(k)
                                            if local_idx < len(muscle_verts_raw):
                                                tri_idx = data['tri_idx']
                                                bary = data['bary']
                                                
                                                if tri_idx < len(skin_faces_raw):
                                                    f_indices = skin_faces_raw[tri_idx]
                                                    
                                                    # Position on REST Skin
                                                    v1_r, v2_r, v3_r = skin_verts_raw[f_indices]
                                                    p_skin_rest = (bary[0] * v1_r) + (bary[1] * v2_r) + (bary[2] * v3_r)
                                                    
                                                    # Position on DEFORMED Skin (s_final)
                                                    v1_d, v2_d, v3_d = skin_verts_deformed[f_indices]
                                                    p_skin_deformed = (bary[0] * v1_d) + (bary[1] * v2_d) + (bary[2] * v3_d)
                                                    
                                                    # Transfer Delta
                                                    delta = p_skin_deformed - p_skin_rest
                                                    muscle_verts_deformed[local_idx] += delta
                                        except: pass
                                    
                                    # C. Export
                                    if export_individual_muscles_bool.value:
                                        out_path = os.path.join(frame_export_dir, f"{m_name}.obj")
                                        tm = trimesh.Trimesh(vertices=muscle_verts_deformed, faces=muscle_faces_raw, process=False)
                                        tm.export(out_path)
                                        processed_count += 1
                                    
                                    # D. Visualize
                                    if show_individual_muscles_bool.value and should_show:
                                        import hashlib
                                        hash_obj = hashlib.md5(m_name.encode())
                                        r = int(hash_obj.hexdigest()[0:2], 16)
                                        g = int(hash_obj.hexdigest()[2:4], 16)
                                        b = int(hash_obj.hexdigest()[4:6], 16)
                                        
                                        server.scene.add_mesh_simple(
                                            name=f"/model/tpose/deformed_muscles/{m_name}",
                                            vertices=muscle_verts_deformed,
                                            faces=muscle_faces_raw,
                                            color=(r, g, b),
                                            opacity=1,
                                            wireframe=False,
                                            side="double"
                                        )
                                        server.scene.add_mesh_simple(
                                            name=f"/model/tpose/deformed_muscles_wireframe/{m_name}",
                                            vertices=muscle_verts_deformed,
                                            faces=muscle_faces_raw,
                                            color=(50, 0, 0),
                                            opacity=0.3,
                                            wireframe=True,
                                            side="double"
                                        )
                                    else:
                                        server.scene.remove_by_name(f"/model/tpose/deformed_muscles/{m_name}")
                                        server.scene.remove_by_name(f"/model/tpose/deformed_muscles_wireframe/{m_name}")
                            except Exception as e:
                                pass
                                
                        if export_individual_muscles_bool.value:
                            logger.success(f"[EXPORT] Saved {processed_count} muscles to {frame_export_dir}")
                            export_individual_muscles_bool.value = False # Auto-toggle off after 1 frame to prevent spam
                else:
                    # Cleanup scene if toggled off
                    for m_name in preloaded_muscles.keys():
                        server.scene.remove_by_name(f"/model/tpose/deformed_muscles/{m_name}")
                        server.scene.remove_by_name(f"/model/tpose/deformed_muscles_wireframe/{m_name}")
                # ====================================================================================

                # [DEBUG] Verify Topology Match
                # ---------------------------------------------------------
                num_v_bind = musc_layer.vertices.shape[0]
                num_f_bind = musc_layer.faces.shape[0]
                
                num_v_final = m_final.shape[1] # m_final is (Batch, Vertices, 3)
                # Note: m_final does not have faces, it uses the bind faces.
                
                if num_v_bind != num_v_final:
                    logger.error(f"[MISMATCH] VERTEX COUNT ERROR!")
                    logger.error(f"  > m_bind  : {num_v_bind}")
                    logger.error(f"  > m_final : {num_v_final}")
                else:
                    pass

                # Verify Face Count (Logic Check)
                # Since we export using `musc_layer.faces`, the face count is guaranteed to be `num_f_bind`.
                # The only risk is if `m_final` has fewer vertices than `faces` indices require (which would crash).
                if musc_layer.faces.max() >= num_v_final:
                    logger.error(f"[MISMATCH] FATAL: Face indices reference vertices > {num_v_final}!")
                else:
                    pass
                # ---------------------------------------------------------

                # ---------------------------------------------------------
                # Process markers for S1
                markers_to_remove = [933, 1320, 1327, 1961]
                def extract_marker_int(marker_id):
                    # Assumes marker_id is like 'marker_103_0_0'
                    try:
                        return int(marker_id.split('_')[1])
                    except Exception:
                        return None
                # ---------------------------------------------------------

                # Only mask out markers whose integer part is in markers_to_remove
                mask = np.array([extract_marker_int(mid) not in markers_to_remove for mid in canonical_marker_ids])
                mask[markers_to_remove] = False

                p_bind_theta_filtered = p_bind_theta.squeeze(0).cpu().numpy()[mask]
                p_pred_theta_filtered = p_pred_theta.squeeze(0).cpu().numpy()[mask]
                p_pred_tpose_filtered = p_pred_tpose.squeeze(0).cpu().numpy()[mask]
                p_bind_np = p_bind[mask]

                if output_final_theta_bool.value == True:
                    server.scene.add_mesh_simple(
                        name="/model/theta/m_final",
                        vertices=m_final_theta.cpu().numpy(),
                        faces=musc_layer.faces,
                        color=(200, 0, 0),  # Red
                        wireframe = False
                    )

                    server.scene.add_mesh_simple(
                        name="/model/theta/s_final",
                        vertices=s_final_theta.cpu().numpy(),
                        faces=skin_layer.faces,
                        color=(120, 158, 240), # Blue
                        opacity=0.8,
                        wireframe = False
                    )   

                    server.scene.add_point_cloud(
                        name="/model/theta/p_pred_theta_filtered",
                        points=p_pred_theta_filtered.squeeze(0).cpu().numpy(),
                        colors=(150, 150, 255), # Cyan
                        point_size=0.003
                    )

                    # Recompute and display delta_pred in pose space
                    delta_pred = p_pred_theta_filtered - p_bind_theta_filtered
                    points = p_bind_theta_filtered + delta_pred

                    server.scene.add_point_cloud(
                        name="/model/theta/p_bind_theta + delta_pred",
                        points=points,
                        colors=(0, 0, 200), # Blue
                        point_size=0.0025,
                    )

                    # Both points and p_bind_filtered should have shape (N, 3)
                    # Stack them into (N, 2, 3) for line segments
                    line_segments = np.stack([p_bind_theta_filtered, points], axis=1)  # shape: (N, 2, 3)
                    server.scene.add_line_segments(
                        name="/model/theta/correspondence_changes_pose_space",
                        points=line_segments,
                        line_width=2.0,
                        colors=(255, 0, 255),  # Magenta
                    )

                else:
                    server.scene.remove_by_name("/model/theta/m_final")
                    server.scene.remove_by_name("/model/theta/s_final")
                    server.scene.remove_by_name("/model/theta/p_pred_theta_filtered")
                    server.scene.remove_by_name("/model/theta/p_bind_theta + delta_pred")
                    server.scene.remove_by_name("/model/theta/correspondence_changes_pose_space")

                if output_final_tpose_bool.value == True:
                    server.scene.add_mesh_simple(
                        name="/model/tpose/m_final",
                        vertices=m_final.cpu().numpy(),
                        faces=musc_layer.faces,
                        color=(200, 0, 0),  # Red
                        wireframe = True
                    )

                    server.scene.add_mesh_simple(
                        name="/model/tpose/s_final",
                        vertices=s_final.cpu().numpy(),
                        faces=skin_layer.faces,
                        color=(120, 158, 240), # Blue
                        opacity=0.8,
                        wireframe = True
                    )

                    # PLOTTING B_BIND instead of B_FINAL (DOES NOT EXIST)
                    server.scene.add_mesh_simple(
                        name="/model/tpose/b_final",
                        vertices=b_bind.cpu().numpy(),
                        faces=bone_layer.faces,
                        color=(220, 220, 220), # Beige
                        wireframe = True
                    )

                    # Estimate hips position offset pointcloud
                    hips_idx = 0  # usually 0, but check your skeleton
                    hips_pos = j_rest[hips_idx]  # (3,)
                    if torch.is_tensor(hips_pos):
                        hips_pos = hips_pos.cpu().numpy()
                    
                    points_tpose_canonical = p_pred_tpose_filtered + hips_pos

                    # [FIX] Filter the current_mask to match the filtered points (2306 -> 2302)
                    current_mask_filtered = current_mask[mask] if current_mask is not None else None
                    pred_colors = np.zeros((points_tpose_canonical.shape[0], 3), dtype=np.uint8)

                    # [NEW] Color coding based on Mask for PREDICTED markers
                    # This shows us if the prediction corresponds to valid data (Green) or missing data (Red)
                    pred_colors = np.zeros((points_tpose_canonical.shape[0], 3), dtype=np.uint8)
                    if current_mask_filtered is not None:
                        pred_colors[current_mask_filtered == 1] = [0, 0, 255]  # Blue for VALID (GT exists)
                        # pred_colors[current_mask_filtered == 0] = [255, 200, 200]  # Red for INVALID (GT missing/inferred)
                        pred_colors[current_mask_filtered == 0] = [255, 0, 0]  # Red for INVALID (GT missing/inferred)
                        # pred_colors[current_mask_filtered == 0] = [255, 255, 255]  # Red for INVALID (GT missing/inferred)
                    else:
                        pred_colors[:] = [0, 0, 255] # Default Blue if mask unavailable

                    server.scene.add_point_cloud(
                        name="/model/tpose/p_bind_np + delta_pred_tpose",
                        points=points_tpose_canonical,
                        colors=pred_colors,
                        point_size=0.0025,
                    )

                    # Both points and p_bind_filtered should have shape (N, 3)
                    # Stack them into (N, 2, 3) for line segments
                    line_segments_tpose = np.stack([p_bind_np, points_tpose_canonical], axis=1)  # shape: (N, 2, 3)
                    server.scene.add_line_segments(
                        name="/model/tpose/correspondence_changes_tpose_space",
                        points=line_segments_tpose,
                        line_width=2.0,
                        colors=(255, 0, 255),  # Magenta
                    )

                else:
                    server.scene.remove_by_name("/model/tpose/m_final")
                    server.scene.remove_by_name("/model/tpose/s_final")
                    server.scene.remove_by_name("/model/tpose/b_final")
                    server.scene.remove_by_name("/model/tpose/p_bind_np + delta_pred_tpose")
                    server.scene.remove_by_name("/model/tpose/correspondence_changes_tpose_space")


    # Attach the update function to all GUI elements that should trigger a refresh
    @frame_slider.on_update
    @bvh_global_bool.on_update
    @bvh_6d_bool.on_update
    @hips_axes_bool.on_update
    @residuals_gt_bool.on_update
    @output_final_theta_bool.on_update
    @inference_bool.on_update
    @output_final_tpose_bool.on_update
    @output_final_theta_bool.on_update
    @show_individual_muscles_bool.on_update     
    @export_individual_muscles_bool.on_update    
    @target_muscle_text.on_update               
    def _(event: viser.GuiEvent) -> None:
            # 1. Try to get the camera from the specific client interacting with the GUI
            camera = None
            if event.client is not None:
                camera = event.client.camera
            
            # 2. Fallback: If no client (e.g. during auto-play), grab the first connected client
            if camera is None:
                clients = server.get_clients()
                if len(clients) > 0:
                    # Dictionary values are the clients
                    camera = list(clients.values())[0].camera

            # 3. Call update with the found camera
            update_scene(camera)

    # Initial call to update_scene, handling potential lack of clients
    if server.get_clients():
        # If clients are already connected, get the camera from the first one
        first_client_camera = list(server.get_clients().values())[0].camera
        update_scene(first_client_camera)
    else:
        # If no clients yet, call with None. The update_scene function handles camera_handle=None.
        update_scene(None)

    # ========================================================================================================================================================
    # 6. LOOP
    logger.info("[INFO] 6. STARTING LOOP...")
    # ========================================================================================================================================================

    last_update_time = time.time()

    while True:
        # Auto-play Logic
        if play_bool.value:
            current_time = time.time()
            # Convert ms to seconds
            wait_time = playback_speed.value / 1000.0
            
            if (current_time - last_update_time) >= wait_time:
                # Advance frame
                current_frame = frame_slider.value
                max_frame = frame_slider.max
                
                next_frame = current_frame + 1
                if next_frame > max_frame:
                    next_frame = 0 # Loop back to start
                
                frame_slider.value = next_frame
                last_update_time = current_time

        time.sleep(0.1)  # Keep the server alive without burning CPU

if __name__ == "__main__":
    main()