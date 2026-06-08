import os
import glob
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from tqdm import tqdm

import trimesh
from trimesh.smoothing import laplacian_calculation

from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree
import scipy.sparse as sp

import viser
import viser.transforms as tf
from pymotion.io.bvh import BVH
from pymotion.ops.skeleton import fk
import pymotion.rotations.quat as quat
import pymotion.rotations.quat_torch as quat_torch
import pymotion.ops.skeleton as sk
import pymotion.rotations.ortho6d as sixd

from typing import Optional

# Quick class to calculate prisms volume
class PrismVolumeLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Gauss-Legendre Quadrature Constants
        self.inv_sqrt3 = 0.5773502691896257
        self.t1 = 0.5 * (1.0 - self.inv_sqrt3)
        self.t2 = 0.5 * (1.0 + self.inv_sqrt3)
        self.alpha = 1.0 / 3.0
        self.beta  = 1.0 / 3.0
        self.gamma = 1.0 / 3.0

    def compute_prism_volume(self, inner_verts, outer_verts, faces):
        """
        Calculates exact prism volume using 2-point Gauss Quadrature.
        inner_verts: (B, V, 3) or (V, 3) - e.g., Bone or Muscle
        outer_verts: (B, V, 3) or (V, 3) - e.g., Muscle or Skin
        faces: (F, 3) - Corresponding valid topology
        """
        f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]

        # Gather vertices: (..., F, 3)
        x1 = inner_verts[..., f0, :]
        x2 = inner_verts[..., f1, :]
        x3 = inner_verts[..., f2, :]
        x4 = outer_verts[..., f0, :]
        x5 = outer_verts[..., f1, :]
        x6 = outer_verts[..., f2, :]

        def compute_detJ(xi_scalar):
            # 1. Derivatives wrt local coords (a, b)
            dx_da = xi_scalar * (x1 - x3) + (1.0 - xi_scalar) * (x4 - x6)
            dx_db = xi_scalar * (x2 - x3) + (1.0 - xi_scalar) * (x5 - x6)
            
            # 2. Derivative wrt thickness (xi)
            dx_dxi = self.alpha * (x1 - x4) + self.beta * (x2 - x5) + self.gamma * (x3 - x6)
            
            # 3. Determinant of Jacobian via Scalar Triple Product
            cross_prod = torch.cross(dx_db, dx_dxi, dim=-1)
            return torch.sum(dx_da * cross_prod, dim=-1)

        detJ_1 = compute_detJ(self.t1)
        detJ_2 = compute_detJ(self.t2)
        
        # No .abs() here. We want negative volume if inverted!
        return 0.25 * (detJ_1 + detJ_2)

    def forward(self, curr_inner_verts, curr_outer_verts, rest_volume, faces):
        # 1. Compute current signed volume
        curr_vol = self.compute_prism_volume(curr_inner_verts, curr_outer_verts, faces)
        # Squeeze rest_volume to 1D (F,)
        rest_vol_1d = rest_volume.squeeze()        

        # Safety mask: Ignore perfectly welded/flat geometry to prevent division by zero
        valid_mask = rest_vol_1d.abs() > 1e-7
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=curr_inner_verts.device)

        curr_vol_valid = curr_vol[:, valid_mask]
        rest_vol_valid = rest_vol_1d[valid_mask]
        
        # 2. Calculate Absolute Volume Difference: (V - V0)
        diff = curr_vol_valid - rest_vol_valid        

        # 3. Create a Numerically Safe Denominator
        # We add 10% of the mean absolute volume to the denominator. 
        # For thick muscles, |V0| dominates (matching the paper perfectly).
        # For paper-thin tissues, it prevents division by zero/singularity explosions.
        mean_vol = rest_vol_valid.abs().mean().clamp(min=1e-8)
        safe_denom = rest_vol_valid.abs() + (0.1 * mean_vol)
        
        # 4. Final Energy Calculation: (V - V0)^2 / (|V0| + epsilon)
        weighted_error = diff.pow(2) / safe_denom
        
        # MEAN over the mesh, Mean over the batch
        return weighted_error.mean()

# --- CONFIGURATION (Match your training config) ---
# Fixed system config (non-CLI settings)
CONFIG = {
    "base_path_suffix": "static00",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# CHECKPOINT REFERENCE TABLE #
# -------------------------- #
# Usage: pass the desired path to --checkpoint when running this script, e.g.:
#   python 03_evaluate_metrics.py --subject S1 --checkpoint CHECKPOINTS["arch_mlp_vol10"]["path"]

CHECKPOINTS = {
    # --- Development v2 (weight decay=0, final progression) ---
    # Stage 1: E_data + E_smooth only
    "smoothness": {
        "path": "./checkpoints/01_adding_smoothness/full_MLP_w_data_1_w_smooth_musc_1_w_smooth_skin_5_w_spring_musc_0_w_spring_skin_0_w_tangent_musc_0_w_tangent_skin_0_epoch_9.pth",
        "result": "Deformation visible but spiky",
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
    # Stage 7: + E_vol on both muscle and skinQ
    "volume_both": {
        "path": "./checkpoints/07_adding_volume_both/full_MLP_w_data_50_w_smooth_musc_100_w_smooth_skin_200_w_spring_musc_1000_w_spring_skin_1000_w_tangent_musc_1_w_tangent_skin_01_w_biharmonic_musc_100_w_biharmonic_skin_200_w_vol_musc_10_w_vol_skin_10_epoch_5.pth",
        "result": "Final",
    },
    # Stage 8: + E_vol on both, with mask
    "volume_both_mask": {
        "path": "./checkpoints/08_adding_volume_both_clean/full_MLP_w_data_50_w_smooth_musc_100_w_smooth_skin_200_w_spring_musc_1000_w_spring_skin_1000_w_tangent_musc_1_w_tangent_skin_01_w_biharmonic_musc_100_w_biharmonic_skin_200_w_vol_musc_10_w_vol_skin_10_epoch_5.pth",
        "result": "Final (with mask)",
    },
    # --- Architecture ablations (09_architectures/) ---
    "arch_linear_vol10": {
        "path": "./checkpoints/09_architectures/ABLATION_LINEAR_w_data_50_w_smooth_musc_100_w_smooth_skin_200_w_spring_musc_1000_w_spring_skin_1000_w_tangent_musc_1_w_tangent_skin_01_w_biharmonic_musc_100_w_biharmonic_skin_200_w_vol_musc_10_w_vol_skin_10_epoch_11.pth",
        "result": "",
    },
    "arch_linear_vol1000": {
        "path": "./checkpoints/09_architectures/ABLATION_LINEAR_w_data_50_w_smooth_musc_100_w_smooth_skin_200_w_spring_musc_1000_w_spring_skin_1000_w_tangent_musc_1_w_tangent_skin_01_w_biharmonic_musc_100_w_biharmonic_skin_200_w_vol_musc_1000_w_vol_skin_1000_epoch_4.pth",
        "result": "",
    },
    "arch_mlp_vol10": {
        "path": "./checkpoints/09_architectures/ABLATION_MLP_w_data_50_w_smooth_musc_100_w_smooth_skin_200_w_spring_musc_1000_w_spring_skin_1000_w_tangent_musc_1_w_tangent_skin_01_w_biharmonic_musc_100_w_biharmonic_skin_200_w_vol_musc_10_w_vol_skin_10_epoch_18.pth",
        "result": "",
    },
    "arch_mlp_vol1000": {
        "path": "./checkpoints/09_architectures/ABLATION_MLP_w_data_50_w_smooth_musc_100_w_smooth_skin_200_w_spring_musc_1000_w_spring_skin_1000_w_tangent_musc_1_w_tangent_skin_01_w_biharmonic_musc_100_w_biharmonic_skin_200_w_vol_musc_1000_w_vol_skin_1000_epoch_17.pth",
        "result": "",
    },
    "arch_unet_vol10": {
        "path": "./checkpoints/09_architectures/ABLATION_UNET_w_data_50_w_smooth_musc_100_w_smooth_skin_200_w_spring_musc_1000_w_spring_skin_1000_w_tangent_musc_1_w_tangent_skin_01_w_biharmonic_musc_100_w_biharmonic_skin_200_w_vol_musc_10_w_vol_skin_10_epoch_3.pth",
        "result": "",
    },
    "arch_unet_vol1000": {
        "path": "./checkpoints/09_architectures/ABLATION_UNET_w_data_50_w_smooth_musc_100_w_smooth_skin_200_w_spring_musc_1000_w_spring_skin_1000_w_tangent_musc_1_w_tangent_skin_01_w_biharmonic_musc_100_w_biharmonic_skin_200_w_vol_musc_1000_w_vol_skin_1000_epoch_2.pth",
        "result": "",
    },
    # --- Prior ablations (10_priors/) ---
    "prior_no_smooth_tan": {
        "path": "./checkpoints/10_priors/PRIOR_ABLATION_MLP_w_data_1_w_smooth_musc_0_w_smooth_skin_0_w_spring_musc_1_w_spring_skin_1_w_tangent_musc_0_w_tangent_skin_0_w_biharmonic_musc_1_w_biharmonic_skin_25_w_vol_musc_10_w_vol_skin_10_epoch_3.pth",
        "result": "Done",
    },
    "prior_no_physics": {
        "path": "./checkpoints/10_priors/PRIOR_ABLATION_MLP_w_data_1_w_smooth_musc_001_w_smooth_skin_005_w_spring_musc_0_w_spring_skin_0_w_tangent_musc_1_w_tangent_skin_01_w_biharmonic_musc_0_w_biharmonic_skin_0_w_vol_musc_0_w_vol_skin_0_epoch_2.pth",
        "result": "Done",
    },
    "prior_no_vol": {
        "path": "./checkpoints/05_changing_tan/full_MLP_w_data_1_w_smooth_musc_001_w_smooth_skin_005_w_spring_musc_1_w_spring_skin_1_w_tangent_musc_1_w_tangent_skin_01_w_biharmonic_musc_1_w_biharmonic_skin_25_w_vol_musc_0_w_vol_skin_0_epoch_6.pth",
        "result": "Done",
    },
    "prior_no_bi_stretch": {
        "path": "./checkpoints/10_priors/PRIOR_ABLATION_MLP_w_data_1_w_smooth_musc_001_w_smooth_skin_005_w_spring_musc_0_w_spring_skin_0_w_tangent_musc_1_w_tangent_skin_01_w_biharmonic_musc_0_w_biharmonic_skin_0_w_vol_musc_10_w_vol_skin_10_epoch_3.pth",
        "result": "",
    },
}

# PATH SETUP is deferred to main() after argparse resolves SUBJECT.

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
        bone_weights = torch.zeros_like(self.skin_weights)
        bone_weights.scatter_(1, self.skin_weights.argmax(dim=1, keepdim=True), 1.0)

        b_bind_batch = self.b_bind.unsqueeze(0).expand(batch_size, -1, -1)
        # b_final_theta = lbs_working_batch_rotmat(b_bind_batch, rot_mats, self.skin_weights, self.j_rest, self.parents, root_positions)
        b_final_theta = lbs_working_batch_rotmat(b_bind_batch, rot_mats, bone_weights, self.j_rest, self.parents, root_positions)        
        
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
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SOMA checkpoint with interactive Viser visualization."
    )
    parser.add_argument(
        "--subject", type=str, required=True,
        help="Subject identifier (e.g. S1, S4)."
    )
    parser.add_argument(
        "--shot", type=str, default="shot_001",
        help="Shot identifier to evaluate (default: shot_001)."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the .pth checkpoint file to evaluate."
    )
    args = parser.parse_args()

    SUBJECT         = args.subject
    SHOT            = args.shot
    CHECKPOINT_PATH = args.checkpoint

    logger.info("--- STARTING INTERACTIVE VALIDATION ---")
    logger.info(f"Subject:    {SUBJECT}")
    logger.info(f"Shot:       {SHOT}")
    logger.info(f"Checkpoint: {CHECKPOINT_PATH}")

    device = torch.device(CONFIG["device"])
    logger.info(f"Device:     {device}")

    BASE_DIR = f"/CT/SOMA/{CONFIG['base_path_suffix']}/{SUBJECT}"
    PATHS = {
        "raw":          os.path.join(BASE_DIR, "raw"),
        "processed":    os.path.join(BASE_DIR, "preprocessed_vFinal_clean"),
        "layers_tpose": os.path.join(BASE_DIR, "layers", "tpose"),
        "canonical":    os.path.join(BASE_DIR, "canonical_model"),
    }

    # ========================================================================================================================================================
    # 1. Configure paths
    logger.info("[INFO] 1. CONFIGURING PATHS...")
    # ========================================================================================================================================================
    
    # ---------------------------------------------------------
    # A. BVH file

    logger.info("[BVH] FINDING SAMPLE BVH...")

    bvh_path = glob.glob(os.path.join(PATHS['raw'], f"{SHOT}_captury", f"{SUBJECT}_{SHOT}.bvh"))[0]

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

    if SUBJECT == "S1":
        SCALE = 1.0
    else:
        SCALE = 0.001

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

    if SUBJECT == "S1":
        SCALE = 1.0
    else:
        SCALE = 0.001

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
    angles_6d = np.array([0, np.pi / 2, 0])[..., np.newaxis] # angles.shape = [3, 1]
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
    # 4. QUANTITATIVE EVALUATION (HEADLESS)
    logger.info("[INFO] 4. STARTING QUANTITATIVE EVALUATION...")
    # ========================================================================================================================================================
    
    # Initialize the volume calculator
    prism_calc = PrismVolumeLoss().to(device)
    
    # 1. Pre-compute Rest Volumes
    with torch.no_grad():
        rest_vol_skin = prism_calc.compute_prism_volume(m_bind.unsqueeze(0), s_bind.unsqueeze(0), s_faces)
        rest_vol_musc = prism_calc.compute_prism_volume(b_bind.unsqueeze(0), m_bind.unsqueeze(0), m_faces)
        
        # Valid prism masks (ignore degenerate areas like face/hands)
        valid_fat_mask = rest_vol_skin.abs() > 1e-7
        valid_musc_mask = rest_vol_musc.abs() > 1e-7

    # 2. Setup accumulators
    all_mpme_mm = []
    all_med_pme_mm = []
    all_p90_pme_mm = []

    # NEW: LBS Accumulators
    all_lbs_mpme_mm = []
    all_lbs_med_pme_mm = []
    all_lbs_p90_pme_mm = []

    all_dynamic_pme = []
    all_lbs_dynamic_pme = []

    all_dynamic_15_soma, all_dynamic_15_lbs = [], []
    all_dynamic_20_soma, all_dynamic_20_lbs = [], []
    all_dynamic_25_soma, all_dynamic_25_lbs = [], []

    # SOMA Accumulators
    all_vol_fat_change = []
    all_vol_musc_change = []
    all_intersection_fat = []
    all_intersection_musc = []

    all_vol_musc_change_p90 = []
    all_vol_musc_change_lbs_p90 = []

    # LBS Accumulators
    all_vol_fat_change_lbs = []
    all_vol_musc_change_lbs = []
    all_intersection_fat_lbs = []
    all_intersection_musc_lbs = []

    # Markers to ignore (from your S1 logic)
    markers_to_remove = [933, 1320, 1327, 1961]
    def extract_marker_int(marker_id):
        try: return int(marker_id.split('_')[1])
        except Exception: return None
        
    base_mask = np.array([extract_marker_int(mid) not in markers_to_remove for mid in canonical_marker_ids])

    logger.info(f"Evaluating {len(validation_files)} unseen frames...")
    
    # 3. Evaluation Loop
    with torch.no_grad():
        for frame_idx, pose_filepath in enumerate(tqdm(validation_files, desc="Evaluating Sequence")):
            
            # --- Load Data ---
            rotation_vector = np.load(pose_filepath) 
            num_j = len(rotation_vector) // 6
            rot6d_tensor = torch.from_numpy(rotation_vector).float().view(1, num_j, 6).to(device)
            rot_mats = sixd_to_rotmat(rot6d_tensor)
            
            identity_quat = np.zeros((num_j, 4), dtype=np.float32)
            identity_quat[:, 0] = 1.0  
            rot_identity_tensor = quat_torch.to_matrix(torch.from_numpy(identity_quat).float().to(device)).unsqueeze(0)
            
            root_position = torch.zeros(1, 3, device=device, dtype=s_bind.dtype)
            
            # --- Load GT Residuals & Mask ---
            base_filename = os.path.basename(pose_filepath)
            res_path = os.path.join(PATHS['processed'], 'residuals', base_filename)
            mask_path = os.path.join(PATHS['processed'], 'masks', base_filename)
            
            residuals_gt = np.load(res_path).astype(np.float32).reshape(-1, 3)
            current_mask = np.load(mask_path).reshape(-1)
            
            # Zero out missing data in GT
            residuals_gt[current_mask == 0] = 0.0
            residuals_gt_tensor = torch.from_numpy(residuals_gt).float().unsqueeze(0).to(device)
            
            # Combine missing data mask with specific markers to remove



            # --- Forward Pass ---
            out = model(rot6d_tensor, rot_identity_tensor, root_position)
            
            # =================================================================
            # Evaluate ALL/PARTIAL valid markers
            # =================================================================
            # 1. Take only markers with GT
            # valid_marker_mask = torch.from_numpy(current_mask & base_mask).bool().to(device)

            # 2. Take all
            valid_marker_mask = torch.from_numpy(base_mask).bool().to(device)

            # -------------------------------------------------------------------
            # METRIC 1: Centered Shape Matching & Dynamic Regions
            # -------------------------------------------------------------------
            hips_pos = model.j_rest[0].to(device)
            
            # 1. Get absolute world positions
            pred_world_soma = out['p_pred_tpose'] + hips_pos
            pred_world_lbs = model.p_bind.unsqueeze(0) 
            target_world = model.p_bind.unsqueeze(0) + residuals_gt_tensor
            
            # 2. Center everything
            pred_centered_soma = pred_world_soma - pred_world_soma.mean(dim=1, keepdim=True)
            pred_centered_lbs = pred_world_lbs - pred_world_lbs.mean(dim=1, keepdim=True)
            target_centered = target_world - target_world.mean(dim=1, keepdim=True)
            
            # 3. Calculate Errors
            errors_soma = torch.norm(pred_centered_soma[0, valid_marker_mask] - target_centered[0, valid_marker_mask], dim=-1) * 1000.0
            errors_lbs = torch.norm(pred_world_lbs[0, valid_marker_mask] - target_world[0, valid_marker_mask], dim=-1) * 1000.0
            
            # --- THE DYNAMIC REGION ---
            # Isolate markers where LBS is failing (Error > 15 mm)
            dynamic_mask = errors_lbs > 15.0 
            if dynamic_mask.sum() > 0:
                all_dynamic_pme.append(errors_soma[dynamic_mask].mean().item())     # SOMA's error in dynamic areas
                all_lbs_dynamic_pme.append(errors_lbs[dynamic_mask].mean().item())  # LBS's error in dynamic areas

            # --- THE DYNAMIC REGION ---
            mask_15 = errors_lbs > 15.0
            mask_20 = errors_lbs > 20.0
            mask_25 = errors_lbs > 25.0
            
            if mask_15.sum() > 0:
                all_dynamic_15_soma.append(errors_soma[mask_15].mean().item())
                all_dynamic_15_lbs.append(errors_lbs[mask_15].mean().item())
                
            if mask_20.sum() > 0:
                all_dynamic_20_soma.append(errors_soma[mask_20].mean().item())
                all_dynamic_20_lbs.append(errors_lbs[mask_20].mean().item())
                
            if mask_25.sum() > 0:
                all_dynamic_25_soma.append(errors_soma[mask_25].mean().item())
                all_dynamic_25_lbs.append(errors_lbs[mask_25].mean().item())
            
            # 4. Record Global Metrics SOMA
            all_mpme_mm.append(errors_soma.mean().item())     
            all_med_pme_mm.append(errors_soma.median().item()) 
            all_p90_pme_mm.append(torch.quantile(errors_soma, 0.90).item())

            # 5. Record Global Metrics LBS
            all_lbs_mpme_mm.append(errors_lbs.mean().item())     
            all_lbs_med_pme_mm.append(errors_lbs.median().item()) 
            all_lbs_p90_pme_mm.append(torch.quantile(errors_lbs, 0.90).item())

            # -------------------------------------------------------------------
            # METRIC 2 & 3: Volume Change (%) and Intersections
            # -------------------------------------------------------------------
            
            # --- A. Fat Layer (Evaluated in Canonical T-Pose) ---
            # SOMA (Learned T-Pose Blendshapes)
            curr_vol_fat = prism_calc.compute_prism_volume(out['m_final'], out['s_final'], s_faces)
            fat_change_pct = (curr_vol_fat[valid_fat_mask] - rest_vol_skin[valid_fat_mask]).abs() / rest_vol_skin[valid_fat_mask].abs()
            all_vol_fat_change.append(fat_change_pct.mean().item() * 100.0)
            all_intersection_fat.append((curr_vol_fat[valid_fat_mask] > 0).float().mean().item() * 100.0)

            # LBS BASELINE (Canonical T-Pose is the Bind Pose)
            curr_vol_fat_lbs = prism_calc.compute_prism_volume(m_bind.unsqueeze(0), s_bind.unsqueeze(0), s_faces)
            fat_change_pct_lbs = (curr_vol_fat_lbs[valid_fat_mask] - rest_vol_skin[valid_fat_mask]).abs() / rest_vol_skin[valid_fat_mask].abs()
            all_vol_fat_change_lbs.append(fat_change_pct_lbs.mean().item() * 100.0)
            all_intersection_fat_lbs.append((curr_vol_fat_lbs[valid_fat_mask] > 0).float().mean().item() * 100.0)

            # --- B. Deep Muscle Layer (Evaluated in Posed Space to capture LBS collapse) ---
            # SOMA (Corrected Posed Mesh)
            curr_vol_musc = prism_calc.compute_prism_volume(out['b_final_theta'], out['m_final_theta'], m_faces)
            musc_change_pct = (curr_vol_musc[valid_musc_mask] - rest_vol_musc[valid_musc_mask]).abs() / rest_vol_musc[valid_musc_mask].abs()
            all_vol_musc_change.append(musc_change_pct.mean().item() * 100.0)
            all_intersection_musc.append((curr_vol_musc[valid_musc_mask] > 0).float().mean().item() * 100.0)

            # LBS BASELINE (Uncorrected Posed Mesh)
            # curr_vol_musc_lbs = prism_calc.compute_prism_volume(out['b_final_theta'], out['m_bind_theta'], m_faces)
            # musc_change_pct_lbs = (curr_vol_musc_lbs[valid_musc_mask] - rest_vol_musc[valid_musc_mask]).abs() / rest_vol_musc[valid_musc_mask].abs()
            # all_vol_musc_change_lbs.append(musc_change_pct_lbs.mean().item() * 100.0)
            # all_intersection_musc_lbs.append((curr_vol_musc_lbs[valid_musc_mask] > 0).float().mean().item() * 100.0)

            # LBS BASELINE
            curr_vol_musc_lbs = prism_calc.compute_prism_volume(out['b_final_theta'], out['m_bind_theta'], m_faces)
            musc_change_pct_lbs = (curr_vol_musc_lbs[valid_musc_mask] - rest_vol_musc[valid_musc_mask]).abs() / rest_vol_musc[valid_musc_mask].abs()
            all_vol_musc_change_lbs.append(musc_change_pct_lbs.mean().item() * 100.0)
            all_vol_musc_change_lbs_p90.append(torch.quantile(musc_change_pct_lbs, 0.90).item() * 100.0) # NEW
            all_intersection_musc_lbs.append((curr_vol_musc_lbs[valid_musc_mask] > 0).float().mean().item() * 100.0)

    # -------------------------------------------------------------------
    # FINAL REPORT
    # -------------------------------------------------------------------
    print("\n" + "="*85)
    print(f"  QUANTITATIVE EVALUATION RESULTS ({len(validation_files)} unseen frames)")
    print("="*85)
    print(f"{'METRIC':<30} | {'SOMA (Ours)':<20} | {'LBS BASELINE':<20}")
    print("-" * 85)
    
    print("GLOBAL TRACKING (mm) ↓")
    print(f"  MPME (Mean)                  | {np.mean(all_mpme_mm):<20.2f} | {np.mean(all_lbs_mpme_mm):<20.2f}")
    print(f"  MedPME (Median)              | {np.mean(all_med_pme_mm):<20.2f} | {np.mean(all_lbs_med_pme_mm):<20.2f}")
    print(f"  P90 (90th Percentile)        | {np.mean(all_p90_pme_mm):<20.2f} | {np.mean(all_lbs_p90_pme_mm):<20.2f}")
    print("-" * 85)
    
    print("DYNAMIC REGION ERROR (mm) ↓")
    print(f"  Tau > 15 mm                  | {np.mean(all_dynamic_15_soma):<20.2f} | {np.mean(all_dynamic_15_lbs):<20.2f}")
    print(f"  Tau > 20 mm                  | {np.mean(all_dynamic_20_soma):<20.2f} | {np.mean(all_dynamic_20_lbs):<20.2f}")
    print(f"  Tau > 25 mm                  | {np.mean(all_dynamic_25_soma):<20.2f} | {np.mean(all_dynamic_25_lbs):<20.2f}")
    print("-" * 85)
    
    print("PHYSICAL PLAUSIBILITY: INTERSECTIONS (%) ↓")
    print(f"  Fat Layer (S -> M)           | {np.mean(all_intersection_fat):<20.3f} | {np.mean(all_intersection_fat_lbs):<20.3f}")
    print(f"  Deep Muscle (M -> B)         | {np.mean(all_intersection_musc):<20.3f} | {np.mean(all_intersection_musc_lbs):<20.3f}")
    print("-" * 85)
    
    print("PHYSICAL PLAUSIBILITY: VOLUME CHANGE (%) ↓")
    print(f"  Fat Layer (S -> M)           | {np.mean(all_vol_fat_change):<20.2f} | {np.mean(all_vol_fat_change_lbs):<20.2f}")
    print(f"  Deep Muscle (M -> B)         | {np.mean(all_vol_musc_change):<20.2f} | {np.mean(all_vol_musc_change_lbs):<20.2f}")
    print("="*85)
    print("\n>>> COPY-PASTE THIS ROW INTO YOUR LATEX TABLE <<<")
    latex_str = (
        f"\\textbf{{Ours}} & \\textbf{{{np.mean(all_mpme_mm):.2f}}} & \\textbf{{{np.mean(all_med_pme_mm):.2f}}} & \\textbf{{{np.mean(all_p90_pme_mm):.2f}}} & "
        f"\\textbf{{{np.mean(all_dynamic_15_soma):.2f}}} & \\textbf{{{np.mean(all_dynamic_20_soma):.2f}}} & \\textbf{{{np.mean(all_dynamic_25_soma):.2f}}} & "
        f"\\textbf{{{np.mean(all_intersection_fat):.3f}}} & \\textbf{{{np.mean(all_intersection_musc):.3f}}} & "
        f"\\textbf{{{np.mean(all_vol_fat_change):.2f}}} & \\textbf{{{np.mean(all_vol_musc_change):.2f}}} \\\\"
    )
    print(latex_str)
    print("="*85 + "\n")
if __name__ == "__main__":
    main()