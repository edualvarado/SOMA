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

        # # 3. Create a Numerically Safe Denominator
        # mean_vol = rest_vol_valid.abs().mean().clamp(min=1e-8)
        # safe_denom = rest_vol_valid.abs() + (0.1 * mean_vol)
        
        # # 4. Final Energy Calculation: Energy Density scaled for the optimizer
        # # We multiply by 1e6 to bring the microscopic volumes into balance with data loss
        # weighted_error = (diff.pow(2) / safe_denom) * 1e6
        
        # return weighted_error.mean()

# --- VIZ FUNCTIONS ---

def visualize_output(p_bind, residuals_gt, p_pred, skin_layer, musc_layer, posed_joints, parents, current_m_final=None, current_s_final=None, d_muscle=None, rest_vol_skin=None):
    """
    Visualizes the model output.
    - If target_muscle_id is Set: Shows only that muscle (Red prediction, Colored GT).
    - If target_muscle_id is None: Shows ALL muscles with unique palette colors.
    - current_m_final: (V, 3) numpy array of the DEFORMED mesh. If None, shows static mesh.
    """
    
    logger.info("--- LAUNCHING DEBUG VISUALIZATION ---")
    print("\nOpen your browser to http://localhost:8080")

    server = viser.ViserServer()
    server.scene.add_grid("grid", plane="xz")
    server.scene.set_up_direction("+y")

    BLACK = [0, 0, 0]
    RED = [255, 0, 0]
    GRAY = [50, 50, 50]
    GREEN = [0, 255, 0]
    CYAN = [0, 255, 255]
    BLUE = [0, 0, 255]
    YELLOW = [255, 255, 0]
    MAGENTA = [255, 0, 255]

    # --- 1. Select Data (Deformed vs Rest) ---
    if current_m_final is not None and current_s_final is not None:
        logger.info("[VIZ] Visualizing DEFORMED state.")
        m_verts = current_m_final
        s_verts = current_s_final
        # Topology comes from the bind layers
        m_faces = musc_layer.faces
        s_faces = skin_layer.faces
    else:
        logger.info("[VIZ] Visualizing REST/BIND state (Deformed data missing).")
        m_verts = musc_layer.vertices
        s_verts = skin_layer.vertices
        m_faces = musc_layer.faces
        s_faces = skin_layer.faces

    # Muscle Layer (Inner) - Red Wireframe
    server.scene.add_mesh_simple(
        name="/bind/muscle_layer",
        vertices=musc_layer.vertices,
        faces=musc_layer.faces,
        color=(200, 50, 50),
        wireframe=True,
        opacity=0.4
    )

    # Skin Layer (Outer) - Blue Wireframe
    server.scene.add_mesh_simple(
        name="/bind/skin_layer",
        vertices=skin_layer.vertices,
        faces=skin_layer.faces,
        color=(50, 50, 200),
        wireframe=True,
        opacity=0.4
    )

    # Skeleton
    bone_points = []
    for i, p_idx in enumerate(parents):
        if p_idx != -1:
            bone_points.append([posed_joints[p_idx], posed_joints[i]])
    server.scene.add_line_segments("/posed_skeleton/bones", np.array(bone_points), colors=(0, 0, 255))

    # Colors
    # gt_marker_colors = np.tile(BLACK, (len(p_bind), 1)).astype(np.uint8)
    # pred_marker_colors = np.tile(BLACK, (len(p_bind), 1)).astype(np.uint8)

    # Ground Truth Markers
    p_gt = p_bind + residuals_gt
    server.scene.add_point_cloud(
        "/p_bind + residuals_gt", 
        points=p_gt, 
        colors=GREEN, 
        point_size=0.003
    )

    # Prediction
    server.scene.add_point_cloud(
        "/p_bind + delta_pred_tpose", 
        points=p_pred, 
        colors=BLACK, 
        point_size=0.003
    )

    if current_m_final is not None:
        # Display the muscle mesh in the rest pose (Model Step 2)
        server.scene.add_mesh_simple(
            name="/current_m_final",
            vertices=current_m_final,
            faces=musc_layer.faces,
            color=RED,  # Red
            wireframe = False
        )

        server.scene.add_mesh_simple(
            name="/current_s_final",
            vertices=current_s_final,
            faces=skin_layer.faces,
            color=GRAY,
            wireframe = False
        )
    
    # =========================================================
    # --- NEW: PRISM VISUALIZATION ---
    # =========================================================
    if rest_vol_skin is not None:
        logger.info("[VIZ] Rendering Volume Prisms...")
        
        # This matches the threshold inside PrismVolumeLoss
        valid_prism_threshold = 1e-7
        
        # 1. Determine Valid vs Invalid Faces based on Rest Volume
        rest_vol_1d = rest_vol_skin.squeeze().cpu().numpy()
        valid_face_mask = np.abs(rest_vol_1d) > valid_prism_threshold
        
        faces_np = musc_layer.faces
        valid_faces = faces_np[valid_face_mask]
        invalid_faces = faces_np[~valid_face_mask]
        
        # 2. Determine Valid vs Invalid Vertices for rendering connections
        valid_vert_ids = np.unique(valid_faces)
        all_vert_ids = np.arange(len(current_m_final))
        invalid_vert_ids = np.setdiff1d(all_vert_ids, valid_vert_ids)
        
        # 3. Draw Valid Connections (Green)
        if len(valid_vert_ids) > 0:
            m_valid = current_m_final[valid_vert_ids]
            s_valid = current_s_final[valid_vert_ids]
            lines_valid = np.stack([m_valid, s_valid], axis=1) # (N, 2, 3)
            server.scene.add_line_segments(
                name="/prisms/valid_tissue_connections",
                points=lines_valid,
                colors=(0, 255, 0), # Green = Healthy Volume
                line_width=1.0, 
            )
            
        # 4. Draw Invalid Connections (Red) - Ignored by volume loss
        if len(invalid_vert_ids) > 0:
            m_invalid = current_m_final[invalid_vert_ids]
            s_invalid = current_s_final[invalid_vert_ids]
            lines_invalid = np.stack([m_invalid, s_invalid], axis=1)
            server.scene.add_line_segments(
                name="/prisms/degenerate_connections",
                points=lines_invalid,
                colors=(255, 0, 0), # Red = Degenerate/Thin Volume
                line_width=2.0, 
            )
        
        # 5. Highlight specific prisms for close inspection
        def draw_single_prism(f_idx, prefix, color_m, color_s, color_line):
            face_indices = faces_np[f_idx]
            m_v = current_m_final[face_indices]
            s_v = current_s_final[face_indices]
            single_face = np.array([[0, 1, 2]], dtype=np.int32)
            
            # Muscle Base Triangle
            server.scene.add_mesh_simple(
                name=f"/prisms/inspect_{prefix}/muscle_base",
                vertices=m_v, faces=single_face, color=color_m, opacity=1.0
            )
            # Skin Top Triangle
            server.scene.add_mesh_simple(
                name=f"/prisms/inspect_{prefix}/skin_top",
                vertices=s_v, faces=single_face, color=color_s, opacity=1.0
            )
            # Vertical Connections
            lines = np.stack([m_v, s_v], axis=1)
            server.scene.add_line_segments(
                name=f"/prisms/inspect_{prefix}/connections",
                points=lines, colors=color_line, line_width=4.0
            )
            
        # Draw one random healthy prism and one random degenerate prism
        if valid_faces.shape[0] > 0:
            valid_f_idx = np.random.choice(np.nonzero(valid_face_mask)[0])
            draw_single_prism(valid_f_idx, "valid", (0,200,0), (50,255,50), (255,255,0))
            
        if invalid_faces.shape[0] > 0:
            invalid_f_idx = np.random.choice(np.nonzero(~valid_face_mask)[0])
            draw_single_prism(invalid_f_idx, "degenerate", (200,0,0), (255,50,50), (255,100,100))

    while True:
        time.sleep(0.1)

# --- MODEL FUNCTIONS ---

def lbs_working_batch_rotmat(vertices, rot_mats, weights, j_rest, parents, root_position):
    """
    LBS given rotation matrices.
    vertices: (B,V,3), rot_mats: (B,J,3,3), weights: (V,J), j_rest: (J,3), parents: (J,), root_position: (B,3)
    """
    batch_size = vertices.shape[0]
    num_joints = j_rest.shape[0]
    device = vertices.device

    # Align dtypes for numerical equivalence
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
    """
    Calculates the 3D positions of markers based on barycentric coordinates.
    
    Args:
        deformed_vertices (torch.Tensor): Deformed mesh vertices. Shape: (batch_size, num_vertices, 3)
        bary_verts (torch.Tensor): Vertex indices for each marker. Shape: (num_markers, 3)
        bary_weights (torch.Tensor): Barycentric weights for each marker. Shape: (num_markers, 3)
        
    Returns:
        torch.Tensor: Predicted 3D marker positions. Shape: (batch_size, num_markers, 3)
    """

    # Gather the vertices for each marker's triangle
    # bary_verts has shape (num_markers, 3). deformed_vertices has shape (batch_size, num_markers, 3).
    # We want to select vertices along the Ns dimension.
    v0 = deformed_vertices[:, bary_verts[:, 0], :]  # Shape: (batch_size, num_markers, 3)
    v1 = deformed_vertices[:, bary_verts[:, 1], :]  # Shape: (batch_size, num_markers, 3)
    v2 = deformed_vertices[:, bary_verts[:, 2], :]  # Shape: (batch_size, num_markers, 3)

    # Apply the barycentric weights to interpolate the marker positions
    # bary_weights has shape (num_markers, 3). We broadcast it to (batch_size, num_markers, 3) for batch processing.
    interpolated_positions = (
        bary_weights[:, 0].unsqueeze(0).unsqueeze(-1) * v0 +
        bary_weights[:, 1].unsqueeze(0).unsqueeze(-1) * v1 +
        bary_weights[:, 2].unsqueeze(0).unsqueeze(-1) * v2
    )

    return interpolated_positions  # Shape: (batch_size, num_markers, 3)

# --- TRANSFORMATION FUNCTIONS ---

def _rot_x(points, deg=0.0):
    R = Rotation.from_euler('x', deg, degrees=True).as_matrix()
    return (points @ R.T)

def _rot_y(points, deg=0.0):
    R = Rotation.from_euler('y', deg, degrees=True).as_matrix()
    return (points @ R.T)

def _rot_z(points, deg=0.0):
    R = Rotation.from_euler('z', deg, degrees=True).as_matrix()
    return (points @ R.T)

def rotate_points_x(points, angle_deg=-90):
    """Rotates points around X-axis."""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    # Rotation Matrix for X-axis
    R = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ], dtype=points.dtype)
    return points @ R.T

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
    
def sixd_to_rotmat(sixd_reps):
    """
    Converts 6D rotation representation to 3x3 rotation matrices.
    Input: (B, J, 6)
    Output: (B, J, 3, 3)
    """
    x_raw = sixd_reps[..., 0:3]
    y_raw = sixd_reps[..., 3:6]

    x = F.normalize(x_raw, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)

    # Stack columns to form matrix
    matrix = torch.stack((x, y, z), dim=-1)
    return matrix

# --- LBS FUNCTIONS ---

def convert_weights_to_npy(json_path, num_rows, output_path, bvh_joint_names, is_skin=True, canonical_ids=None):
    """
    Generic function to convert skin or marker weights to a dense .npy matrix,
    using bone names to ensure correct joint ordering.
    """
    logger.debug(f"[LBS] Converting weights from {os.path.basename(json_path)} to .npy...")
    with open(json_path, 'r') as f:
        weights_data = json.load(f)

    num_joints = len(bvh_joint_names)
    # Create a lookup map from name to the correct index for this BVH
    joint_name_to_index = {name: i for i, name in enumerate(bvh_joint_names)}
    
    weights_matrix = np.zeros((num_rows, num_joints), dtype=np.float32)

    data_source = weights_data if is_skin else weights_data.items()
    
    if is_skin:
        # Assuming the skin weights JSON is a list of dicts, ordered by vertex index
        for v_idx, vertex_info in enumerate(weights_data):
            if v_idx >= num_rows: continue
            bone_names = vertex_info.get("bone_names", [])
            weights = vertex_info.get("weights", [])
            for bone_name, weight in zip(bone_names, weights):
                if bone_name in joint_name_to_index:
                    correct_joint_idx = joint_name_to_index[bone_name]
                    weights_matrix[v_idx, correct_joint_idx] = weight
    else: # Markers
        marker_id_to_index = {marker_id: i for i, marker_id in enumerate(canonical_ids)}
        for marker_id, data in data_source:
            if marker_id in marker_id_to_index:
                v_idx = marker_id_to_index[marker_id]
                bone_names = data.get("bone_names", [])
                weights = data.get("weights", [])
                for bone_name, weight in zip(bone_names, weights):
                    if bone_name in joint_name_to_index:
                        correct_joint_idx = joint_name_to_index[bone_name]
                        weights_matrix[v_idx, correct_joint_idx] = weight

    np.save(output_path, weights_matrix)
    logger.debug(f"[LBS] Successfully converted and saved weights to {output_path}")
    return weights_matrix

# --- SKELETON FUNCTIONS ---

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

# --- MESH FUNCTIONS ---

def compute_vertex_stability_mask(m_bind, s_bind, faces, threshold=1e-7):
    """
    Creates a (V, 1) mask based on the exact prism volume.
    1.0 = Vertex belongs to a thick area (Volume > threshold), allowed to deform.
    0.0 = Vertex belongs to a thin/degenerate area (Volume <= threshold), PINNED.
    """
    device = m_bind.device
    
    # 1. Gather Prism Vertices
    f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]
    x1 = m_bind[f0]; x2 = m_bind[f1]; x3 = m_bind[f2]
    x4 = s_bind[f0]; x5 = s_bind[f1]; x6 = s_bind[f2]
    
    # 2. Compute exact volume using 2-point Gauss Quadrature (syncs perfectly with loss)
    inv_sqrt3 = 0.5773502691896257
    t1 = 0.5 * (1.0 - inv_sqrt3)
    t2 = 0.5 * (1.0 + inv_sqrt3)
    alpha = beta = gamma = 1.0 / 3.0

    def compute_detJ(xi_scalar):
        dx_da = xi_scalar * (x1 - x3) + (1.0 - xi_scalar) * (x4 - x6)
        dx_db = xi_scalar * (x2 - x3) + (1.0 - xi_scalar) * (x5 - x6)
        dx_dxi = alpha * (x1 - x4) + beta * (x2 - x5) + gamma * (x3 - x6)
        return torch.sum(dx_da * torch.cross(dx_db, dx_dxi, dim=-1), dim=-1)

    # Calculate exact rest volume per face
    face_volumes = 0.25 * (compute_detJ(t1) + compute_detJ(t2))
    
    # 3. Identify Valid Prisms using your exact threshold
    valid_prism_mask = face_volumes.abs() > threshold
    
    # 4. Map Valid Prisms to Vertices
    num_verts = m_bind.shape[0]
    vertex_mask = torch.zeros(num_verts, 1, device=device)
    
    # Get all unique vertex indices that belong to thick faces
    valid_faces = faces[valid_prism_mask] 
    valid_indices = torch.unique(valid_faces)
    
    # Set those vertices to 1.0 (allow deformation)
    vertex_mask[valid_indices] = 1.0
    
    return vertex_mask

def load_obj_simple(file_path):
    """
    A simple, robust OBJ loader that only reads vertex positions and faces.
    Guarantees vertex count matches the 'v' lines.
    """
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
        L_sub = laplacian_calculation(mesh, equal_weight=True)
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
    laplacian_sp = laplacian_calculation(trimesh_obj, equal_weight=True)

    # --- FACE ADJACENCY (For Normal Consistency) ---
    face_adjacency_np = trimesh_obj.face_adjacency
    face_adjacency = torch.from_numpy(face_adjacency_np).long()
    
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
        'face_adjacency': face_adjacency,
        'laplacian_sp': laplacian_sp,
        'laplacian': laplacian,
        'laplacian_degree': laplacian_degree
    }

# --- CONFIGURATION ---
# Centralized config to keep parameters organized
CONFIG = {
    "subject": "S1",
    # Paths (We will build absolute paths dynamically below)
    "base_path_suffix": r"static00", 
    
    # Training Hyperparameters
    # "learning_rate": 1e-3,
    "learning_rate": 1e-4,
    "epochs": 20,
    "batch_size": 128,
    # "weight_decay": 1e-4,
    "weight_decay": 0.0,
    "train_split": 0.9,

    "architecture": "mlp", # Options: "linear", "mlp", "unet"
    
    # System
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4, # Set to 0 if using RAM loading (faster), else 4
    "preload_to_ram": False, # CRITICAL OPTIMIZATION: Load all .npy to RAM
    "testing_dataset": False,
    
    # Modes
    "overfit_single": False,   # Train on 1 frame only
    "overfit_multiple": False, # Train on first 300 frames only
    "viz_enabled": False,        # Enable Viser visualization

    "checkpoint_path": None # Path to the file just saved}
}
 
# --- LOSS WEIGHT PRESETS ---
# Add new presets here. Change ACTIVE_PRESET to switch between them.
# Keys: w_data, w_smooth_musc/skin, w_biharmonic_musc/skin, w_spring_musc/skin,
#       w_tangent_musc/skin, w_vol_musc/skin

LAMBDAS_PRESETS = {
    # All losses enabled at high weights
    "full": {
        "w_data": 50,
        "w_smooth_musc": 100,       "w_smooth_skin": 200,
        "w_biharmonic_musc": 100,   "w_biharmonic_skin": 200,
        "w_spring_musc": 1000,      "w_spring_skin": 1000,
        "w_tangent_musc": 1,        "w_tangent_skin": 0.1,
        "w_vol_musc": 1000,         "w_vol_skin": 1000,
    },
    # Ablation: remove vector losses (E_smooth, E_tan)
    "no_smooth_tan": {
        "w_data": 1,
        "w_smooth_musc": 0,         "w_smooth_skin": 0,
        "w_biharmonic_musc": 1,     "w_biharmonic_skin": 2.5,
        "w_spring_musc": 1,         "w_spring_skin": 1,
        "w_tangent_musc": 0,        "w_tangent_skin": 0,
        "w_vol_musc": 10,           "w_vol_skin": 10,
    },
    # Ablation: remove all physics losses (E_bi, E_spring, E_vol)
    "no_physics": {
        "w_data": 1,
        "w_smooth_musc": 0.01,      "w_smooth_skin": 0.05,
        "w_biharmonic_musc": 0,     "w_biharmonic_skin": 0,
        "w_spring_musc": 0,         "w_spring_skin": 0,
        "w_tangent_musc": 1,        "w_tangent_skin": 0.1,
        "w_vol_musc": 0,            "w_vol_skin": 0,
    },
    # Ablation: remove volume loss only
    "no_vol": {
        "w_data": 1,
        "w_smooth_musc": 0.01,      "w_smooth_skin": 0.05,
        "w_biharmonic_musc": 1,     "w_biharmonic_skin": 2.5,
        "w_spring_musc": 1,         "w_spring_skin": 1,
        "w_tangent_musc": 1,        "w_tangent_skin": 0.1,
        "w_vol_musc": 0,            "w_vol_skin": 0,
    },
    # Ablation: remove biharmonic and stretch losses (E_bi, E_spring)
    "no_bi_stretch": {
        "w_data": 1,
        "w_smooth_musc": 0.01,      "w_smooth_skin": 0.05,
        "w_biharmonic_musc": 0,     "w_biharmonic_skin": 0,
        "w_spring_musc": 0,         "w_spring_skin": 0,
        "w_tangent_musc": 1,        "w_tangent_skin": 0.1,
        "w_vol_musc": 10,           "w_vol_skin": 10,
    },
}

# ------------------------------------------------------------------
ACTIVE_PRESET = "no_bi_stretch"  # <-- Change this to switch presets
LAMBDAS = LAMBDAS_PRESETS[ACTIVE_PRESET]
# ------------------------------------------------------------------

# --- PATH SETUP ---
# Adjust this base path to match new directory structure
BASE_DIR = rf"/CT/SOMA/{CONFIG['base_path_suffix']}/{CONFIG['subject']}"

PATHS = {
    "raw": os.path.join(BASE_DIR, "raw"),
    "processed": os.path.join(BASE_DIR, "preprocessed_vFinal_clean"),
    "layers_tpose": os.path.join(BASE_DIR, "layers", "tpose"),
    "layers_apose": os.path.join(BASE_DIR, "layers", "apose"),
    "canonical": os.path.join(BASE_DIR, "canonical_model"),
    "checkpoints": os.path.join(os.getcwd(), "checkpoints"),
    "logs": os.path.join(os.getcwd(), "runs")
}

# --- DATASET CLASS ---
class ProcessedMotionDataset(Dataset):
    def __init__(self, processed_dir, preload=True):
        self.processed_dir = processed_dir
        self.preload = preload
        
        # 1. Find all files
        self.pose_rot_files = sorted(glob.glob(os.path.join(processed_dir, 'pose_rotations', '*.npy')))
        self.residuals_dir = os.path.join(processed_dir, 'residuals')
        self.masks_dir = os.path.join(processed_dir, 'masks')
        self.canonical_lbs_dir = os.path.join(processed_dir, 'canonical_lbs')

        if len(self.pose_rot_files) == 0:
            raise ValueError(f"No data found in {processed_dir}")

        # 2. Pre-load into RAM (Optimization)
        self.cache = []
        if self.preload:
            logger.info(f"[DATA] Pre-loading {len(self.pose_rot_files)} frames into RAM...")
            for rot_path in tqdm(self.pose_rot_files, desc="Loading Data"):
                self.cache.append(self._load_frame(rot_path))
            logger.success("[DATA] Pre-loading complete.")
        else:
            logger.info("[DATA] RAM Pre-loading DISABLED. Reading from disk (slower training, faster startup).")

    def _load_frame(self, rot_path):
        """Helper to load a single frame's tuple from disk"""
        base_filename = os.path.basename(rot_path)
        res_path = os.path.join(self.residuals_dir, base_filename)
        mask_path = os.path.join(self.masks_dir, base_filename)
        lbs_path = os.path.join(self.canonical_lbs_dir, base_filename)
        
        return (
            torch.from_numpy(np.load(rot_path)).float(),
            torch.from_numpy(np.load(res_path)).float(),
            torch.from_numpy(np.load(mask_path)).float(),
            torch.from_numpy(np.load(lbs_path)).float()
        )

    def __len__(self):
        return len(self.pose_rot_files)

    def __getitem__(self, idx):
        if self.preload:
            return self.cache[idx]
        else:
            return self._load_frame(self.pose_rot_files[idx])

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

class SOMALossOriginal(nn.Module):
    def __init__(self, skin_geo, muscle_geo, bone_geo, lambdas):
        """
        Implements the loss terms from the SOMA paper.
        Args:
            skin_geo: Dictionary containing skin 'verts', 'edges', etc.
            muscle_geo: Dictionary containing muscle 'verts', 'edges', etc.
            lambdas: Dictionary of weights (e.g. {'data': 1.0, 'smooth': 0.1})
        """
        super().__init__()
        self.skin = skin_geo
        self.muscle = muscle_geo
        self.bone = bone_geo
        self.lambdas = lambdas
        
        # Pre-compute Rest State Edge Lengths (Target for regularization)
        # We detach() to ensure we don't try to train the rest pose itself
        self.rest_skin_edges = compute_edge_lengths(
            self.skin['verts'].unsqueeze(0), 
            self.skin['edges']
        ).detach()
        
        self.rest_muscle_edges = compute_edge_lengths(
            self.muscle['verts'].unsqueeze(0), 
            self.muscle['edges']
        ).detach()

        # Pre-compute Laplancian and Mass for Smoothness
        self.l_skin = self.skin['laplacian']
        self.l_muscle = self.muscle['laplacian']
        
        # We store 1.0 / Area (Inverse Mass) because we multiply by it in the loss
        # Energy = ||Force||^2 / Area. Clamped to avoid division by zero.
        self.s_inv_mass = 1.0 / (self.skin['vertex_mass'] + 1e-8)
        self.m_inv_mass = 1.0 / (self.muscle['vertex_mass'] + 1e-8)
        
        # Pre-compute Rest Edge Lengths 
        # edges shape: (E, 2)
        def get_rest_lengths(verts, edges):
            v0 = verts[edges[:, 0]]
            v1 = verts[edges[:, 1]]
            return torch.norm(v0 - v1, dim=-1).detach() # (E,)

        self.rest_len_m = get_rest_lengths(self.muscle['verts'], self.muscle['edges'])
        self.rest_len_s = get_rest_lengths(self.skin['verts'], self.skin['edges'])

        # Pre-compute Rest Pose Dihedral Angles (Cosine)
        def get_rest_cosine(verts, faces, face_adj):
            # verts: (V, 3), faces: (F, 3), face_adj: (E_int, 2)
            v0 = verts[faces[:, 0]]
            v1 = verts[faces[:, 1]]
            v2 = verts[faces[:, 2]]
            
            # Cross product gets the face normals
            n = torch.cross(v1 - v0, v2 - v0, dim=1)
            n = torch.nn.functional.normalize(n, p=2, dim=1) # (F, 3)
            
            # Get normals of adjacent faces
            n1 = n[face_adj[:, 0]] # (E_int, 3)
            n2 = n[face_adj[:, 1]] # (E_int, 3)
            
            # Dot product is the cosine of the angle between them
            return (n1 * n2).sum(dim=1).detach() # (E_int,)

        self.rest_cos_m = get_rest_cosine(self.muscle['verts'], self.muscle['faces'], self.muscle['face_adjacency'])
        self.rest_cos_s = get_rest_cosine(self.skin['verts'], self.skin['faces'], self.skin['face_adjacency'])

        self.prism_calculator = PrismVolumeLoss()

        with torch.no_grad():
            # 1. Muscle-to-Skin (Layer S)
            self.rest_vol_skin = self.prism_calculator.compute_prism_volume(
                self.muscle['verts'].unsqueeze(0), self.skin['verts'].unsqueeze(0), self.skin['faces']
            ).detach()
            
            # 2. Bone-to-Muscle (Layer M)
            # Assuming self.b_bind exists and shares the topology `self.bone_faces`
            self.rest_vol_musc = self.prism_calculator.compute_prism_volume(
                self.bone['verts'].unsqueeze(0), self.muscle['verts'].unsqueeze(0), self.muscle['faces']
            ).detach()

    def forward(self, 
                pred_markers, gt_markers, mask, p_bind,     # Data Terms
                s_final_theta, m_final_theta, b_final_theta,   # Geometry Terms
                disp_skin, disp_musc):              # Displacement Terms (for smooth)

        losses = {}

        batch_size = pred_markers.shape[0]

        assert gt_markers.shape == pred_markers.shape, "Shapes of residuals_gt and p_pred_tpose do not match!"
        assert mask.shape == pred_markers.shape[:2], "Shapes of masks_filtered and p_predicted do not match!"
        
        # Build residual
        p_gt = p_bind + gt_markers

        # ==========================================================
        # A. DATA LOSS (Masked L2 Distance)
        # ==========================================================
        error = (pred_markers - p_gt) * mask.unsqueeze(-1)
        data_loss = error.pow(2).mean()
        losses['data'] = data_loss

        # ---

        # error = (pred_markers - p_gt) * mask.unsqueeze(-1)  
        # # NEW: Divide ONLY by the number of active coordinates
        # num_active_coords = mask.sum() * 3.0 
        # data_loss = error.pow(2).sum() / (num_active_coords + 1e-8)
        # losses['data'] = data_loss

        # ==========================================================
        # B. SPATIAL SMOOTHNESS (MUSCLE)
        # ==========================================================
        num_verts_m = disp_musc.shape[1]

        # 1. Reshape for Matrix Multiplication
        # We need (Vertices, Batch * 3) to multiply with the (Vertices, Vertices) sparse Laplacian
        d_reshaped_m = disp_musc.permute(1, 0, 2).reshape(num_verts_m, -1)

        # 2. Compute Laplacian Coordinates (L @ d)
        # This calculates (d_i - average_neighbors) for every vertex
        lap_1_m = self.l_muscle @ d_reshaped_m

        # 3. Reshape back to (Vertices, Batch, 3) to apply per-vertex area weights
        lap_diff_m = lap_1_m.view(num_verts_m, batch_size, 3)

        # 4. Energy = Sum( ||Force||^2 / Area )
        # Broadcasting: (V, B) * (V, 1)
        weighted_diff_m = lap_diff_m.pow(2).sum(dim=-1) * self.m_inv_mass.unsqueeze(1) 
        
        losses['smooth_musc'] = weighted_diff_m.mean()

        # 2. [NEW] Second-Order Laplacian (Biharmonic / Bending Resistance)
        # We multiply by inv_mass to complete the discrete operator, then apply L again
        lap_2_m = self.l_muscle @ lap_1_m
        
        lap_2_m_view = lap_2_m.view(num_verts_m, batch_size, 3)
        weighted_diff_2_m = lap_2_m_view.pow(2).sum(dim=-1) * self.m_inv_mass.unsqueeze(1)
        losses['biharmonic_musc'] = weighted_diff_2_m.mean()

        # ==========================================================
        # C. SPATIAL SMOOTHNESS (SKIN)
        # ==========================================================
        # CRITICAL: The skin is physically attached to the muscle. If the muscle moves 
        # smoothly, the skin follows. Therefore, we only want to penalize the "roughness" 
        # of the Skin Residual (the extra sliding/wrinkling), not the total skin displacement.
        d_resid = disp_skin - disp_musc
        num_verts_s = d_resid.shape[1]

        # 1. Reshape
        d_reshaped_s = d_resid.permute(1, 0, 2).reshape(num_verts_s, -1)

        # 2. Apply Laplacian
        lap_1_s = self.l_skin @ d_reshaped_s

        # 3. Reshape back
        lap_diff_s = lap_1_s.view(num_verts_s, batch_size, 3)

        # 4. Weight by Area
        weighted_diff_s = lap_diff_s.pow(2).sum(dim=-1) * self.s_inv_mass.unsqueeze(1)
        
        losses['smooth_skin'] = weighted_diff_s.mean()

        # 2. [NEW] Second-Order Laplacian (Biharmonic)
        lap_2_s = self.l_skin @ lap_1_s
        
        lap_2_s_view = lap_2_s.view(num_verts_s, batch_size, 3)
        weighted_diff_2_s = lap_2_s_view.pow(2).sum(dim=-1) * self.s_inv_mass.unsqueeze(1)
        losses['biharmonic_skin'] = weighted_diff_2_s.mean()

        # ==========================================================
        # D. STRETCH LOSS (EDGE LENGTH PRESERVATION)
        # ==========================================================
        def compute_spring_loss(verts, edges, rest_lengths, edge_weights):
            # verts: (B, V, 3), edges: (E, 2)
            v0 = verts[:, edges[:, 0], :]
            v1 = verts[:, edges[:, 1], :]
            curr_lengths = torch.norm(v0 - v1, dim=-1) # (B, E)
            
            # 1. Calculate raw squared error per edge
            raw_error = (curr_lengths - rest_lengths.unsqueeze(0)).pow(2) # (B, E)
            
            # 2. Multiply by the physical area weight of that edge
            # edge_weights shape: (E,) -> unsqueeze to (1, E) for broadcasting
            weighted_error = raw_error * edge_weights.unsqueeze(0)
            
            # 3. Sum the errors for the whole mesh, then mean across the batch
            return weighted_error.sum(dim=-1).mean()

        # Get deformed verts
        curr_m = self.muscle['verts'].unsqueeze(0) + disp_musc
        curr_s = self.skin['verts'].unsqueeze(0) + disp_skin
        curr_b = self.bone['verts']

        losses['stretch_musc'] = compute_spring_loss(curr_m, self.muscle['edges'], self.rest_len_m, self.muscle['edge_weights'])
        losses['stretch_skin'] = compute_spring_loss(curr_s, self.skin['edges'], self.rest_len_s, self.skin['edge_weights'])

        # ==========================================================
        # E. TANGENTIAL LOSS (PREVENT SLIDING)
        # ==========================================================
        def compute_tangential_loss(disp, normals, inv_mass):
            # disp: (B, V, 3) - The displacement vector
            # normals: (V, 3) - The rest pose normal vector
            
            # 1. Project displacement onto the normal (Dot Product)
            # Result shape: (B, V, 1)
            dot_product = (disp * normals.unsqueeze(0)).sum(dim=-1, keepdim=True) 
            
            # 2. Get the normal and tangential vector components
            d_normal = dot_product * normals.unsqueeze(0) # (B, V, 3)
            d_tangent = disp - d_normal                   # (B, V, 3)
            
            # 3. Penalize the squared magnitude of the tangential component
            # We weight by inverse mass (area) for discretization invariance
            return (d_tangent.pow(2).sum(dim=-1) * inv_mass.unsqueeze(0)).mean()

        # Extract rest-pose normals from your geometry dictionaries
        m_normals = self.muscle['normals']
        s_normals = self.skin['normals']

        # Calculate loss
        losses['tangent_musc'] = compute_tangential_loss(disp_musc, m_normals, self.m_inv_mass)
        
        # For skin, we penalize the sliding of the residual displacement
        d_resid = disp_skin - disp_musc
        losses['tangent_skin'] = compute_tangential_loss(d_resid, s_normals, self.s_inv_mass)

        # ==========================================================
        # F. VOLUME PRESERVATION LOSS
        # ==========================================================

        # Expand bone to match batch size (Bone is locally rigid in this canonical step)
        curr_b = self.bone['verts'].unsqueeze(0).expand(batch_size, -1, -1)
        
        # 1. Skin-to-Muscle Volume (Evaluated in Canonical Space)
        # We use curr_m and curr_s (Canonical offset vertices calculated in stretch block)
        loss_vol_skin = self.prism_calculator(curr_m, curr_s, self.rest_vol_skin, self.skin['faces'])

        # 2. Muscle-to-Bone Volume (Evaluated in POSED Space!)
        # We use b_final_theta and m_final_theta (The crushed LBS vertices)
        loss_vol_musc = self.prism_calculator(b_final_theta, m_final_theta, self.rest_vol_musc, self.muscle['faces'])

        losses['vol_skin'] = loss_vol_skin
        losses['vol_musc'] = loss_vol_musc

        # ==========================================================

        total_loss = (
            self.lambdas.get('w_data', 0.0) * losses['data'] +
            self.lambdas.get('w_smooth_musc', 0.0) * losses['smooth_musc'] +
            self.lambdas.get('w_smooth_skin', 0.0) * losses['smooth_skin'] +
            self.lambdas.get('w_biharmonic_musc', 0.0) * losses['biharmonic_musc'] +
            self.lambdas.get('w_biharmonic_skin', 0.0) * losses['biharmonic_skin'] +
            self.lambdas.get('w_spring_musc', 0.0) * losses['stretch_musc'] +
            self.lambdas.get('w_spring_skin', 0.0) * losses['stretch_skin'] +
            self.lambdas.get('w_tangent_musc', 0.0) * losses['tangent_musc'] +
            self.lambdas.get('w_tangent_skin', 0.0) * losses['tangent_skin'] +
            self.lambdas.get('w_vol_skin', 0.0) * losses['vol_skin'] +
            self.lambdas.get('w_vol_musc', 0.0) * losses['vol_musc']
        )
        
        return total_loss, losses

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
                 stability_mask=None,
                 architecture = "mlp"):
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
    

        if architecture == "linear":
            logger.info("[MODEL] Using LINEAR BLENDSHAPES architecture")
            self.deformation_model = MuscleBlendshapeModel(
                num_vertices=self.m_bind.shape[0], num_joints=self.j_rest.shape[0]
            )
        elif architecture == "mlp":
            logger.info("[MODEL] Using MLP architecture")
            self.deformation_model = MuscleMLPModel(
                num_vertices=self.m_bind.shape[0], num_joints=self.j_rest.shape[0]
            )
        elif architecture == "unet":
            logger.info("[MODEL] Using U-NET architecture")
            self.deformation_model = MuscleUNetModel(
                num_vertices=self.m_bind.shape[0], num_joints=self.j_rest.shape[0]
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

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
        rot_mats = rot_mats.transpose(-2, -1)

        batch_size = rot_mats.shape[0]

        if root_positions is None:
            root_positions = torch.zeros(rot_mats.shape[0], 3, device=rot_mats.device, dtype=self.s_bind.dtype)

        # ===========

        # 1. The muscle blendshapes are driven by the relative rotations
        d_muscle_raw, d_skin_raw = self.deformation_model(rot_mats, rot_identity_input)  # (B,V,3)

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


# Main Training Script Structure
def train_full_model():
    """
    Initializes model, optimizer, AND loads Meshes + Canonical JSON Data.
    """
    device = torch.device(CONFIG["device"])
    logger.info(f"[TRAIN] Running on device: {device}")
    
    # Print full information about the device, CUDA version, and available memory
    if torch.cuda.is_available():
        logger.debug(f"[TRAIN] Using GPU: {torch.cuda.get_device_name(0)}")
        logger.debug(f"[TRAIN] CUDA Version: {torch.version.cuda}")
        logger.debug(f"[TRAIN] Available Memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
    else:
        logger.debug(f"[TRAIN] Using CPU")

    # ========================================================================================================================================================
    # 1. Configure paths
    logger.info("+++++++++++++++++++++++++++++++")
    logger.info("[INFO] 1. CONFIGURING PATHS...")
    logger.info("+++++++++++++++++++++++++++++++")
    # ========================================================================================================================================================

    try:
        # ----------------
        # 0. Dataset

        subject = CONFIG['subject']
        base_dir = BASE_DIR

        PROCESSED_DATA_DIRECTORY = PATHS["processed"]
        RAW_DATA_DIRECTORY = PATHS["raw"]
        LAYERS_DATA_DIRECTORY_TPOSE = PATHS["layers_tpose"]
        LAYERS_DATA_DIRECTORY_APOSE = PATHS["layers_apose"]
        CANONICAL_DATA_DIRECTORY = PATHS["canonical"]

        # General Hyperparameters
        LEARNING_RATE = CONFIG["learning_rate"]
        EPOCHS = CONFIG["epochs"]
        BATCH_SIZE = CONFIG["batch_size"]
        WEIGHT_DECAY = CONFIG["weight_decay"]
        TRAIN_SIZE = CONFIG["train_split"]
        
        # ----------------

        # ----------------
        # A. BVH file

        logger.info("[BVH] FINDING SAMPLE BVH...")

        # This is used as a sample data to get the skeleton structure
        sample_bvh_path = os.path.join(RAW_DATA_DIRECTORY, f'shot_001_captury/{subject}_shot_001.bvh')

        if not os.path.exists(sample_bvh_path):
            raise FileNotFoundError(f"Sample BVH file not found at {sample_bvh_path}")

        logger.success("[BVH] Sample BVH file found!")
        logger.info("============================================")

        # ----------------

        # ----------------
        # B. Static Meshes
        
        logger.info("[MESH] FINDING MESHES...")

        # Load all static assets
        muscle_mesh_path = os.path.join(LAYERS_DATA_DIRECTORY_TPOSE, f"musc_layer-{subject}-TPose.obj")
        skin_mesh_path = os.path.join(LAYERS_DATA_DIRECTORY_TPOSE, f"skin_layer-{subject}-TPose.obj")
        bone_mesh_path = os.path.join(LAYERS_DATA_DIRECTORY_TPOSE, f"skel_layer-{subject}-TPose.obj")

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
        muscle_obj_dir = os.path.join(LAYERS_DATA_DIRECTORY_TPOSE, "muscle_meshes_tpose")

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

        canonical_markers_path = os.path.join(CANONICAL_DATA_DIRECTORY, f"{subject}_canonical_data_tpose.json")

        if not os.path.exists(canonical_markers_path):
            raise FileNotFoundError(f"Canonical Pointcloud (p_bind) file not found at {canonical_markers_path}")
        else:
            logger.debug(f"[CANONICAL] Canonical Pointcloud (p_bind) Path: {canonical_markers_path}")   

        logger.success("[CANONICAL] Canonical Data defined!")
        logger.info("============================================")

        # ----------------

        # ----------------
        # E. LBS 

        skin_weights_npy_path = os.path.join(CANONICAL_DATA_DIRECTORY, f'lbs_skin/{subject}_skin_lbs_weights_exported.npy')
        skin_weights_json_path = os.path.join(CANONICAL_DATA_DIRECTORY, f'lbs_skin/{subject}_skin_lbs_weights_exported.json')

        if not os.path.exists(skin_weights_npy_path):
            raise FileNotFoundError(f"LBS Skin Weights file not found at {skin_weights_npy_path}")
        else:
            logger.debug(f"[LBS] LBS Skin Weights Path: {skin_weights_npy_path}")
            
        logger.success("[LBS] LBS Skin weight defined!")
        logger.info("============================================")

        # ----------------

        # ----------------
        # F. Barycentrinc Interpolation

        bary_map_path = os.path.join(CANONICAL_DATA_DIRECTORY, 'generated_marker_barycentric_map.json')

        if not os.path.exists(bary_map_path):
            raise FileNotFoundError(f"[BARYCENTRIC] Barycentric Map file not found at {bary_map_path}")
        else:
            logger.debug(f"[BARYCENTRIC] Barycentric Map Path: {bary_map_path}")

        logger.success("[BARYCENTRIC] Barycentric Map defined!")
        logger.info("============================================")

        # ----------------

        # ----------------
        # H. Muscle Vertex Mapping

        muscle_vertex_mapping_path = os.path.join(CANONICAL_DATA_DIRECTORY, 'individual_muscle_to_skin_binding.json')

        if not os.path.exists(muscle_vertex_mapping_path):
            logger.warning(f"[MUSCLE] Muscle Vertex Mapping file not found at {muscle_vertex_mapping_path}")
        else:
            logger.debug(f"[MUSCLE] Muscle Vertex Mapping Path: {muscle_vertex_mapping_path}")

        logger.success("[MUSCLE] Muscle Vertex Mapping defined!")
        logger.info("============================================")

        # ----------------

    except Exception as e:
        logger.error("[ERROR] Configuring paths. Please check the paths.")
        raise e

    # ========================================================================================================================================================
    # 2. Prepare Dataset and DataLoader
    logger.info("+++++++++++++++++++++++++++++++")
    logger.info("[INFO] 2. PREPARING DATA...")
    logger.info("+++++++++++++++++++++++++++++++")
    # ========================================================================================================================================================

    try:
        # ----------------
        # 0. Dataset

        logger.info("[DATA] Setting up datasets and dataloaders...")

        if not os.path.isdir(PROCESSED_DATA_DIRECTORY):
            logger.error(f"[DATA] The specified processed data directory does not exist: {PROCESSED_DATA_DIRECTORY}")
            logger.info("[DATA] Please run the pre-processing script first and update the path.")
        else:
            full_dataset = ProcessedMotionDataset(PROCESSED_DATA_DIRECTORY, preload=CONFIG["preload_to_ram"])
            if len(full_dataset) == 0:
                logger.error("[DATA] Dataset is empty. Did the pre-processing run correctly?")
            else:
                logger.debug(f"[DATA] Found {full_dataset.__len__()} total frames in the processed dataset.")

        # Always split the dataset to get a consistent validation set for monitoring
        train_size = int(TRAIN_SIZE * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset_full, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Save validation set information for later use
        # validation_indices = val_dataset.indices
        # all_files = full_dataset.pose_rot_files
        # validation_files = [all_files[i] for i in validation_indices]

        val_json_path = f"{subject}_validation_filepaths.json"
        all_files = full_dataset.pose_rot_files

        # # Delete any existing json with the name 
        # if os.path.exists("validation_filepaths.json"):
        #     os.remove("validation_filepaths.json")

        if os.path.exists(val_json_path):
            # 1. IF SPLIT EXISTS: Load it to guarantee fair comparison!
            logger.info(f"[DATA] Found existing validation split at {val_json_path}. Loading...")
            with open(val_json_path, "r") as f:
                validation_files = json.load(f)
            
            # Map filenames back to dataset indices
            val_files_set = set(validation_files)
            train_indices = [i for i, f in enumerate(all_files) if f not in val_files_set]
            val_indices = [i for i, f in enumerate(all_files) if f in val_files_set]
            
            train_dataset_full = Subset(full_dataset, train_indices)
            val_dataset = Subset(full_dataset, val_indices)
        else:
            # 2. IF SPLIT DOES NOT EXIST: Create it and save it forever.
            logger.warning(f"[DATA] No existing validation split found. Creating a new one...")
            train_size = int(TRAIN_SIZE * len(full_dataset))
            val_size = len(full_dataset) - train_size
            
            # Use a fixed generator seed for reproducibility just in case
            generator = torch.Generator().manual_seed(42)
            train_dataset_full, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
            
            validation_indices = val_dataset.indices
            validation_files = [all_files[i] for i in validation_indices]
            
            with open(val_json_path, "w") as f:
                json.dump(validation_files, f, indent=4)
            logger.success(f"[DATA] Saved new validation split to {val_json_path}")

        # with open(f"{subject}_validation_filepaths.json", "w") as f:
        #     json.dump(validation_files, f, indent=4)

        # logger.success(f"[DATA] Saved {len(validation_files)} validation file paths to validation_filepaths.json")

        # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=CONFIG["num_workers"])
        
        logger.success(f"[DATA] Full training samples: {len(train_dataset_full)}, Validation samples: {len(val_dataset)}")
        logger.info("============================================")

        # ----------------

        # # Calculate Global Offset Correction (Bias Removal)
        # # We take the residuals from Frame 0 (index 0 of sorted dataset) and subtract them from all others.
        # # This ensures Frame 0 is exactly the canonical pose, removing rigid offsets in the data.
        # logger.info("[DATA] Calculating Global Offset (Bias) from Frame 0...")
        # _, res_0_tensor, _, _ = full_dataset[0] 
        # global_offset = res_0_tensor.view(-1, 3) # Shape: (N, 3)

        # # Safety check
        # if torch.isnan(global_offset).any():
        #     logger.error("[CRITICAL] Frame 0 contains NaNs! Global offset is corrupted.")
        #     # Fallback to zero if data is bad
        #     global_offset = torch.zeros_like(global_offset)

        # # Move to device BEFORE using in training loop
        # global_offset = global_offset.to(device)

        # logger.success(f"[DATA] Global Offset loaded. Shape: {global_offset.shape}. This will be subtracted from all GT residuals.")
        # logger.info("============================================")

        # ----------------

        OVERFIT_SINGLE = CONFIG.get("overfit_single")
        OVERFIT_MULTIPLE = CONFIG.get("overfit_multiple")
        VISUALIZATION_BOOL = CONFIG.get("viz_enabled")

        arch_name = CONFIG["architecture"].upper()

        MODEL = f"ABLATION_{arch_name}_{ACTIVE_PRESET}"

        # --- NEW: Modify name if fine-tuning ---
        if CONFIG.get("checkpoint_path"):
            MODEL += "_FINETUNED"

        # ----------------

        if OVERFIT_SINGLE:
            logger.warning("[WARNING] Overfit SINGLE SAMPLE mode ACTIVE.")
            
            rot_s, res_s, mask_s, canon_s = full_dataset[0]  # unbatched: (J*6,), (2310,3), (2310,), (2310,3)

            class OneFrameDataset(Dataset):
                def __len__(self): return 1
                def __getitem__(self, idx): return rot_s, res_s, mask_s, canon_s

            train_loader = DataLoader(OneFrameDataset(), batch_size=1, shuffle=False, drop_last = True)
            val_loader = None  # ← No validation during overfitting

            # Replacing hyperparameters
            EPOCHS = 30000
            LEARNING_RATE = 1e-4
            WEIGHT_DECAY = 0.0
        elif OVERFIT_MULTIPLE: 
            logger.warning("[WARNING] Overfit MULTIPLE SAMPLES mode ACTIVE.")
            
            # Select the first X samples for overfitting
            num_overfit = 500

            # Train on first 300
            train_indices = list(range(num_overfit))

            # Create the subsets manually
            train_loader = DataLoader(Subset(full_dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
            val_loader = None  # ← No validation during overfitting

            # Replacing hyperparameters
            EPOCHS = 30
            LEARNING_RATE = 1e-4
            WEIGHT_DECAY = 0.0
        else:
            logger.warning("[WARNING] TRAINING FULL DATASET")
            train_loader = DataLoader(train_dataset_full, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
            # val_loader already set above from random_split

        logger.success("[DATA] Datasets and dataloaders created successfully.")
        logger.info("============================================")

        # ----------------

        # ----------------
        # A1. BVH file (Global - RED)

        # Load the global BVH file
        bvh_global = BVH()
        bvh_global.load(sample_bvh_path)

        # Convert joint names to standard Python strings
        joint_names_global = [str(name) for name in bvh_global.data['names']]
        num_joints_global = len(joint_names_global)

        logger.success(f"[BVH] Inferred num_joints_global of joints from .bvh data: {num_joints_global}")
        logger.info("============================================")
        
        # Print bones with indexes
        for i, name in enumerate(joint_names_global):
            logger.info(f"[BVH] Preparing BVH (global) with Joint {i}: {name}")
        logger.info(f"[BVH] Loaded BVH (global) with {num_joints_global} joints.")

        if subject == "S1":
            SCALE = 1.0
        else:
            SCALE = 0.001

        # Get the rest joint locations and convert to torch tensors
        j_rest_global, parents_global, offsets_global = get_rest_joint_locations(bvh_global, scale=SCALE) # Scale is always 1.0 / offset is NOT set to zero in the hips
        j_rest_tensor_global = torch.from_numpy(j_rest_global).float() # (24, 3)    <------------------------------------ TENSOR
        parents_tensor_global = torch.from_numpy(parents_global).long() # (24,)     <------------------------------------ TENSOR
        offsets_tensor_global = torch.from_numpy(offsets_global).float() # (24, 3)  <------------------------------------ TENSOR

        # ----------------

        # ----------------
        # A2. 6D Representation

        logger.info("[BVH] Determining skeleton structure from pre-processed data...")

        # Get the first sample file to determine the pose vector size
        sample_rot_path = glob.glob(os.path.join(PROCESSED_DATA_DIRECTORY, 'pose_rotations', '*.npy'))[0]
        sample_rot_vector = np.load(sample_rot_path)
        
        # Infer number of joints from the length of the pose vector directly from NPY file
        num_joints = len(sample_rot_vector) // 6 # 24 joints, 6 channels each (6D)
        logger.success(f"[BVH] Inferred number of joints from .npy data: {num_joints}")

        # Get rest joint locations and parents from BVH
        bvh_6d = BVH()
        bvh_6d.load(sample_bvh_path)

        # Convert joint names to standard Python strings
        joint_names_6d = [str(name) for name in bvh_6d.data['names']]
        num_joints_6d = len(joint_names_6d)

        logger.success(f"[BVH] Inferred num_joints_6d of joints from .bvh data: {num_joints_6d}")
        logger.info("============================================")

        if subject == "S1":
            SCALE = 1.0
        else:
            SCALE = 0.001
        
        j_rest_6d, parents_6d, offsets_6d = get_rest_joint_locations_zero_offset(bvh_6d, scale=SCALE)
        j_rest_tensor_6d = torch.from_numpy(j_rest_6d).float() # (24, 3)    <------------------------------------ TENSOR
        parents_tensor_6d = torch.from_numpy(parents_6d).long() # (24,)     <------------------------------------ TENSOR
        offsets_tensor_6d = torch.from_numpy(offsets_6d).float() # (24, 3)  <------------------------------------ TENSOR

        # Print types for debugging
        logger.debug(f"[BVH] j_rest_tensor_6d shape: {j_rest_tensor_6d.shape}")
        logger.debug(f"[BVH] parents_tensor_6d shape: {parents_tensor_6d.shape}")
        logger.debug(f"[BVH] offsets_tensor_6d shape: {offsets_tensor_6d.shape}")
        logger.debug(f"[BVH] j_rest_6d type: {type(j_rest_6d)}")
        logger.debug(f"[BVH] j_rest_6d dtype: {j_rest_6d.dtype}")
        logger.debug(f"[BVH] parents_6d type: {type(parents_6d)}")
        logger.debug(f"[BVH] parents_6d dtype: {parents_6d.dtype}")

        # Add a check to ensure joint counts match
        # j_rest_6d.shape[0] is 24, same as num_joints inferred from data
        if j_rest_6d.shape[0] != num_joints_6d:
            logger.error(f"[BVH] Joint count mismatch! Inferred {num_joints_6d} from data, but sample BVH has {j_rest_6d.shape[0]}.")
            logger.error(f"[BVH] Please ensure all BVH files have the same skeleton.")
        else:
            logger.success(f"[BVH] Joint count MATCH! Inferred {num_joints} from .npy data, and sample BVH has {j_rest_6d.shape[0]}")
            logger.info("============================================")

        # ----------------

        # ----------------
        # B. Static Meshes
        
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
            logger.success(f"[MESH] Vertices: skin={s_bind_vertices_np.shape}, muscle={m_bind_vertices_np.shape}, bone={b_bind_vertices_np.shape}")
            logger.success(f"[MESH] Faces: skin={len(skin_layer.faces)}, muscle={len(musc_layer.faces)}, bone={len(bone_layer.faces)}")
            logger.info("============================================")

            # ----------------

            # ----------------
            # C. Canonical Data
                
            # Load and process the canonical data
            logger.info("[MARKER] Loading and processing canonical data...")

            with open(canonical_markers_path, 'r') as f:
                canonical_data = json.load(f).get("0", {})

            # Get a canonical, sorted list of marker IDs and their base positions
            canonical_marker_ids = sorted(canonical_data.keys())
            marker_id_to_index = {marker_id: i for i, marker_id in enumerate(canonical_marker_ids)}
            num_markers = len(canonical_marker_ids)

            # This is because in the barycentric map, there are 4 markers that do not map to the skin mesh
            # We will remove these markers from the canonical data when calculating the loss
            logger.debug(f"[MARKER] Found {num_markers} canonical markers.") # 2310

            # Load barycentric map and extract bary_verts and bary_weights
            with open(bary_map_path, 'r') as f:
                barycentric_map = json.load(f)

            bary_marker_ids = sorted(barycentric_map.keys())
            p_bind = np.zeros((len(bary_marker_ids), 3), dtype=np.float32)
            for i, marker_id in enumerate(bary_marker_ids):
                pos_list = canonical_data[marker_id][0]
                p_bind[i] = np.array(pos_list)

            # Rotate canonical markers to align with BVH visualization
            DEGREE_BVH_X = -90.0
            p_bind = _rot_x(p_bind, deg=DEGREE_BVH_X)

            p_bind_tensor = torch.from_numpy(p_bind).float() #  <------------------------------------ TENSOR

            logger.success(f"[MARKER] Canonical Bind Positions loaded. Shape: {p_bind_tensor.shape}")
            logger.info("============================================")

            # ----------------

            # ----------------
            # E. LBS

            logger.info("[LBS] Loading weights and skeleton info...")

            weights_data = convert_weights_to_npy(skin_weights_json_path, len(s_bind_vertices), skin_weights_npy_path, joint_names_global, is_skin=True)
            skin_weights_tensor = torch.from_numpy(weights_data).float() #  <------------------------------------ TENSOR

            logger.success(f"[LBS] Loaded skin_weights matrix with shape: {skin_weights_tensor.shape}")  # (23752, 24)
            logger.info("============================================")

            # ---------------- 

            # ----------------
            # F. Barycentrinc Interpolation
            
            # Load and process the barycentric map
            logger.info("[BARYCENTRIC] Loading and processing barycentric map...")

            with open(bary_map_path, 'r') as f:
                barycentric_map = json.load(f)

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

            # ----------------
            # G. MUSCLE TO MARKER

            print("TO IMPLEMENT FROM THE OTHER SCRIPT")

            # ----------------

            # ----------------
            # H. Muscle Vertex Mapping -> [[DONE]]

            logger.info("[MAPPING] Loading Muscle-Skin Map...")

            muscle_bary_map = None
            with open(muscle_vertex_mapping_path, 'r') as f:
                muscle_bary_map = json.load(f)
                
            logger.success(f"[MAPPING] Loaded Muscle-Skin Map from {muscle_vertex_mapping_path}")
            logger.info("============================================")

            # ----------------

        except Exception as e:
            print(f"Error loading mesh file: {e}")

        # ========================================================================================================================================================
        # 3. Prepare the Model
        logger.info("+++++++++++++++++++++++++++++++")
        logger.info("[INFO] 3. PREPARING THE MODEL...")
        logger.info("+++++++++++++++++++++++++++++++")
        # ========================================================================================================================================================

        try:            
            # MESH GEOMETRY: Vertex positions and face indices
            mesh_geometry_tensors = {
                'm_bind': m_bind_vertices,           # Muscle bind pose vertices (23752, 3)
                's_bind': s_bind_vertices,           # Skin bind pose vertices (23752, 3)
                'b_bind': b_bind_vertices,           # Bone bind pose vertices (23752, 3)
                'm_faces': m_bind_faces,             # Muscle face indices (n_faces, 3)
                's_faces': s_bind_faces,             # Skin face indices (n_faces, 3)
                'b_faces': b_bind_faces             # Bone face indices (n_faces, 3)
            }
            
            # MESH PROPERTIES: Areas, normals, masses
            mesh_properties_tensors = {
                'm_rest_areas': m_rest_areas,        # Muscle face rest areas (n_faces,)
                's_rest_areas': s_rest_areas,        # Skin face rest areas (n_faces,)
                'b_rest_areas': b_rest_areas,        # Bone face rest areas (n_faces,)
                'm_vertex_mass': m_vertex_mass,      # Muscle vertex mass diagonal (23752,)
                's_vertex_mass': s_vertex_mass,      # Skin vertex mass diagonal (23752,)
                'b_vertex_mass': b_vertex_mass,      # Bone vertex mass diagonal (23752,)
                'm_normals': m_normals_tensor,       # Muscle vertex normals (23752, 3)
                's_normals': s_normals_tensor,       # Skin vertex normals (23752, 3)
                'b_normals': b_normals_tensor,       # Bone vertex normals (23752, 3)
                'm_face_adjacency': musc_props['face_adjacency'],
                's_face_adjacency': skin_props['face_adjacency'],
                'b_face_adjacency': bone_props['face_adjacency'],
            }
            
            # MESH REGULARIZATION: Laplacians and edge weights
            mesh_regularization_tensors = {
                'l_muscle': l_muscle_torch,          # Muscle Laplacian (23752, 23752) sparse
                'l_muscle_degree': muscle_degree_tensor,  # Muscle Laplacian degree (23752,)
                'l_skin': l_skin_torch,              # Skin Laplacian (23752, 23752) sparse
                'l_bone': l_bone_torch,              # Bone Laplacian (23752, 23752) sparse
                'musc_edge_weights': musc_edge_weights,   # Muscle edge weights (n_edges,)
                'skin_edge_weights': skin_edge_weights,   # Skin edge weights (n_edges,)
                'bone_edge_weights': bone_edge_weights,   # Bone edge weights (n_edges,)
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
            m_face_adjacency = device_tensors['m_face_adjacency']
            s_face_adjacency = device_tensors['s_face_adjacency']
            b_face_adjacency = device_tensors['b_face_adjacency']
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
                stability_mask=stability_mask,
                architecture=CONFIG["architecture"]
            )

            logger.success(f"[MODEL] SOMAModel instantiated successfully.")
            logger.success(f"[MODEL] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

            model = model.to(device)
            logger.success(f"[MODEL] Model moved to device: {device}")

            # =================================================================
            # --- NEW: LOAD CHECKPOINT (IF PROVIDED) ---
            # =================================================================
            checkpoint_path = CONFIG.get("checkpoint_path")
            if checkpoint_path and os.path.exists(checkpoint_path):
                logger.info(f"[MODEL] Loading pre-trained weights from: {checkpoint_path}")
                blendshape_weights = torch.load(checkpoint_path, map_location=device)
                
                # Load weights strictly into the deformation MLP
                model.deformation_model.load_state_dict(blendshape_weights)
                logger.success("[MODEL] Successfully loaded checkpoint! Resuming training/fine-tuning.")
            elif checkpoint_path:
                logger.warning(f"[MODEL] Checkpoint path provided but file NOT FOUND: {checkpoint_path}. Training from scratch.")
            else:
                logger.info("[MODEL] No checkpoint path provided. Training from scratch.")
            # =================================================================

        except Exception as e:
            logger.error("[ERROR] Model instantiation failed.")
            raise e

        # ========================================================================================================================================================
        # 4. Defining Loss Function and Optimizer
        logger.info("+++++++++++++++++++++++++++++++")
        logger.info("[INFO] 4. LOSS...")
        logger.info("+++++++++++++++++++++++++++++++")
        # ========================================================================================================================================================

        try:
            # Create geometry dictionaries for the loss function
            skin_geo = {
                'verts': s_bind,
                'faces': s_faces,
                'edges': skin_edges_np,
                'rest_areas': s_rest_areas,
                'vertex_mass': s_vertex_mass,
                'normals': s_normals_tensor,
                'laplacian': l_skin,
                'edge_weights': skin_edge_weights,
                'face_adjacency': s_face_adjacency,
            }
            
            muscle_geo = {
                'verts': m_bind,
                'faces': m_faces,
                'edges': musc_edges_np,
                'rest_areas': m_rest_areas,
                'vertex_mass': m_vertex_mass,
                'normals': m_normals_tensor,
                'laplacian': l_muscle,
                'edge_weights': musc_edge_weights,
                'face_adjacency': m_face_adjacency,
            }

            bone_geo = {
                'verts': b_bind,
                'faces': b_faces,
                'edges': bone_edges_np,
                'rest_areas': b_rest_areas,
                'vertex_mass': b_vertex_mass,
                'normals': b_normals_tensor,
                'laplacian': l_bone,
                'edge_weights': bone_edge_weights,
                'face_adjacency': b_face_adjacency,
            }
            
            loss_fn = SOMALossOriginal(
                skin_geo=skin_geo,
                muscle_geo=muscle_geo,
                bone_geo=bone_geo,
                lambdas=LAMBDAS
            )

            logger.success("[LOSS] Loss function created successfully.")

            # We only want to train the parameters of the muscle model
            logger.info(f"[LOSS] LR: {LEARNING_RATE}")
            optimizer = torch.optim.Adam(model.deformation_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

            logger.success("[LOSS] Optimizer created successfully.")

        except Exception as e:
            logger.error(f"[ERROR] Failed to create loss function: {e}")
            raise e

        # ========================================================================================================================================================
        # 5. Setup Optimizer
        logger.info("+++++++++++++++++++++++++++++++")
        logger.info("[INFO] 5. OPTIMIZER SETUP...")
        logger.info("+++++++++++++++++++++++++++++++")
        # ========================================================================================================================================================

        try:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY
            )
            logger.success(f"[TRAIN] Optimizer created: Adam with lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to create optimizer: {e}")
            raise e

        # ========================================================================================================================================================
        # 6. Training Loop
        logger.info("+++++++++++++++++++++++++++++++")
        logger.info("[INFO] 6. TRAINING LOOP...")
        logger.info("+++++++++++++++++++++++++++++++")
        # ========================================================================================================================================================

        logger.info("[TRAIN] Starting training loop...")
        writer = SummaryWriter(os.path.join(PATHS['logs'], MODEL))
        best_val_loss = float('inf')
        best_train_loss = float('inf')

        for epoch in range(EPOCHS):
            model.train()

            # Accumulated Training Losses
            train_loss_total = 0.0
            train_data_loss_total = 0.0
            train_smooth_m_total = 0.0  
            train_smooth_s_total = 0.0  
            train_stretch_m_total = 0.0
            train_stretch_s_total = 0.0
            train_tangent_m_total = 0.0
            train_tangent_s_total = 0.0
            train_biharmonic_m_total = 0.0
            train_biharmonic_s_total = 0.0
            train_volume_s_total = 0.0
            train_volume_m_total = 0.0

            num_train_batches = 0

            # Training Progress
            train_pbar = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{EPOCHS}", position=0, leave=True)

            for batch_idx, (pose_rot, residuals_gt, masks, canonical_lbs) in enumerate(train_pbar):

                # Moving to device
                pose_rot = pose_rot.to(device)
                residuals_gt = residuals_gt.to(device)
                masks = masks.to(device)
                canonical_lbs = canonical_lbs.to(device)

                batch_size = pose_rot.shape[0]

                # A. Prepare constant identity pose (T-pose)
                identity_quat = np.zeros((j_rest_6d.shape[0], 4), dtype=np.float32)
                identity_quat[:, 0] = 1.0  # w=1, x=y=z=0
                rot_identity = quat_torch.to_matrix(torch.from_numpy(identity_quat).float().to(device)) 
                rot_identity_tensor = rot_identity.float().unsqueeze(0).to(device)  # (1, 24, 3, 3)

                # B. Prepare current pose from BVH (6D continuous representation)
                # Reshape pose_rot from flat vector to (batch, num_joints, 6)
                rot6d_tensor = pose_rot.view(batch_size, num_joints, 6) # torch.Size([B, 24, 6])  

                # C. Root position
                root_position = torch.zeros(batch_size, 3, device=device, dtype=s_bind.dtype) # torch.Size([B, 3])

                # D. GT Residuals
                batch_size_residuals = residuals_gt.size(0)  # Get the actual batch size
                residuals_gt = residuals_gt.view(batch_size_residuals, -1, 3)  # Automatically infer num_markers

                # V2: REMOVING
                # [FIX] Apply Global Offset Correction
                # Subtract the bias (frame 0 residuals) from the current batch residuals
                # residuals_gt = residuals_gt - global_offset

                # [FIX] Ensure invalid markers (mask=0) remain at 0 residual.
                # Otherwise, '0 - global_offset' creates a phantom displacement for missing data.
                residuals_gt[masks == 0] = 0.0

                # Forward pass - returns dictionary
                forward_output = model(rot6d_tensor, rot_identity_tensor, root_position)
                
                p_pred_theta = forward_output['p_pred_theta']        # (B, M, 3) - predicted markers in pose space
                p_pred_tpose = forward_output['p_pred_tpose']        # (B, M, 3) - predicted markers in t-pose
                s_final_theta = forward_output['s_final_theta']      # (B, V, 3) - skin vertices in pose space
                m_final_theta = forward_output['m_final_theta']      # (B, V, 3) - muscle vertices in pose space
                b_final_theta = forward_output['b_final_theta']      # (B, V, 3) - bone vertices in pose space
                s_final = forward_output['s_final']      # (B, V, 3) - skin vertices in pose space
                m_final = forward_output['m_final']      # (B, V, 3) - muscle vertices in pose space
                d_skin = forward_output['d_skin']                    # (B, V, 3) - skin displacement
                d_muscle = forward_output['d_muscle']                # (B, V, 3) - muscle displacement

                # ---

                # # V1
                # # CRITICAL FIX: Add hip position offset to predicted markers
                # # The model predicts markers in local/canonical space relative to rest pose
                # # We need to add the hip joint position to transform to world coordinates
                # hips_idx = 0  # Root joint is at index 0
                # hips_pos = j_rest[hips_idx]  # (3,) - hip joint position in world space
                # p_pred_tpose_offset = p_pred_tpose + hips_pos  # Add offset: (B, M, 3)

                # total_loss, loss_dict = loss_fn(p_pred_tpose_offset, residuals_gt, masks, p_bind_tensor, s_final_theta, m_final_theta, d_skin, d_muscle)

                # # Calculate loss with GT markers
                # # DEBUG: Print shapes and values to understand what's happening
                # if batch_idx == 0 and epoch == 0:
                #     logger.debug(f"[DEBUG] p_pred_tpose_offset shape: {p_pred_tpose_offset.shape}, min: {p_pred_tpose_offset.min():.6f}, max: {p_pred_tpose_offset.max():.6f}")
                #     logger.debug(f"[DEBUG] residuals_gt min: {residuals_gt.min():.6f}, max: {residuals_gt.max():.6f}")
                #     logger.debug(f"[DEBUG] masks shape: {masks.shape}, sum: {masks.sum():.0f}")

                # V2
                # A. Construct Worlds
                hips_idx = 0
                hips_pos = j_rest[hips_idx]
                pred_world = p_pred_tpose + hips_pos  
                target_world = p_bind_tensor + residuals_gt

                # B. Calculate Centroids
                pred_centroid = pred_world.mean(dim=1, keepdim=True)
                target_centroid = target_world.mean(dim=1, keepdim=True)

                # C. Center Both
                pred_centered = pred_world - pred_centroid
                target_centered = target_world - target_centroid

                # D. Loss Function (Pass Zeros for p_bind)
                zero_bind = torch.zeros_like(pred_centered)
                total_loss, loss_dict = loss_fn(
                    pred_centered, 
                    target_centered, 
                    masks, 
                    zero_bind, 
                    s_final_theta, 
                    m_final_theta, 
                    b_final_theta,
                    d_skin, 
                    d_muscle
                )

                # Calculate loss with GT markers
                # DEBUG: Print shapes and values to understand what's happening
                if batch_idx == 0 and epoch == 0:
                    logger.debug(f"[DEBUG] pred_world shape: {pred_world.shape}, min: {pred_world.min():.6f}, max: {pred_world.max():.6f}")
                    logger.debug(f"[DEBUG] target_world shape: {target_world.shape}, min: {target_world.min():.6f}, max: {target_world.max():.6f}")
                    logger.debug(f"[DEBUG] masks shape: {masks.shape}, sum: {masks.sum():.0f}")

                # ---
                
                # =================================================================                
                if VISUALIZATION_BOOL == True:
                    if epoch == 1 and batch_idx == 0:
                        try:
                            # We need to calculate the posed skeleton for visualization context
                            rot6d_np = rot6d_tensor[0].cpu().numpy() # First item in batch
                            C = np.stack([rot6d_np[:, :3], rot6d_np[:, 3:]], axis=-1)
                            reconstructed_quats = sixd.to_quat(C)
                            posed_joints, _ = fk(reconstructed_quats, np.zeros(3), offsets_6d, parents_6d)

                            # Predicted Muscle and Skin Wrap
                            m_final_viz = m_final[0].detach().cpu().numpy() 
                            s_final_viz = s_final[0].detach().cpu().numpy()
                            
                            # Muscle Displacement
                            d_muscle_viz = d_muscle[0].detach().cpu().numpy()

                            visualize_output(
                                p_bind=p_bind_tensor.cpu().numpy(),
                                residuals_gt=residuals_gt[0].detach().cpu().numpy(),
                                p_pred=pred_world[0].detach().cpu().numpy(),
                                skin_layer=skin_layer,
                                musc_layer=musc_layer,
                                posed_joints=posed_joints,
                                parents=parents_6d,
                                current_m_final=m_final_viz,
                                current_s_final=s_final_viz,
                                d_muscle=d_muscle_viz,
                                rest_vol_skin=loss_fn.rest_vol_skin
                            )

                            logger.info("Visualization closed. Resuming training.")
                        except Exception as e:
                            logger.error(f"Could not launch visualization: {e}")
                # =================================================================

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()

                # Clip Gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss_total += total_loss.item()
                train_data_loss_total += loss_dict.get('data', 0.0)
                train_smooth_m_total += loss_dict.get('smooth_musc', 0.0) 
                train_smooth_s_total += loss_dict.get('smooth_skin', 0.0) 
                train_biharmonic_m_total += loss_dict.get('biharmonic_musc', 0.0)
                train_biharmonic_s_total += loss_dict.get('biharmonic_skin', 0.0)
                train_stretch_m_total += loss_dict.get('stretch_musc', 0.0)
                train_stretch_s_total += loss_dict.get('stretch_skin', 0.0)
                train_tangent_m_total += loss_dict.get('tangent_musc', 0.0)
                train_tangent_s_total += loss_dict.get('tangent_skin', 0.0)
                train_volume_m_total += loss_dict.get('vol_musc', 0.0)
                train_volume_s_total += loss_dict.get('vol_skin', 0.0)
                
                num_train_batches += 1

                # =================================================================
                # ===== TENSORBOARD LOGGING (BATCH-LEVEL) =========================
                # =================================================================
                global_step = epoch * len(train_loader) + batch_idx
                
                writer.add_scalar('Loss_Batch/Train_Total', total_loss.item(), global_step)
                writer.add_scalar('Loss_Batch/Train_Data', loss_dict.get('data', 0.0), global_step)
                writer.add_scalar('Loss_Batch/Train_Smooth_M', loss_dict.get('smooth_musc', 0.0), global_step) # <--- NEW
                writer.add_scalar('Loss_Batch/Train_Smooth_S', loss_dict.get('smooth_skin', 0.0), global_step) # <--- NEW
                writer.add_scalar('Loss_Batch/Train_Biharmonic_M', loss_dict.get('biharmonic_musc', 0.0), global_step) # <--- NEW
                writer.add_scalar('Loss_Batch/Train_Biharmonic_S', loss_dict.get('biharmonic_skin', 0.0), global_step) # <--- NEW
                writer.add_scalar('Loss_Batch/Train_Stretch_M', loss_dict.get('stretch_musc', 0.0), global_step) # <--- NEW
                writer.add_scalar('Loss_Batch/Train_Stretch_S', loss_dict.get('stretch_skin', 0.0), global_step) # <--- NEW
                writer.add_scalar('Loss_Batch/Train_Tangent_M', loss_dict.get('tangent_musc', 0.0), global_step) # <--- NEW
                writer.add_scalar('Loss_Batch/Train_Tangent_S', loss_dict.get('tangent_skin', 0.0), global_step) # <--- NEW
                writer.add_scalar('Loss_Batch/Train_Volume_M', loss_dict.get('vol_musc', 0.0), global_step) # <--- NEW
                writer.add_scalar('Loss_Batch/Train_Volume_S', loss_dict.get('vol_skin', 0.0), global_step) # <--- NEW
                # =================================================================

                # Update progress bar
                train_pbar.set_postfix({
                    'tot': f"{total_loss.item():.5f}",
                    'data': f"{loss_dict.get('data', 0.0):.5f}",
                    'sm_m': f"{loss_dict.get('smooth_musc', 0.0):.5f}",
                    'sm_s': f"{loss_dict.get('smooth_skin', 0.0):.5f}",
                    'bi_m': f"{loss_dict.get('biharmonic_musc', 0.0):.5f}",
                    'bi_s': f"{loss_dict.get('biharmonic_skin', 0.0):.5f}",
                    'st_m': f"{loss_dict.get('stretch_musc', 0.0):.5f}",
                    'st_s': f"{loss_dict.get('stretch_skin', 0.0):.5f}",
                    'tan_m': f"{loss_dict.get('tangent_musc', 0.0):.5f}",
                    'tan_s': f"{loss_dict.get('tangent_skin', 0.0):.5f}",
                    'vol_m': f"{loss_dict.get('vol_musc', 0.0):.5f}",
                    'vol_s': f"{loss_dict.get('vol_skin', 0.0):.5f}"
                    
                })

            # --- End of Training Batch Loop ---

            # Log epoch-level training metrics
            avg_train_loss = train_loss_total / num_train_batches
            avg_train_data_loss = train_data_loss_total / num_train_batches
            avg_train_sm_m = train_smooth_m_total / num_train_batches
            avg_train_sm_s = train_smooth_s_total / num_train_batches
            avg_train_st_m = train_stretch_m_total / num_train_batches
            avg_train_st_s = train_stretch_s_total / num_train_batches
            avg_train_tan_m = train_tangent_m_total / num_train_batches
            avg_train_tan_s = train_tangent_s_total / num_train_batches
            avg_train_vol_m = train_volume_m_total / num_train_batches
            avg_train_vol_s = train_volume_s_total / num_train_batches
            avg_train_bi_m = train_biharmonic_m_total / num_train_batches
            avg_train_bi_s = train_biharmonic_s_total / num_train_batches
            
            writer.add_scalar('Loss_Epoch/Train_Total', avg_train_loss, epoch)
            writer.add_scalar('Loss_Epoch/Train_Data', avg_train_data_loss, epoch)
            writer.add_scalar('Loss_Epoch/Train_Smooth_M', avg_train_sm_m, epoch)
            writer.add_scalar('Loss_Epoch/Train_Smooth_S', avg_train_sm_s, epoch)
            writer.add_scalar('Loss_Epoch/Train_Stretch_M', avg_train_st_m, epoch)
            writer.add_scalar('Loss_Epoch/Train_Stretch_S', avg_train_st_s, epoch)
            writer.add_scalar('Loss_Epoch/Train_Tangent_M', avg_train_tan_m, epoch)
            writer.add_scalar('Loss_Epoch/Train_Tangent_S', avg_train_tan_s, epoch)
            writer.add_scalar('Loss_Epoch/Train_Volume_M', avg_train_vol_m, epoch)
            writer.add_scalar('Loss_Epoch/Train_Volume_S', avg_train_vol_s, epoch)
            writer.add_scalar('Loss_Epoch/Train_Biharmonic_M', avg_train_bi_m, epoch)
            writer.add_scalar('Loss_Epoch/Train_Biharmonic_S', avg_train_bi_s, epoch)

            # Save best model based on validation loss
            if CONFIG["overfit_multiple"] == True or CONFIG["overfit_single"] == True:
                if avg_train_loss < best_train_loss:
                    best_train_loss = avg_train_loss
                    checkpoint_path = os.path.join(PATHS['checkpoints'], f'{MODEL}_epoch_{epoch}.pth')
                    os.makedirs(PATHS['checkpoints'], exist_ok=True)
                    torch.save(model.deformation_model.state_dict(), checkpoint_path)
                    logger.success(f"✓ New best model saved (Train Loss: {best_train_loss:.6f})")
                
            logger.info(f"[TRAIN] Epoch {epoch+1}/{EPOCHS} | Tot: {avg_train_loss:.5f} | Data: {avg_train_data_loss:.5f} | Sm_M: {avg_train_sm_m:.5f} | Sm_S: {avg_train_sm_s:.5f} | St_M: {avg_train_st_m:.5f} | St_S: {avg_train_st_s:.5f} | Tan_M: {avg_train_tan_m:.5f} | Tan_S: {avg_train_tan_s:.5f} | Vol_M: {avg_train_vol_m:.5f} | Vol_S: {avg_train_vol_s:.5f} | Bi_M: {avg_train_bi_m:.5f} | Bi_S: {avg_train_bi_s:.5f}")
            
            # --- Validation Loop ---
            if val_loader is not None:
                model.train() # Keep train mode for BN statistics
                
                val_loss_total = 0.0
                val_data_loss_total = 0.0
                val_smooth_m_total = 0.0
                val_smooth_s_total = 0.0
                val_stretch_m_total = 0.0
                val_stretch_s_total = 0.0
                val_tangent_m_total = 0.0
                val_tangent_s_total = 0.0
                val_volume_m_total = 0.0
                val_volume_s_total = 0.0
                val_biharmonic_m_total = 0.0
                val_biharmonic_s_total = 0.0
        
                num_val_batches = 0

                val_pbar = tqdm(val_loader, desc=f"[Val] Epoch {epoch+1}/{EPOCHS}", position=1, leave=False)

                with torch.no_grad():
                    for val_pose_rot, val_residuals_gt, val_masks, val_canonical_lbs in val_pbar:
                        
                        # Move to device
                        val_pose_rot = val_pose_rot.to(device)
                        val_residuals_gt = val_residuals_gt.to(device)
                        val_masks = val_masks.to(device)
                        val_batch_size = val_pose_rot.shape[0]

                        # 1. Inputs
                        # [FIX] Create Identity Matrix correctly (B, J, 3, 3)
                        val_rot_identity_tensor = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(val_batch_size, j_rest.shape[0], 1, 1)
                        
                        val_rot6d_tensor = val_pose_rot.view(val_batch_size, j_rest.shape[0], 6)
                        val_root_position = torch.zeros(val_batch_size, 3, device=device, dtype=s_bind.dtype)

                        # 2. Forward Pass (Use model output directly)
                        val_forward_output = model(val_rot6d_tensor, val_rot_identity_tensor, val_root_position)
                        val_p_pred_tpose = val_forward_output['p_pred_tpose'] # (B, M, 3)
                        val_s_final_theta = val_forward_output['s_final_theta']
                        val_m_final_theta = val_forward_output['m_final_theta']
                        val_b_final_theta = val_forward_output['b_final_theta']
                        val_d_skin = val_forward_output['d_skin']
                        val_d_muscle = val_forward_output['d_muscle']

                        # 3. Add Hips Offset
                        hips_idx = 0
                        hips_pos = j_rest[hips_idx]
                        pred_world = val_p_pred_tpose + hips_pos
                        
                        # 4. Target World
                        target_world = p_bind_tensor + val_residuals_gt.view(val_batch_size, -1, 3)

                        # 5. Centering
                        pred_centroid = pred_world.mean(dim=1, keepdim=True)
                        target_centroid = target_world.mean(dim=1, keepdim=True)
                        
                        pred_centered = pred_world - pred_centroid
                        target_centered = target_world - target_centroid

                        # 6. Calculate Loss (Pass Zero for p_bind to compare centered shapes directly)
                        zero_bind = torch.zeros_like(pred_centered)
                        
                        val_total_loss, val_loss_dict = loss_fn(
                            pred_centered,      # Predicted Markers (Centered)
                            target_centered,    # GT Markers (Centered)
                            val_masks, 
                            zero_bind,          # Zero Bind
                            val_s_final_theta, 
                            val_m_final_theta, 
                            val_b_final_theta,
                            val_d_skin, 
                            val_d_muscle
                        )

                        val_loss_total += val_total_loss.item()
                        val_data_loss_total += val_loss_dict.get('data', 0.0)
                        val_smooth_m_total += val_loss_dict.get('smooth_musc', 0.0)
                        val_smooth_s_total += val_loss_dict.get('smooth_skin', 0.0)
                        val_stretch_m_total += val_loss_dict.get('stretch_musc', 0.0)
                        val_stretch_s_total += val_loss_dict.get('stretch_skin', 0.0)
                        val_tangent_m_total += val_loss_dict.get('tangent_musc', 0.0)
                        val_tangent_s_total += val_loss_dict.get('tangent_skin', 0.0)
                        val_biharmonic_m_total += val_loss_dict.get('biharmonic_musc', 0.0)
                        val_biharmonic_s_total += val_loss_dict.get('biharmonic_skin', 0.0)
                        val_volume_m_total += val_loss_dict.get('vol_musc', 0.0)
                        val_volume_s_total += val_loss_dict.get('vol_skin', 0.0)

                        num_val_batches += 1
                        
                        val_pbar.set_postfix({
                            'val_loss': val_total_loss.item(),
                            'val_data': val_loss_dict.get('data', 0.0),
                            'val_smooth_m': val_loss_dict.get('smooth_musc', 0.0),
                            'val_smooth_s': val_loss_dict.get('smooth_skin', 0.0),
                            'val_stretch_m': val_loss_dict.get('stretch_musc', 0.0),
                            'val_stretch_s': val_loss_dict.get('stretch_skin', 0.0),
                            'val_tangent_m': val_loss_dict.get('tangent_musc', 0.0),
                            'val_tangent_s': val_loss_dict.get('tangent_skin', 0.0),
                            'val_biharmonic_m': val_loss_dict.get('biharmonic_musc', 0.0),
                            'val_biharmonic_s': val_loss_dict.get('biharmonic_skin', 0.0),
                            'val_volume_m': val_loss_dict.get('vol_musc', 0.0),
                            'val_volume_s': val_loss_dict.get('vol_skin', 0.0)
                        })

                # Log epoch-level validation metrics
                avg_val_loss = val_loss_total / num_val_batches
                avg_val_data_loss = val_data_loss_total / num_val_batches
                avg_val_sm_m = val_smooth_m_total / num_val_batches
                avg_val_sm_s = val_smooth_s_total / num_val_batches
                avg_val_st_m = val_stretch_m_total / num_val_batches
                avg_val_st_s = val_stretch_s_total / num_val_batches
                avg_val_tan_m = val_tangent_m_total / num_val_batches
                avg_val_tan_s = val_tangent_s_total / num_val_batches
                avg_val_bi_m = val_biharmonic_m_total / num_val_batches
                avg_val_bi_s = val_biharmonic_s_total / num_val_batches
                avg_val_vol_m = val_volume_m_total / num_val_batches
                avg_val_vol_s = val_volume_s_total / num_val_batches
                
                writer.add_scalar('Loss_Epoch/Val_Total', avg_val_loss, epoch)
                writer.add_scalar('Loss_Epoch/Val_Data', avg_val_data_loss, epoch)
                writer.add_scalar('Loss_Epoch/Val_Smooth_M', avg_val_sm_m, epoch)
                writer.add_scalar('Loss_Epoch/Val_Smooth_S', avg_val_sm_s, epoch)
                writer.add_scalar('Loss_Epoch/Val_Stretch_M', avg_val_st_m, epoch)
                writer.add_scalar('Loss_Epoch/Val_Stretch_S', avg_val_st_s, epoch)
                writer.add_scalar('Loss_Epoch/Val_Tangent_M', avg_val_tan_m, epoch)
                writer.add_scalar('Loss_Epoch/Val_Tangent_S', avg_val_tan_s, epoch)
                writer.add_scalar('Loss_Epoch/Val_Biharmonic_M', avg_val_bi_m, epoch)
                writer.add_scalar('Loss_Epoch/Val_Biharmonic_S', avg_val_bi_s, epoch)
                writer.add_scalar('Loss_Epoch/Val_Volume_M', avg_val_vol_m, epoch)
                writer.add_scalar('Loss_Epoch/Val_Volume_S', avg_val_vol_s, epoch)

                logger.info(f"[VAL] Epoch {epoch+1}/{EPOCHS} - Tot: {avg_val_loss:.6f} | Data: {avg_val_data_loss:.6f} | Sm_M: {avg_val_sm_m:.6f} | Sm_S: {avg_val_sm_s:.6f} | St_M: {avg_val_st_m:.6f} | St_S: {avg_val_st_s:.6f} | Tan_M: {avg_val_tan_m:.6f} | Tan_S: {avg_val_tan_s:.6f} | Vol_M: {avg_val_vol_m:.6f} | Vol_S: {avg_val_vol_s:.6f} | Bi_M: {avg_val_bi_m:.6f} | Bi_S: {avg_val_bi_s:.6f}")

                # Save best model based on validation loss
                if CONFIG["overfit_multiple"] == False and CONFIG["overfit_single"] == False:
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        checkpoint_path = os.path.join(PATHS['checkpoints'], f'{MODEL}_epoch_{epoch}.pth')
                        os.makedirs(PATHS['checkpoints'], exist_ok=True)
                        torch.save(model.deformation_model.state_dict(), checkpoint_path)
                        logger.success(f"✓ New best model saved (Val Loss: {best_val_loss:.6f})")

                model.train()  # Switch back to training mode

        logger.success(f"[TRAIN] Training completed!")
        writer.close()

    except Exception as e:
        logger.error("[ERROR] Training failed. Please check the logs above.")
        raise e

if __name__ == '__main__':
    logger.info("=== END-TO-END TRAINING SCRIPT ===")
    logger.info(f"[CONFIG] Subject: {CONFIG['subject']} | Architecture: {CONFIG['architecture']} | Preset: {ACTIVE_PRESET}")
    logger.info(f"[CONFIG] Lambdas: { {k: v for k, v in LAMBDAS.items()} }")
    train_full_model()