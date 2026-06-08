import time
import viser
import numpy as np
import json
import os
import trimesh
import torch
import viser.transforms as tf
from pymotion.io.bvh import BVH
from pymotion.ops.skeleton import fk
from scipy.spatial.transform import Rotation
import pymotion.rotations.quat_torch as quat
from scipy.spatial import KDTree
import pymotion.ops.skeleton as sk

def load_weights(json_path, num_vertices, num_joints):
    """
    Load skinning weights from a JSON file and convert them to a dense matrix.
    
    Args:
        json_path (str): Path to the JSON file containing the weights.
        num_vertices (int): Number of vertices in the mesh.
        num_joints (int): Number of joints in the skeleton.
    
    Returns:
        torch.Tensor: Skinning weights as a (num_vertices, num_joints) tensor.
    """
    with open(json_path, 'r') as f:
        weights_data = json.load(f)

    weights_matrix = np.zeros((num_vertices, num_joints), dtype=np.float32)

    for v_idx, vertex_info in enumerate(weights_data):
        bone_names = vertex_info.get("bone_names", [])
        weights = vertex_info.get("weights", [])
        for bone_name, weight in zip(bone_names, weights):
            if bone_name in bone_names:
                joint_idx = bone_names.index(bone_name)
                weights_matrix[v_idx, joint_idx] = weight

    return torch.from_numpy(weights_matrix).float()

def lbs(vertices, pose_quats, weights, j_rest, parents, root_position):
    """
    Performs Linear Blend Skinning using quaternion rotations.
    
    Args:
        vertices (torch.Tensor): Vertices of the mesh in bind pose. (V, 3)
        pose_quats (torch.Tensor): Pose parameters as quaternions [w, x, y, z]. (J, 4)
        weights (torch.Tensor): Skinning weights. (V, J)
        j_rest (torch.Tensor): Joint locations in bind pose. (J, 3)
        parents (torch.Tensor): Parent index for each joint. (J,)
        root_position (torch.Tensor): The translation of the root joint for the current frame. (3,)
        
    Returns:
        torch.Tensor: Deformed vertices.
    """
    device = vertices.device
    num_joints = j_rest.shape[0]

    # 1. Convert quaternion rotations to 4x4 homogeneous transformation matrices
    rot_mats = quat.to_matrix(pose_quats)

    # 2. Calculate global transformations for the rest pose (G_rest)
    transforms_rest = torch.zeros(num_joints, 4, 4, device=device)
    for i in range(num_joints):
        if parents[i] == -1:  # Root in World space
            transform = torch.eye(4, device=device)
            transform[:3, 3] = j_rest[i]  # Set the position of the root joint
        else:
            local_offset_transform = torch.eye(4, device=device)
            local_offset_transform[:3, 3] = j_rest[i] - j_rest[parents[i]]  # Define the bone offset
            transform = torch.matmul(transforms_rest[parents[i]], local_offset_transform)
        transforms_rest[i] = transform

    # 3. Calculate global transformations for the posed skeleton (G_posed)
    transforms_posed = torch.zeros(num_joints, 4, 4, device=device)
    for i in range(num_joints):
        local_transform = torch.eye(4, device=device)
        local_transform[:3, :3] = rot_mats[i]  # Apply the rotation to the joint
        if parents[i] == -1:  # Root in World space
            local_transform[:3, 3] = root_position
            transform = local_transform
        else:
            local_transform[:3, 3] = j_rest[i] - j_rest[parents[i]]
            transform = torch.matmul(transforms_posed[parents[i]], local_transform)
        transforms_posed[i] = transform

    # 4. Calculate the final skinning matrices: T = G_posed * inv(G_rest)
    transforms_rest_inv = torch.inverse(transforms_rest)
    skinning_matrices = torch.matmul(transforms_posed, transforms_rest_inv)

    # 5. Apply the skinning matrices to the vertices
    homo_vertices = torch.cat([vertices, torch.ones(vertices.shape[0], 1, device=device)], dim=1)
    blended_transforms = torch.einsum('vj,jmn->vmn', weights, skinning_matrices)
    deformed_vertices_homo = torch.einsum('vmn,vn->vm', blended_transforms, homo_vertices)

    # 6. Correct the coordinate system (if necessary)
    # Swap Y and Z axes and negate Z to match Blender's convention
    deformed_vertices = deformed_vertices_homo[:, :3]
    corrected_vertices = torch.stack([
        deformed_vertices[:, 0],  # X remains the same
        deformed_vertices[:, 2],  # Z becomes Y
        -deformed_vertices[:, 1]  # Y becomes -Z
    ], dim=1)

    return corrected_vertices

def get_rest_joint_locations(bvh_obj, scale=1.0):
    """
    Calculates the global joint positions for the rest pose from the BVH skeleton.
    """
    _, _, parents, offsets, _, _ = bvh_obj.get_data()

    # Set root to -1
    parents[0] = -1

    # Debug
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
        else:
            j_rest[i] = j_rest[parents[i]] + offsets[i]
    return j_rest, parents

def main():
    # ----------------------------------------
    # 1. Configure paths
    # ----------------------------------------

    # ----------------
    # A. BVH file

    SCALE = 1
    bvh_path = "C:/Users/ealvarad/00-Local/02-Python/Blender/data/new_bvh_corrected/shot_001.bvh"

    # ----------------

    # ----------------
    # B. Static Meshes

    skin_static_layer_path = "C:/Users/ealvarad/00-Local/02-Python/Blender/data/layers/tpose/skin_layer_tpose_norot.obj"

    # ----------------

    # ----------------
    # C. LBS

    skin_weights_json_path = "C:/Users/ealvarad/00-Local/02-Python/Blender/data/weights/canonical_model/lbs_skin/skin_lbs_weights_exported_tpose.json"

    # ----------------

    # ----------------------------------------
    # 2. Prepare data
    # ----------------------------------------
    
    # ----------------
    # A. BVH file

    bvh = BVH()
    bvh.load(bvh_path)

    # Convert joint names to standard Python strings
    joint_names = [str(name) for name in bvh.data['names']]
    num_joints = len(joint_names)

    # Print bones with indexes
    print("------------------------")
    for i, name in enumerate(joint_names):
        print(f"[BVH] Preparing BVH with Joint {i}: {name}")
    print(f"[BVH] Loaded BVH with {num_joints} joints.")


    # Get the rest joint locations and convert to torch tensors
    j_rest, parents = get_rest_joint_locations(bvh, scale=SCALE)
    j_rest_tensor = torch.from_numpy(j_rest).float() # (24, 3)
    parents_tensor = torch.from_numpy(parents).long() # (24,)

    # Extract motion data
    local_rotations, local_positions, parents, offsets, end_sites, end_sites_parents = bvh.get_data()
    
    global_positions = local_positions[:, 0, :]  # root joint

    # Scale the skeleton
    local_positions[:, 0, :] *= SCALE # Same way we scale the offsets for the rest pose, we need to scale the root position

    num_frames = local_rotations.shape[0]

    print(f"[BVH] Loaded BVH with {num_frames} frames.")
    print("------------------------")

    # ----------------

    # ----------------
    # B. Static Meshes

    try:
        skin_layer = trimesh.load(skin_static_layer_path, process=False)
        vertices = skin_layer.vertices.tolist()

        if not isinstance(skin_layer, trimesh.Trimesh):
            raise ValueError(f"Loaded object is not a valid Trimesh: {type(skin_layer)}")
        
        # Apply a -90-degree rotation in the X-axis
        angle_rad = np.deg2rad(-90.0)  # Convert -90 degrees to radians
        rotation_matrix_x = np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad), 0],
            [0, np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 0, 1]
        ])
        skin_layer.apply_transform(rotation_matrix_x)

        print("------------------------")
        print(f"[MESH] Loaded mesh '{os.path.basename(skin_static_layer_path)}' successfully.")
    except Exception as e:
        print(f"Error loading mesh file: {e}")
        return
    
    s_bind_vertices_np = skin_layer.vertices.astype(np.float32)
    s_bind_vertices = torch.from_numpy(s_bind_vertices_np).float()
    s_bind_num_vertices = len(s_bind_vertices)

    print(f"[MESH] Loaded skin mesh with {s_bind_num_vertices} vertices and {len(skin_layer.faces)} faces.")
    print("------------------------")

    # ----------------

    # ----------------
    # E. LBS

    print(f"[LBS] Loaded skin mesh with {s_bind_num_vertices} vertices and {len(skin_layer.faces)} faces.")

    skin_weights_tensor = load_weights(skin_weights_json_path, len(s_bind_vertices), num_joints)

    # print(f"[LBS] Loaded skin_weights matrix with shape: {skin_weights.shape}") # (23752, 24)

    # ----------------

    # ----------------------------------------
    # 3. Visualization
    # ----------------------------------------
    
    server = viser.ViserServer()
    
    frame_slider = server.gui.add_slider("Frame", min=0, max=num_frames - 2, step=1, initial_value=0)
    bvh_bool = server.gui.add_checkbox("BVH Information", initial_value=False)
    lbs_bool = server.gui.add_checkbox("LBS", initial_value=False)

    # ----------------
    # A. BVH file

    # Add the static skeleton 
    bone_points = []
    for i, p_idx in enumerate(parents):
        if p_idx != -1:
            # Each bone is a pair of [start_point, end_point]
            bone_points.append([j_rest[p_idx], j_rest[i]])
    
    # Convert to a single NumPy array of shape (num_bones, 2, 3)
    bone_points = np.array(bone_points)
    
    server.scene.add_line_segments(
        name="/theta/bones",
        points=bone_points,
        line_width=3.0,
        colors=(0, 255, 0), # Green
        wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
    )

    # Update the joint positions (spheres)
    server.scene.add_point_cloud(
        name="/theta/joints",
        points=j_rest,
        colors=(255, 255, 0), # Yellow
        point_size=0.015,
        wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
    )

    # ----------------

    # ----------------
    # B. Static Meshes

    # Add the static mesh to the scene
    server.scene.add_mesh_trimesh(
        name="/s_bind",
        mesh=skin_layer,
        wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
    )

    # ----------------

    # ----------------------------------------
    # 4. Animation
    # ----------------------------------------

    print("\nOpen your browser to http://localhost:8080")
    print("Press Ctrl+C in the terminal to exit.")

    # --- 6. MAIN VISUALIZATION LOOP ---
    while True:
        # Get the current frame index and scale factor from the sliders
        current_frame_idx = frame_slider.value
        
        # ----------------
        # A. BVH file

        if bvh_bool.value == True:
            # 1. Calculate the skeleton's pose relative to its own origin (0,0,0)
            posed_joints_local, _ = fk(local_rotations[current_frame_idx], np.zeros(3), offsets, parents)
            posed_joints_world = posed_joints_local + global_positions[current_frame_idx, :]

            # Add the dynamic skeleton 
            bone_points = []
            for i, p_idx in enumerate(parents):
                if p_idx != -1:
                    # Each bone is a pair of [start_point, end_point]
                    bone_points.append([posed_joints_world[p_idx], posed_joints_world[i]])
            
            # Convert to a single NumPy array of shape (num_bones, 2, 3)
            bone_points = np.array(bone_points)
            
            server.scene.add_line_segments(
                name="/theta/bones",
                points=bone_points,
                line_width=3.0,
                colors=(255, 0, 0), # Red
                wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
            )

            # Update the joint positions (spheres)
            server.scene.add_point_cloud(
                name="/theta/joints",
                points=posed_joints_world,
                colors=(255, 255, 0), # Yellow
                point_size=0.015,
                wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
            )
            
        # # ----------------

        # ----------------
        # E. LBS

        if lbs_bool.value == True:
            root_position_tensor = torch.from_numpy(local_positions[current_frame_idx, 0, :]).float()
            pose_rotations_tensor = torch.from_numpy(local_rotations[current_frame_idx]).float()

            with torch.no_grad():
                deformed_vertices = lbs(s_bind_vertices, pose_rotations_tensor, skin_weights_tensor, j_rest_tensor, parents_tensor, root_position_tensor)

            deformed_skin_mesh = trimesh.Trimesh(
                vertices=deformed_vertices.cpu().numpy(),
                faces=skin_layer.faces
            )

            server.scene.add_mesh_trimesh(
                name="/lbs(skin,theta)",
                mesh=deformed_skin_mesh,
                wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
            )
            
        # # ----------------

        time.sleep(0.01)

    
if __name__ == "__main__":
    main()
