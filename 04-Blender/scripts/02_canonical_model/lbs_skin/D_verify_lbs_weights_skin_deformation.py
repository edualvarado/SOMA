import bpy
import json
from mathutils import Vector

# --- USER: SET THESE ---
SKINNED_OBJECT_NAME = "skin_test"  # Name of the skinned mesh
ARMATURE_NAME = "shot_002"        # Name of the armature
WEIGHTS_JSON_FILE = "C:/Users/ealvarad/00-Local/02-Python/Blender/data/weights/canonical_model/lbs_skin/skin_lbs_weights_exported_tpose.json"
# ------------------------

def load_weights(json_file):
    """Load vertex weights and data from JSON."""
    with open(json_file, 'r') as f:
        return json.load(f)

def get_bone_matrices(armature):
    """Get the transformation matrices of all bones in the armature, including constraints and parent transformations."""
    bone_matrices = {}

    # Evaluate the armature to include constraint effects
    depsgraph = bpy.context.evaluated_depsgraph_get()
    armature_eval = armature.evaluated_get(depsgraph)
    armature_matrix = armature_eval.matrix_world  # Armature's world matrix

    # Include parent transformations
    if armature.parent:
        parent_matrix = armature.parent.matrix_world
        armature_matrix = parent_matrix @ armature_matrix

    for bone in armature_eval.pose.bones:
        # Combine the armature's world matrix with the bone's pose and rest matrices
        rest_matrix = bone.bone.matrix_local  # Bone's rest matrix
        pose_matrix = bone.matrix  # Bone's pose matrix (relative to rest position)
        inverse_bind_matrix = rest_matrix.inverted()  # Inverse bind matrix

        # Compute the final bone matrix
        bone_matrix = armature_matrix @ pose_matrix @ inverse_bind_matrix

        # Debug: Print the bone matrix
        print(f"Bone: {bone.name}")
        print(f"Rest Matrix:\n{rest_matrix}")
        print(f"Pose Matrix:\n{pose_matrix}")
        print(f"Inverse Bind Matrix:\n{inverse_bind_matrix}")
        print(f"Final Bone Matrix:\n{bone_matrix}")

        bone_matrices[bone.name] = bone_matrix

    return bone_matrices

def apply_lbs(vertex_data, bone_matrices):
    """Apply Linear Blend Skinning (LBS) to deform the vertices."""
    deformed_vertices = []

    # Get the mesh's world matrix
    mesh_matrix_world = bpy.data.objects[SKINNED_OBJECT_NAME].matrix_world

    for v_idx, vertex in enumerate(vertex_data):
        # Transform the vertex position to global space
        original_pos_local = Vector(vertex["coords_local"])
        original_pos = mesh_matrix_world @ original_pos_local  # Transform to global space

        bone_indices = vertex["bone_indices"]           # Bone indices influencing this vertex
        weights = vertex["weights"]                     # Corresponding weights
        weight_sum = sum(weights)
        if abs(weight_sum - 1.0) > 1e-6:  # Allow for small floating-point errors
            print(f"WARNING: Weights for vertex {v_idx} do not sum to 1.0 (sum = {weight_sum})")
            weights = [w / weight_sum for w in weights]  # Normalize weights

        deformed_pos = Vector((0.0, 0.0, 0.0))  # Initialize deformed position

        for bone_idx, weight in zip(bone_indices, weights):
            bone_name = list(bone_matrices.keys())[bone_idx]  # Get bone name from index
            bone_matrix = bone_matrices[bone_name]            # Get bone transformation matrix

            # Transform the vertex position by the bone matrix
            transformed_pos = bone_matrix @ original_pos  # Perform matrix multiplication
            deformed_pos += weight * transformed_pos  # Apply weight and accumulate

        # Debug: Test different axis corrections
        # corrected_pos = Vector((deformed_pos.x, deformed_pos.z, -deformed_pos.y))  # Original correction
        # corrected_pos = Vector((deformed_pos.x, -deformed_pos.z, deformed_pos.y))  # Alternative correction
        corrected_pos = Vector((deformed_pos.x, deformed_pos.y, deformed_pos.z))   # No correction
        # corrected_pos = Vector((deformed_pos.x, -deformed_pos.y, deformed_pos.z))  # Another alternative

        # Transform the deformed position back to local space
        deformed_pos_local = mesh_matrix_world.inverted() @ corrected_pos
        deformed_vertices.append(deformed_pos_local[:])  # Convert Vector back to a list

        # Debug: Print original and deformed positions
        if v_idx < 10:  # Limit debug output to the first 10 vertices
            print(f"Vertex {v_idx}:")
            print(f"  Original (local): {original_pos_local}")
            print(f"  Original (global): {original_pos}")
            print(f"  Deformed (global): {deformed_pos}")
            print(f"  Corrected (global): {corrected_pos}")
            print(f"  Deformed (local): {deformed_pos_local}")

    return deformed_vertices

def deform_mesh():
    # Get the skinned object and armature
    obj = bpy.data.objects.get(SKINNED_OBJECT_NAME)
    armature = bpy.data.objects.get(ARMATURE_NAME)

    if not obj or not armature:
        print(f"ERROR: Could not find object '{SKINNED_OBJECT_NAME}' or armature '{ARMATURE_NAME}'.")
        return

    # Load weights from JSON
    vertex_data = load_weights(WEIGHTS_JSON_FILE)

    # Get bone transformation matrices
    bone_matrices = get_bone_matrices(armature)

    # Apply LBS to deform the vertices
    deformed_vertices = apply_lbs(vertex_data, bone_matrices)

    # Update the mesh with the deformed vertex positions
    mesh = obj.data
    for v_idx, vertex in enumerate(mesh.vertices):
        vertex.co = deformed_vertices[v_idx]

    # Update the mesh in Blender
    mesh.update()
    print(f"Mesh '{SKINNED_OBJECT_NAME}' successfully deformed using LBS.")

    # Compare with Blender's deformation
    print("Comparing LBS results with Blender's deformation...")
    for v_idx, vertex in enumerate(obj.data.vertices):
        blender_deformed_pos = obj.matrix_world @ vertex.co  # Blender's deformed position
        custom_deformed_pos = deformed_vertices[v_idx]      # LBS-computed position

        if (v_idx == 15444):
            print(f"Vertex {v_idx}:")
            print(f"  Blender: {blender_deformed_pos}")
            print(f"  LBS:     {custom_deformed_pos}")

if __name__ == "__main__":
    deform_mesh()