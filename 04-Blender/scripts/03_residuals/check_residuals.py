import json
from mathutils import Vector, Matrix
import bpy

# File paths
CANONICAL_MARKERS_JSON_PATH = "C:/Users/ealvarad/00-Local/02-Python/Blender/data/registration/canonical_model/canonical_data.json"
MARKER_LBS_WEIGHTS_JSON_PATH = "C:/Users/ealvarad/00-Local/02-Python/Blender/data/weights/canonical_model/lbs_markers/markers_lbs_weights_exported.json"
EXPORTED_LBS_MARKERS_JSON_PATH = "C:/Users/ealvarad/00-Local/02-Python/Blender/data/displacements/shot_001/canonical/canonical_markers_lbs_shot_001_exported.json"

# Load canonical data
with open(CANONICAL_MARKERS_JSON_PATH, 'r') as f:
    canonical_data = json.load(f).get("0", {})

# Load LBS weights
with open(MARKER_LBS_WEIGHTS_JSON_PATH, 'r') as f:
    lbs_weights = json.load(f)

# Load exported LBS marker data
with open(EXPORTED_LBS_MARKERS_JSON_PATH, 'r') as f:
    exported_lbs_data = json.load(f)

# Placeholder for armature object (replace with actual armature object in Blender)
armature_obj = bpy.data.objects.get("root_001")  # Replace "root_001" with your armature name
if not armature_obj or armature_obj.type != 'ARMATURE':
    raise ValueError("Armature object not found or is not of type 'ARMATURE'.")

# Get bind pose and inverse bind matrices
bind_pose_matrices = {b.name: armature_obj.matrix_world @ b.matrix_local for b in armature_obj.data.bones}
inverse_bind_matrices = {name: mat.inverted() for name, mat in bind_pose_matrices.items()}

# Compare data
for frame, markers in exported_lbs_data.items():
    # Get posed bone matrices for the current frame
    bpy.context.scene.frame_set(int(frame))
    posed_bone_matrices = {bone.name: bone.matrix for bone in armature_obj.pose.bones}

    for marker_key, exported_position in markers.items():
        # Get canonical position
        canonical_position = Vector(canonical_data[marker_key][0])

        # Apply LBS weights to compute the LBS position
        weights_info = lbs_weights.get(marker_key)
        if not weights_info or not weights_info.get("bone_indices"):
            continue  # Skip if no weights

        p_unposed_homogeneous = canonical_position.to_4d()
        blended_transform = Matrix.Identity(4)
        blended_transform.zero()
        for i, bone_idx in enumerate(weights_info["bone_indices"]):
            weight = weights_info["weights"][i]
            bone_name = armature_obj.data.bones[int(bone_idx)].name
            posed_bone_matrix = posed_bone_matrices[bone_name]
            inverse_bind_matrix = inverse_bind_matrices[bone_name]
            skinning_matrix = posed_bone_matrix @ inverse_bind_matrix
            blended_transform += weight * skinning_matrix

        # Compute the LBS-deformed position in local space
        lbs_position = (blended_transform @ p_unposed_homogeneous).to_3d()

        # Transform the LBS-deformed position into world space
        lbs_position_world = armature_obj.matrix_world @ lbs_position

        # Compare with exported position
        if not all(abs(exported_position[0][i] - lbs_position_world[i]) <= 1e-6 for i in range(3)):
            print(f"Mismatch for marker {marker_key} at frame {frame}:")
            print(f"  Computed LBS position (world): {lbs_position_world}")
            print(f"  Exported position: {exported_position[0]}")