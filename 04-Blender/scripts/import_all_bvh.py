"""
Script: import_all_bvh.py
Goal: Import (batch or single) BVH files, re-center armature origins, and correct animation offsets.
"""

import bpy
import os
from mathutils import Vector

# --- User Configuration ---
# SHOT_NAMES = ["Shot_020"]
SHOT_NAMES = ["Shot_015", "Shot_016", "Shot_017", "Shot_018", "Shot_019", "Shot_020"]

BASE_BVH_DIR = "C:/Users/ealvarad/00-Local/02-Python/Blender/data/new_skeleton"
BVH_IMPORT_SCALE = 0.001
IMPORTED_BVH_BASE_NAME = "unknown"
# --------------------------

def apply_animation_offset(armature_obj, offset_vector):
    """
    Applies the calculated origin offset to the Hips bone's animation curves,
    using the specified Y/Z axis swap to correct for coordinate system differences.
    """
    print(f"Applying custom animation offset to 'Hips' bone...")

    if not armature_obj.animation_data or not armature_obj.animation_data.action:
        print("Warning: No animation data or action found to correct.")
        return

    action = armature_obj.animation_data.action

    # --- NEW: Direct Axis Mapping based on your findings ---
    # The X-axis of the offset applies to the X-curve.
    # The Z-axis of the offset applies to the Y-curve.
    # The Y-axis of the offset applies to the Z-curve.
    x_location_offset = offset_vector.x
    y_location_offset = offset_vector.z
    z_location_offset = -offset_vector.y

    print(f"-> Captured World Offset: {offset_vector.to_tuple(4)}")
    print(f"-> Applying to F-Curves: X+={x_location_offset:.4f}, Y+={y_location_offset:.4f}, Z+={z_location_offset:.4f}")

    # --- Apply the corresponding offsets to each curve ---
    corrected_curves = 0
    for fcurve in action.fcurves:
        # We target the location property of the "Hips" bone specifically
        if fcurve.data_path == 'pose.bones["Hips"].location':

            offset_value = 0.0

            # array_index 0 is X, 1 is Y, 2 is Z
            if fcurve.array_index == 0:  # X Location curve
                offset_value = x_location_offset
            elif fcurve.array_index == 1:  # Y Location curve
                offset_value = y_location_offset
            elif fcurve.array_index == 2:  # Z Location curve
                offset_value = z_location_offset

            # Only apply if there's a significant offset to avoid unnecessary updates
            if abs(offset_value) > 1e-6:
                # Add the calculated offset to the value of every keyframe point
                for keyframe in fcurve.keyframe_points:
                    keyframe.co[1] += offset_value
                    keyframe.handle_left[1] += offset_value
                    keyframe.handle_right[1] += offset_value
                corrected_curves += 1

    if corrected_curves > 0:
        print("-> Animation curves corrected.")
    else:
        print("-> No significant offsets needed for Hips.location curves.")

def recenter_armature_origin(armature_obj):
    """
    Changes the armature's origin, moves it to world 0,0,0, and corrects the animation.
    """
    print(f"Attempting to re-center origin for '{armature_obj.name}'...")

    bones_to_center_on = ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"]
    original_cursor_location = bpy.context.scene.cursor.location.copy()

    if armature_obj.mode != 'OBJECT': bpy.ops.object.mode_set(mode='OBJECT')

    armature_matrix_world = armature_obj.matrix_world

    joint_locations = []
    for bone_name in bones_to_center_on:
        bone = armature_obj.data.bones.get(bone_name)
        if bone:
            joint_locations.append(armature_matrix_world @ bone.head_local)
            joint_locations.append(armature_matrix_world @ bone.tail_local)
        else:
            print(f"Warning: Bone '{bone_name}' not found.")

    if not joint_locations:
        print("ERROR: Could not find bones to re-center. Aborting re-center.");
        return

    # Calculate center and move cursor
    center_point = sum(joint_locations, Vector()) / len(joint_locations)
    bpy.context.scene.cursor.location = center_point

    # Set the object's origin to the 3D cursor
    previous_active_obj = bpy.context.view_layer.objects.active
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')

    # 1. Before moving the object, copy the offset its location now represents.
    offset_to_correct = armature_obj.location.copy()
    print(f"       Captured origin offset: {offset_to_correct.to_tuple(4)}")

    # 2. Move the object's origin to the world center.
    armature_obj.location = (0.0, 0.0, 0.0)
    print(f"       Moved '{armature_obj.name}' to world origin.")

    # 3. Apply the captured offset to the animation curves.
    apply_animation_offset(armature_obj, offset_to_correct)

    bpy.context.scene.cursor.location = original_cursor_location
    bpy.context.view_layer.objects.active = previous_active_obj
    print(f"       Successfully re-centered origin and corrected animation.")


def setup_bvh_imports(shot_list):
    """
    Main function to create collections and import/configure BVH files.
    """
    print("--- Importing and Configuring Studio BVH files ---")
    # ... (The rest of this function is the same as the last version) ...
    scene_collection = bpy.context.scene.collection
    for shot_name in shot_list:
        if shot_name in bpy.data.collections:
            parent_collection = bpy.data.collections[shot_name]
        else:
            parent_collection = bpy.data.collections.new(shot_name)
            scene_collection.children.link(parent_collection)
            print(f"  Created parent collection: '{shot_name}'")
        try:
            shot_number = shot_name.split('_')[-1]
            child_collection_name = f"StudioBVH_{shot_number}"
            if child_collection_name not in parent_collection.children:
                child_collection = bpy.data.collections.new(child_collection_name)
                parent_collection.children.link(child_collection)
                print(f"    -> Created child collection: '{child_collection_name}'")
            else:
                child_collection = parent_collection.children[child_collection_name]
        except IndexError:
            print(f"    Warning: Could not parse number from '{shot_name}'."); continue

        bvh_folder_name = f"shot_{shot_number}_captury"
        bvh_folder_path = os.path.join(BASE_BVH_DIR, bvh_folder_name)
        bvh_file_path = next(
            (os.path.join(bvh_folder_path, f) for f in os.listdir(bvh_folder_path) if f.lower().endswith('.bvh')),
            None) if os.path.isdir(bvh_folder_path) else None

        if not bvh_file_path: print(f"    Warning: No .bvh file found in {bvh_folder_path}"); continue
        new_armature_name = f"{IMPORTED_BVH_BASE_NAME}_{shot_number}"

        if new_armature_name in child_collection.objects: print(
            f"-> Armature '{new_armature_name}' already exists. Skipping import."); continue
        print(f"-> Importing BVH from: {bvh_file_path}")

        try:
            layer_collection = bpy.context.view_layer.layer_collection
            active_child_collection = layer_collection.children[parent_collection.name].children[child_collection.name]
            bpy.context.view_layer.active_layer_collection = active_child_collection
            bpy.ops.import_anim.bvh(filepath=bvh_file_path, global_scale=BVH_IMPORT_SCALE, use_fps_scale=False,
                                    update_scene_duration=False)
            imported_obj = bpy.context.active_object
            if imported_obj and imported_obj.type == 'ARMATURE':
                imported_obj.name = new_armature_name
                print(f"Renamed imported armature to '{imported_obj.name}'")
                recenter_armature_origin(imported_obj)
                imported_obj.hide_set(True)
                print(f"Set '{imported_obj.name}' to be invisible.")
        except Exception as e:
            print(f"ERROR: Failed to import BVH file '{bvh_file_path}'. Error: {e}")

    print(f"\nSetup complete.")


# --- Main execution block ---
if __name__ == "__main__":
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    setup_bvh_imports(SHOT_NAMES)