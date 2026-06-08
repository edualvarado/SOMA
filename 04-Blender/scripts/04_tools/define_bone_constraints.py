"""
Script: define_bone_constraints.py
Goal: Change bone constraints in an armature to a new target object.
"""

import bpy

# --- User Configuration ---
# !!! IMPORTANT: SET THESE NAMES TO MATCH YOUR SCENE !!!
ARMATURE_NAME = "root_001_S4_TPose"  # The exact name of your armature object
NEW_TARGET_NAME = "unknown_001_S4"  # The exact name of the new target object

# --------------------------

def change_all_bone_constraint_targets(armature_name, new_target_name):
    """
    Finds an armature and changes the target of all bone constraints to a new object.
    """
    print(f"--- Starting Constraint Target Update ---")

    # 1. Get the armature and new target objects from the scene
    armature_obj = bpy.data.objects.get(armature_name)
    new_target_obj = bpy.data.objects.get(new_target_name)

    # 2. Safety checks to ensure objects exist
    if not armature_obj or armature_obj.type != 'ARMATURE':
        print(f"Error: Could not find an armature named '{armature_name}'. Aborting.")
        return

    if not new_target_obj:
        print(f"Error: Could not find the new target object named '{new_target_name}'. Aborting.")
        return

    print(f"Found Armature: '{armature_obj.name}'")
    print(f"Found New Target: '{new_target_obj.name}'")

    # 3. Iterate through all pose bones and their constraints
    updated_constraints_count = 0
    # Constraints live on Pose Bones, so we iterate through `pose.bones`
    for pose_bone in armature_obj.pose.bones:
        for constraint in pose_bone.constraints:
            # Check if the constraint has a 'target' property before trying to change it
            if hasattr(constraint, "target"):
                # Only change it if it's not already correct
                if constraint.target != new_target_obj:
                    print(f"Updating constraint '{constraint.name}' on bone '{pose_bone.name}'...")
                    constraint.target = new_target_obj
                    updated_constraints_count += 1

    print(f"\n--- Finished ---")
    print(f"Updated {updated_constraints_count} constraints to target '{new_target_name}'.")


# --- Run the function ---
if __name__ == "__main__":
    change_all_bone_constraint_targets(ARMATURE_NAME, NEW_TARGET_NAME)