"""
Script: check_bone_constraints.py
Goal: Check that all bone constraints in an armature are visible and enabled.
       If a constraint is muted, it will be automatically unmuted.
"""

import bpy

# --- User Configuration ---
# !!! IMPORTANT: SET THIS NAME TO MATCH YOUR SCENE !!!
ARMATURE_NAME = "root_019"  # The exact name of your armature object

# --------------------------

def check_and_unmute_bone_constraints(armature_name):
    """
    Checks all bone constraints in the specified armature to ensure they are visible and enabled.
    If a constraint is muted, it will be unmuted.
    """
    print(f"--- Starting Constraint Check and Unmute ---")

    # 1. Get the armature object from the scene
    armature_obj = bpy.data.objects.get(armature_name)

    # 2. Safety check to ensure the armature exists
    if not armature_obj or armature_obj.type != 'ARMATURE':
        print(f"Error: Could not find an armature named '{armature_name}'. Aborting.")
        return

    print(f"Found Armature: '{armature_obj.name}'")

    # 3. Iterate through all pose bones and their constraints
    issues_found = False
    for pose_bone in armature_obj.pose.bones:
        for constraint in pose_bone.constraints:
            # Check if the constraint is visible and enabled
            if not constraint.mute and constraint.show_expanded:
                print(f"Constraint '{constraint.name}' on bone '{pose_bone.name}' is visible and enabled.")
            else:
                issues_found = True
                print(f"WARNING: Constraint '{constraint.name}' on bone '{pose_bone.name}' is not properly configured.")
                if constraint.mute:
                    print(f"  - Constraint is muted. Unmuting it now...")
                    constraint.mute = False  # Unmute the constraint
                if not constraint.show_expanded:
                    print(f"  - Constraint is not visible (collapsed).")

    if not issues_found:
        print(f"\n--- Finished ---")
        print(f"All constraints in armature '{armature_name}' are visible and enabled.")
    else:
        print(f"\n--- Finished ---")
        print(f"Issues were found and resolved for some constraints in armature '{armature_name}'. Please review the warnings above.")

# --- Run the function ---
if __name__ == "__main__":
    check_and_unmute_bone_constraints(ARMATURE_NAME)