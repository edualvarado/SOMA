# -----------------------------------------------------------------------------
# FIX: REMOVE ROTATION OFFSET (BONE ROLL) - Pymotion Compatible
# -----------------------------------------------------------------------------
from scipy.spatial.transform import Rotation as R

# 1. Configuration
target_bone_name = "RightUpperLeg" 
bad_frame_idx = 1 # Frame where the "flip" happens relative to frame 0

# 2. Get Data from Pymotion BVH
# bvh_6d is the object you loaded in your script
joint_names = [str(n) for n in bvh_6d.data['names']] 

try:
    bone_idx = joint_names.index(target_bone_name)
    print(f"[FIX] Found '{target_bone_name}' at index {bone_idx}.")
except ValueError:
    print(f"[FIX] Error: Bone '{target_bone_name}' not found.")
    bone_idx = -1

if bone_idx != -1:
    # 3. Extract Rotations and Positions
    # pymotion returns: (rotations, positions, parents, offsets, names, frametime)
    # rotations shape: (Frames, Joints, 4) -> [w, x, y, z]
    rots, pos, _, _, _, _ = bvh_6d.get_data()
    
    # We work on a copy to be safe
    rots_fixed = rots.copy()

    # 4. Helper: Quaternion Conversion
    # Pymotion uses [w, x, y, z]. Scipy uses [x, y, z, w].
    def to_scipy(q): return np.roll(q, -1) # wxyz -> xyzw
    def to_pymotion(q): return np.roll(q, 1) # xyzw -> wxyz

    # 5. Calculate the Error Offset
    # We assume Frame 0 is correct and Frame 1 (or bad_frame_idx) introduces the artificial rotation.
    q_good = to_scipy(rots[0, bone_idx])
    q_bad  = to_scipy(rots[bad_frame_idx, bone_idx])
    
    r_good = R.from_quat(q_good)
    r_bad  = R.from_quat(q_bad)

    # Calculate the difference: "What rotation happened to get from Good to Bad?"
    # R_bad = R_diff * R_good  =>  R_diff = R_bad * R_good^-1
    r_diff = r_bad * r_good.inv()
    
    # The correction is the inverse of that difference
    r_fix = r_diff.inv()

    print(f"[FIX] Applying rotational correction to frames {bad_frame_idx} to end...")

    # 6. Apply Correction to affected frames
    for t in range(bad_frame_idx, rots.shape[0]):
        current_q = to_scipy(rots[t, bone_idx])
        r_curr = R.from_quat(current_q)
        
        # Apply fix: New = Fix * Old
        # Note: If the error is a local axis roll, we multiply on the LEFT (global/parent space relative to bone) 
        # or RIGHT (local space). Usually, gimbal/roll errors are local, so we try:
        # r_new = r_curr * r_fix  <-- Try this if the limb moves wildly
        # r_new = r_fix * r_curr  <-- Try this if the error is an offset (most likely)
        
        r_new = r_fix * r_curr
        
        rots_fixed[t, bone_idx] = to_pymotion(r_new.as_quat())

    # 7. Update the BVH object with fixed data
    # Pymotion's set_data expects (rotations, positions)
    bvh_6d.set_data(rots_fixed, pos)
    
    # Update your local variable if you use it later for plotting
    local_rotations_6d = rots_fixed 
    
    print("[FIX] Bone rotation fixed and data updated.")