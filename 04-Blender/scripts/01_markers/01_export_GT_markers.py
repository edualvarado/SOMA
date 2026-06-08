"""
STATUS: Fixed (Streaming) - Exports corrected markers incrementally to prevent Memory Errors and Truncated JSONs.
Script: export_GT_markers_stream.py
Goal: Reads input JSON, applies transforms, and writes to disk frame-by-frame to handle large datasets (1GB+).
"""

import bpy
import json
import math
from mathutils import Vector, Matrix
import os
import gc # Garbage collector

# --- User Configuration ---
shots = ["shot_001"]  

# Rotation and translation corrections
ROT_X_DEG = 90.0
ROT_Y_DEG = 0.0
ROT_Z_DEG = 180.0

TRANS_X = 0
TRANS_Y = 0
TRANS_Z = 0

def apply_transformations(position):
    """ Apply rotation and translation to a list [x, y, z]. """
    rot_x_rad = math.radians(ROT_X_DEG)
    rot_y_rad = math.radians(ROT_Y_DEG)
    rot_z_rad = math.radians(ROT_Z_DEG)

    rot_x_matrix = Matrix.Rotation(rot_x_rad, 4, 'X')
    rot_y_matrix = Matrix.Rotation(rot_y_rad, 4, 'Y')
    rot_z_matrix = Matrix.Rotation(rot_z_rad, 4, 'Z')

    rotation_matrix = rot_z_matrix @ rot_y_matrix @ rot_x_matrix
    transformed_position = rotation_matrix @ Vector(position)
    transformed_position += Vector((TRANS_X, TRANS_Y, TRANS_Z))

    return [transformed_position.x, transformed_position.y, transformed_position.z]

def process_frame_data(frame_data):
    """ Transforms all markers in a single frame dictionary. """
    corrected_markers = {}
    for marker_key, marker_data in frame_data.items():
        original_position = marker_data[0] 
        corrected_position = apply_transformations(original_position)
        corrected_markers[marker_key] = [corrected_position]
    return corrected_markers

def export_corrected_markers_streaming(shot):
    # INPUT_JSON_PATH = f"S:/work/03-MUSK/04-Blender/data/registration/{shot}/triangulated_sequence_{shot}.json"
    INPUT_JSON_PATH = f"S:/work/03-MUSK/04-Blender/data/registration/S4/{shot}/S4_triangulated_sequence_{shot}.json"

    # OUTPUT_JSON_PATH = f"S:/work/03-MUSK/04-Blender/data/registration/{shot}/triangulated_sequence_{shot}_transformed.json"
    OUTPUT_JSON_PATH = f"S:/work/03-MUSK/04-Blender/data/registration/S4/{shot}/S4_triangulated_sequence_{shot}_transformed.json"

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    print(f"--- Processing {shot} (Streaming Mode) ---")

    # 1. Load Input (We still load input to memory, but we won't duplicate it)
    try:
        print("Loading input JSON...")
        with open(INPUT_JSON_PATH, 'r') as f:
            input_data = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load {shot}. {e}")
        return

    # Sort input frames numerically to ensure order
    # Filter out non-numeric keys if any exist
    sorted_input_keys = sorted([k for k in input_data.keys() if k.isdigit()], key=int)
    print(f"Loaded {len(sorted_input_keys)} frames.")

    # 2. Open Output File for Writing
    try:
        with open(OUTPUT_JSON_PATH, 'w') as f_out:
            f_out.write('{\n') # Start JSON object

            # --- Handle Special "Frame 0" Logic ---
            # Your logic: Frame "0" in output is a copy of Input "0" (which becomes Frame "1")
            # Effectively: Output "0" = Transform(Input "0")
            
            first_input_frame_key = sorted_input_keys[0] # Usually "0"
            first_frame_data = input_data[first_input_frame_key]
            
            # Process Frame 0
            frame_0_markers = process_frame_data(first_frame_data)
            
            # Write Frame 0
            f_out.write(f'  "0": {json.dumps(frame_0_markers)}')
            f_out.write(',\n') # Comma because more frames follow

            # --- Handle Remaining Frames ---
            # Your logic: Input "0" becomes Output "1", Input "1" becomes Output "2", etc.
            
            total_frames = len(sorted_input_keys)
            
            for i, input_key in enumerate(sorted_input_keys):
                # Calculate new frame index (Input + 1)
                new_frame_idx = str(int(input_key) + 1)
                
                # Process data
                frame_markers = process_frame_data(input_data[input_key])
                
                # Write to file immediately
                f_out.write(f'  "{new_frame_idx}": {json.dumps(frame_markers)}')
                
                # Add comma if this is NOT the last frame
                if i < total_frames - 1:
                    f_out.write(',\n')
                else:
                    f_out.write('\n') # Last frame, no comma
                
                # Periodic print to show it's alive
                if i % 100 == 0:
                    print(f"Written frame {new_frame_idx}/{total_frames}...", end='\r')

            f_out.write('}') # Close JSON object
            
        print(f"\nSUCCESS: Saved fully to {OUTPUT_JSON_PATH}")

    except Exception as e:
        print(f"\nCRITICAL ERROR during write: {e}")

    # 3. Clean up memory
    del input_data
    gc.collect()

if __name__ == "__main__":
    for shot in shots:
        export_corrected_markers_streaming(shot)

# ---

# """
# STATUS: Completed - Create transform to fix rotation and translation of GT markers.
# Script: export_GT_markers.py
# Goal: Export the triangulated marker data after applying rotation and translation corrections.
# """

# import bpy
# import json
# import math
# from mathutils import Vector, Matrix
# import os

# # --- User Configuration ---

# # shots = ["shot_001", "shot_002", "shot_003", "shot_004", "shot_005", "shot_006", "shot_007", "shot_008", "shot_009", "shot_010", "shot_011", "shot_012", "shot_013", "shot_014", "shot_015", "shot_016", "shot_017", "shot_018", "shot_019", "shot_020"]  # Add your shot names here
# shots = ["shot_001"]  # Add your shot names here

# # Rotation and translation corrections
# ROT_X_DEG = 90.0  # Rotation around X-axis in degrees
# ROT_Y_DEG = 0.0  # Rotation around Y-axis in degrees
# ROT_Z_DEG = 180.0 # Rotation around Z-axis in degrees

# TRANS_X = 0  # Translation along X-axis
# TRANS_Y = 0  # Translation along Y-axis
# TRANS_Z = 0  # Translation along Z-axis

# def apply_transformations(position):
#     """
#     Apply rotation and translation transformations to a 3D position.
#     """
#     # Convert degrees to radians
#     rot_x_rad = math.radians(ROT_X_DEG)
#     rot_y_rad = math.radians(ROT_Y_DEG)
#     rot_z_rad = math.radians(ROT_Z_DEG)

#     # Create rotation matrices
#     rot_x_matrix = Matrix.Rotation(rot_x_rad, 4, 'X')
#     rot_y_matrix = Matrix.Rotation(rot_y_rad, 4, 'Y')
#     rot_z_matrix = Matrix.Rotation(rot_z_rad, 4, 'Z')

#     # Combine rotations (Z * Y * X order)
#     rotation_matrix = rot_z_matrix @ rot_y_matrix @ rot_x_matrix

#     # Apply rotation
#     transformed_position = rotation_matrix @ Vector(position)

#     # Apply translation
#     transformed_position += Vector((TRANS_X, TRANS_Y, TRANS_Z))

#     return transformed_position

# def export_corrected_markers_for_shot(shot):
#     """
#     Export the corrected marker positions for a specific shot to a new JSON file.
#     """

#     # INPUT_JSON_PATH = f"S:/work/03-MUSK/04-Blender/data/registration/{shot}/triangulated_sequence_{shot}.json"
#     INPUT_JSON_PATH = f"S:/work/03-MUSK/04-Blender/data/registration/S5/{shot}/triangulated_sequence_{shot}.json"

#     # OUTPUT_JSON_PATH = f"S:/work/03-MUSK/04-Blender/data/registration/{shot}/triangulated_sequence_{shot}_transformed.json"
#     OUTPUT_JSON_PATH = f"S:/work/03-MUSK/04-Blender/data/registration/S5/{shot}/triangulated_sequence_{shot}_transformed.json"

#     # Ensure the output directory exists
#     os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

#     print(f"--- Processing {shot} ---")
#     print("--- Step 1: Loading Input Data ---")
#     try:
#         with open(INPUT_JSON_PATH, 'r') as f:
#             input_data = json.load(f)
#     except Exception as e:
#         print(f"ERROR: Failed to load input JSON file for {shot}. {e}")
#         return

#     print(f"Loaded marker data for {len(input_data)} frames.")

#     # --- Step 2: Apply Transformations ---
#     corrected_data = {}

#     # Create a new frame at the beginning by copying data from frame 0
#     all_frames_posed_data = {str(int(frame) + 1): data for frame, data in input_data.items()}
#     all_frames_posed_data["0"] = all_frames_posed_data.get("1", {})

#     for frame in all_frames_posed_data:
#         corrected_markers = {}
#         for marker_key, marker_data in all_frames_posed_data[frame].items(): 
#             # Apply transformations to the marker position
#             original_position = marker_data[0]  # Assuming format [[X, Y, Z]]
#             corrected_position = apply_transformations(original_position)
#             corrected_markers[marker_key] = [corrected_position[:]]  # Convert Vector back to list

#         corrected_data[frame] = corrected_markers

#     print("\nFinished applying transformations to all frames.")

#     # --- Step 3: Save Corrected Data ---
#     print(f"Saving corrected marker data to: {OUTPUT_JSON_PATH}")
#     try:
#         with open(OUTPUT_JSON_PATH, 'w') as f:
#             json.dump(corrected_data, f, indent=2)
#         print("Corrected marker data saved successfully.")
#     except Exception as e:
#         print(f"ERROR: Failed to save corrected JSON file for {shot}. {e}")

# if __name__ == "__main__":
#     for shot in shots:
#         export_corrected_markers_for_shot(shot)