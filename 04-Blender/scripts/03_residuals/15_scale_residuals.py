"""
STATUS: Fixed (Robust Paths) - Scales residuals for multiple subjects.
Script: 15_scale_residuals_robust.py
Goal: 
1. Uses absolute paths derived from the script location to find the 'data' folder.
2. Handles S3, S4, S5.
"""

import json
import os
import gc
from pathlib import Path

# --- User Configuration ---
subjects = ["S2", "S3", "S4", "S5"]
shots = ["shot_001"]
SCALE_FACTOR = 0.01 

# ----------------------------------------------------

def scale_residuals_streaming(subject, shot):
    print(f"--- Scaling Residuals for Subject: {subject} | Shot: {shot} ---")

    # --- 1. Robust Path Construction ---
    # Get the folder containing this script (scripts/03_residuals)
    script_dir = Path(__file__).resolve().parent
    
    # Go up 2 levels to get '04-Blender' (scripts/03_residuals/../.. -> 04-Blender)
    # Then go down into 'data/registration'
    base_data_dir = script_dir.parents[1] / "data" / "registration"
    
    # Final path: .../data/registration/S5/shot_001/residuals_shot_001_world_lbs_tpose.json
    subject_shot_dir = base_data_dir / subject / shot
    input_filename = f"{subject}_residuals_{shot}_world_lbs_tpose.json"
    output_filename = f"{subject}_residuals_{shot}_world_lbs_scaled_tpose.json"
    
    input_path = subject_shot_dir / input_filename
    output_path = subject_shot_dir / output_filename

    # --- 2. Validation ---
    if not input_path.exists():
        print(f"ERROR: Input file not found at: {input_path}")
        print(f"      (Checked relative to script at: {script_dir})")
        return

    print(f"Reading from: {input_path}")
    
    try:
        with open(input_path, 'r') as f:
            residuals_data = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load input file. {e}")
        return

    total_frames = len(residuals_data)
    sorted_frames = sorted([k for k in residuals_data.keys() if k.isdigit()], key=int)

    # --- 3. Process & Write ---
    print(f"Streaming scaled results to: {output_path}")
    
    try:
        with open(output_path, 'w') as f_out:
            f_out.write('{\n')

            for i, frame in enumerate(sorted_frames):
                markers = residuals_data[frame]
                scaled_markers = {}

                for marker_key, val in markers.items():
                    # Handle nested [[x,y,z]] vs flat [x,y,z]
                    if len(val) == 1 and isinstance(val[0], list):
                        coords = val[0]
                    else:
                        coords = val    
                    
                    # Scale
                    if len(coords) == 3:
                        scaled_coords = [c * SCALE_FACTOR for c in coords]
                        scaled_markers[marker_key] = [scaled_coords]
                    else:
                        scaled_markers[marker_key] = val

                json_str = json.dumps(scaled_markers)
                f_out.write(f'  "{frame}": {json_str}')

                if i < total_frames - 1:
                    f_out.write(',\n')
                else:
                    f_out.write('\n')
            
            f_out.write('}')

        print(f"SUCCESS: Saved scaled residuals for {subject}/{shot}")

    except Exception as e:
        print(f"CRITICAL ERROR during write: {e}")

    # Cleanup
    del residuals_data
    gc.collect()

if __name__ == "__main__":
    for subject in subjects:
        for shot in shots:
            scale_residuals_streaming(subject, shot)

# # --

# """
# Script: scale_residuals_batch.py
# Goal: Apply a scale factor to the residuals JSON for multiple shots and save new versions of the files.
# """

# import json
# import os
# from mathutils import Vector

# # --- User Configuration ---

# # List of shots to process
# # shots = [
# #     "shot_001", "shot_002", "shot_003", "shot_004", "shot_005",
# #     "shot_006", "shot_007", "shot_008", "shot_009", "shot_010",
# #     "shot_011", "shot_012", "shot_013", "shot_014", "shot_015",
# #     "shot_016", "shot_017", "shot_018", "shot_019", "shot_020"
# # ]

# shots = ["shot_001"]

# # Scale factor to apply to the residuals
# SCALE_FACTOR = 0.01

# # ----------------------------------------------------

# def scale_residuals(shot):
#     """Apply the scale factor to the residuals for a single shot and save the result."""

#     print(f"--- Scaling Residuals for {shot} ---")

#     # A) Input and output paths for the current shot (entire residuals)
#     # input_residuals_json_path = f"C:/Users/ealvarad/00-Local/02-Python/Blender/data/displacements/{shot}/residuals/residuals_{shot}_world_lbs_tpose.json"
#     input_residuals_json_path = f"../../data/registration/S2/{shot}/S2_residuals_{shot}_world_lbs_tpose.json"

#     # output_scaled_residuals_json_path = f"C:/Users/ealvarad/00-Local/02-Python/Blender/data/displacements/{shot}/residuals/residuals_{shot}_world_lbs_scaled_001_tpose.json"
#     output_scaled_residuals_json_path = f"../../data/registration/S2/{shot}/S2_residuals_{shot}_world_lbs_scaled_tpose.json"

#     # B1) Input and output paths for the current shot (separated residuals) <- TODO!
#     # input_residuals_json_path = f"C:/Users/ealvarad/00-Local/02-Python/Blender/data/displacements/{shot}/residuals/observed_residuals_only_{shot}_world_lbs.json"
#     # output_scaled_residuals_json_path = f"C:/Users/ealvarad/00-Local/02-Python/Blender/data/displacements/{shot}/residuals/observed_residuals_only_{shot}_world_lbs_scaled.json"
    
#     # B2) Input and output paths for the current shot (separated residuals) <- TODO!
#     # input_residuals_json_path = f"C:/Users/ealvarad/00-Local/02-Python/Blender/data/displacements/{shot}/residuals/unobserved_residuals_only_{shot}_world_lbs.json"
#     # output_scaled_residuals_json_path = f"C:/Users/ealvarad/00-Local/02-Python/Blender/data/displacements/{shot}/residuals/unobserved_residuals_only_{shot}_world_lbs_scaled.json"

#     # --- 1. Load the Input Residuals JSON ---
#     try:
#         with open(input_residuals_json_path, 'r') as f:
#             residuals_data = json.load(f)
#     except Exception as e:
#         print(f"ERROR: Failed to load the input residuals JSON file for {shot}. Check the path. Error: {e}")
#         return

#     print(f"Loaded residuals for {len(residuals_data)} frames.")

#     # --- 2. Apply the Scale Factor ---
#     scaled_residuals = {}

#     for frame, markers in residuals_data.items():
#         scaled_residuals[frame] = {}
#         for marker, difference_vector in markers.items():
#             # Scale the difference vector
#             scaled_vector = Vector(difference_vector) * SCALE_FACTOR
#             scaled_residuals[frame][marker] = scaled_vector[:]

#     print(f"Applied scale factor to all residuals for {shot}.")

#     # --- 3. Save the Scaled Residuals to a New JSON File ---
#     output_dir = os.path.dirname(output_scaled_residuals_json_path)
#     os.makedirs(output_dir, exist_ok=True)

#     try:
#         with open(output_scaled_residuals_json_path, 'w') as f:
#             json.dump(scaled_residuals, f, indent=2)
#         print(f"Scaled residuals saved to: {output_scaled_residuals_json_path}")
#     except Exception as e:
#         print(f"ERROR: Failed to save the scaled residuals JSON file for {shot}. Error: {e}")


# # --- Run the main function for all shots ---
# if __name__ == "__main__":
#     for shot in shots:
#         scale_residuals(shot)