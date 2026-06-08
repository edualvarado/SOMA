"""
STATUS: Optimization (Streaming) - Creates binary mask for observed vs unobserved markers.
Script: estimate_residuals_mask_stream.py
Goal: Writes the mask file incrementally to handle large datasets without memory crashes.
"""

import bpy
import json
import os
import gc 

def calculate_masked_residuals_streaming(shot):
    """
    Calculates and saves masked residuals frame-by-frame to minimize memory usage.
    """
    
    # --- Configuration for Paths ---
    # Update base path if necessary
    BASE_PATH = "S:/work/03-MUSK/04-Blender/data/registration"
    
    CANONICAL_MARKERS_JSON_PATH = f"{BASE_PATH}/S5/canonical_model/S5_canonical_data.json"
    MOTION_DATA_JSON_PATH = f"{BASE_PATH}/S5/{shot}/S5_triangulated_sequence_{shot}_transformed.json"
    OUTPUT_MASKED_DISPLACEMENTS_JSON_PATH = f"{BASE_PATH}/S5/{shot}/S5_masked_residuals_{shot}_world_tpose.json"

    # --- 1. Load Input Data ---
    print(f"--- Processing {shot} (Streaming Mode) ---")
    print("Loading Canonical and Motion Data...")
    
    try:
        with open(CANONICAL_MARKERS_JSON_PATH, 'r') as f:
            # Safely get frame "0" or just the dict depending on structure
            loaded_canonical = json.load(f)
            canonical_points_raw = loaded_canonical.get("0", loaded_canonical)
            
        with open(MOTION_DATA_JSON_PATH, 'r') as f:
            motion_data_by_frame = json.load(f)
            
    except Exception as e:
        print(f"ERROR: Failed to load inputs. {e}")
        return

    # Extract just the keys (marker IDs) since we only need to check existence
    canonical_keys = set(canonical_points_raw.keys())
    print(f"Loaded {len(canonical_keys)} canonical markers.")
    
    # Sort frames to ensure clean sequential writing
    # Filter to ensure we only get numeric frame keys
    sorted_frame_keys = sorted([k for k in motion_data_by_frame.keys() if k.isdigit()], key=int)
    total_frames = len(sorted_frame_keys)
    print(f"Loaded {total_frames} frames of motion data.")

    # --- 2. Stream Process & Write ---
    print(f"Streaming results to: {OUTPUT_MASKED_DISPLACEMENTS_JSON_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_MASKED_DISPLACEMENTS_JSON_PATH), exist_ok=True)

    try:
        with open(OUTPUT_MASKED_DISPLACEMENTS_JSON_PATH, 'w') as f_out:
            f_out.write('{\n') # Start JSON object

            for i, frame_str in enumerate(sorted_frame_keys):
                
                # Get observed markers for this frame
                observed_markers_t = motion_data_by_frame.get(frame_str, {})
                
                # Create mask for this frame ONLY
                # 1 if observed (in motion data), 0 if missing
                masked_displacements_t = {}
                
                for marker_key in canonical_keys:
                    if marker_key in observed_markers_t:
                        masked_displacements_t[marker_key] = [1]
                    else:
                        masked_displacements_t[marker_key] = [0]

                # Write immediate frame result
                # Indent for pretty printing (2 spaces)
                json_string = json.dumps(masked_displacements_t)
                f_out.write(f'  "{frame_str}": {json_string}')

                # Add comma if not the last frame
                if i < total_frames - 1:
                    f_out.write(',\n')
                else:
                    f_out.write('\n') # Last frame, no comma

                # Progress Update
                if i % 100 == 0:
                    print(f"  Processed frame {frame_str}/{sorted_frame_keys[-1]}...", end='\r')

            f_out.write('}') # Close JSON object

        print(f"\nSUCCESS: Completed {shot}.")

    except Exception as e:
        print(f"\nCRITICAL ERROR during write: {e}")

    # Clean up memory
    del motion_data_by_frame
    del canonical_keys
    gc.collect()

if __name__ == "__main__":
    # Define your shots here
    shots = ["shot_001"] 
    
    for shot in shots:
        calculate_masked_residuals_streaming(shot)


# ---

# """
# STATUS: Complete - Creates a binary mask for observed vs unobserved markers.
# Script: estimate_residuals_mask.py
# Goal: For each frame, set 1 for the OBSERVED markers. For UNOBSERVED markers, assign a zero.
# """

# import bpy
# import json
# import numpy as np
# from mathutils import Vector, Matrix
# import os


# def calculate_masked_residuals():
#     """Main function to calculate masked displacements."""

#     # --- 1. Load All Necessary Data Files ---
#     print("--- Step 1: Loading Input Data ---")
#     try:
#         with open(CANONICAL_MARKERS_JSON_PATH, 'r') as f:
#             canonical_points_raw = json.load(f).get("0", {})
#         with open(MOTION_DATA_JSON_PATH, 'r') as f:
#             motion_data_by_frame = json.load(f)
#     except Exception as e:
#         print(f"ERROR: Failed to load one or more JSON files. Check paths. Error: {e}"); return

#     canonical_points = {key: Vector(val[0]) for key, val in canonical_points_raw.items()}
#     print(f"Loaded {len(canonical_points)} canonical marker positions.")
#     print(f"Loaded observed motion data for {len(motion_data_by_frame)} frames.")

#     # --- 3. Set up and Run Animation Loop ---
#     all_frames_masked_displacements = {}

#     if not motion_data_by_frame:
#         print("Motion data file is empty. Nothing to process."); return

#     sorted_frame_keys = sorted(motion_data_by_frame.keys(), key=int)
#     print(f"Processing {len(sorted_frame_keys)} frames...")

#     for frame_str in sorted_frame_keys:
#         frame = int(frame_str)
#         if frame % 25 == 0:
#             print(f"  Processing frame {frame}...")

#         observed_markers_t = motion_data_by_frame[frame_str]
#         masked_displacements_t = {}

#         # --- CORE LOGIC: Loop through ALL canonical markers ---
#         for marker_key, _ in canonical_points.items():
#             # Check if this canonical marker was observed in the current frame
#             if marker_key in observed_markers_t:
#                 masked_displacements_t[marker_key] = [1]
#             else:
#                 masked_displacements_t[marker_key] = [0]

#         all_frames_masked_displacements[frame_str] = masked_displacements_t

#     print("\nFinished processing all animation frames.")

#     # --- 4. Save the Final Results to TWO JSON Files ---
#     print(f"Saving masked residuals to: {OUTPUT_MASKED_DISPLACEMENTS_JSON_PATH}")
    
#     # Ensure the directory exists
#     output_dir = os.path.dirname(OUTPUT_MASKED_DISPLACEMENTS_JSON_PATH)
#     os.makedirs(output_dir, exist_ok=True)

#     try:
#         with open(OUTPUT_MASKED_DISPLACEMENTS_JSON_PATH, 'w') as f:
#             json.dump(all_frames_masked_displacements, f, indent=2)
#         print("Save complete.")
#     except Exception as e:
#         print(f"ERROR writing observed displacements JSON: {e}")

# # --- Run the main function ---
# if __name__ == "__main__":
#     # ---
#     shots = ["shot_001"]  # Change this to your shot name
#     # shots = ["shot_002", "shot_003", "shot_004", "shot_005", "shot_006", "shot_007", "shot_008", "shot_009", "shot_010", "shot_011", "shot_012", "shot_013", "shot_014", "shot_015", "shot_016", "shot_017", "shot_018", "shot_019", "shot_020"]
#     # ---

#     for shot in shots:
#         print(f"Processing shot: {shot}")

#         # CANONICAL_MARKERS_JSON_PATH = "C:/Users/ealvarad/00-Local/02-Python/Blender/data/registration/canonical_model/canonical_data.json"
#         CANONICAL_MARKERS_JSON_PATH = "S:/work/03-MUSK/04-Blender/data/registration/S5/canonical_model/S5_canonical_data.json"
#         # CANONICAL_MARKERS_JSON_PATH = "C:/Users/ealvarad/00-Local/02-Python/Blender/data/registration/canonical_model/canonical_data_tpose.json" # <-- TODO: Check if we need this.

#         # MOTION_DATA_JSON_PATH = f"C:/Users/ealvarad/00-Local/02-Python/Blender/data/registration/{shot}/triangulated_sequence_{shot}_transformed_filtered.json"
#         MOTION_DATA_JSON_PATH = f"S:/work/03-MUSK/04-Blender/data/registration/S5/{shot}/triangulated_sequence_{shot}_transformed.json"

#         # Output path
#         # OUTPUT_MASKED_DISPLACEMENTS_JSON_PATH = f"C:/Users/ealvarad/00-Local/02-Python/Blender/data/displacements/{shot}/residuals/masked_residuals_{shot}_world_tpose.json"
#         OUTPUT_MASKED_DISPLACEMENTS_JSON_PATH = f"S:/work/03-MUSK/04-Blender/data/registration/S5/{shot}/masked_residuals_{shot}_world_tpose.json"

#         # PARENT_COLLECTION_NAME = shot.capitalize()
#         PARENT_COLLECTION_NAME =  "Shot_001_S5"

#         calculate_masked_residuals()