"""
Script: avg_displacement_markers.py
Plot the markers average displacement for a specific muscle group across multiple clips.
"""

import json
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# --- User Configuration ---

shot_A = f"shot_001"
shot_B = f"shot_004"
shot_C = f"shot_008"
shot_D = f"shot_012"

CLIPS_TO_COMPARE = {
    "0 Kg": f"C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/displacements/{shot_A}/reconstruction/refined_two_pass_displacements_{shot_A}.json",
    "10 Kg": f"C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/displacements/{shot_B}/reconstruction/refined_two_pass_displacements_{shot_B}.json",
    "15 Kg": f"C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/displacements/{shot_C}/reconstruction/refined_two_pass_displacements_{shot_C}.json",
    "20 Kg": f"C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/displacements/{shot_D}/reconstruction/refined_two_pass_displacements_{shot_D}.json"
}

MARKER_TO_MUSCLE_MAP_JSON = f"C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/mappings/marker_to_muscle_map.json"

TARGET_MUSCLE_LIST = [
    "l-rectus-femoris.001",
    "l-iliotibial-tract.001",
    "l-vastus-lateralis.001",
    "l-sartorius.001",
    "l-vastus-medialis.002",
    "r-rectus-femoris.001",
    "r-iliotibial-tract.001",
    "r-vastus-lateralis.001",
    "r-sartorius.001",
    "r-vastus-medialis.002"
]

BASE_DIR = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans"
PLOT_OUTPUT_PATH = os.path.join(BASE_DIR, "analysis/plots/deformation_comparison.png")

OUTPUT_METRICS_JSON_PATH = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/analysis/cache/"

# Smoothing Parameters
APPLY_SMOOTHING = True
# The size of the window used to fit the polynomial. MUST BE AN ODD INTEGER.
# A larger window means more smoothing.
SMOOTHING_WINDOW_SIZE = 21
# The order of the polynomial to fit. MUST BE SMALLER than the window size.
# A small value (like 2 or 3) is usually best to avoid overfitting noise.
SMOOTHING_POLY_ORDER = 3
# --------------------------------

def analyze_single_clip(displacements_path, marker_to_muscle_map, target_muscles):
    """Analyzes a single displacement file and returns the calculated metrics."""

    # Check if the displacement file exists
    try:
        with open(displacements_path, 'r') as f:
            displacements_by_frame = json.load(f)
    except FileNotFoundError:
        print(f"  Warning: Data file not found at {displacements_path}. Skipping.")
        return None
    except Exception as e:
        print(f"  ERROR: Could not load displacement file {displacements_path}. {e}")
        return None

    # Take the marker-to-muscle map and filter it for the target muscle group
    markers_on_target_group = {
        key for key, muscle in marker_to_muscle_map.items()
        if muscle in target_muscles
    }

    print(f"Found {len(markers_on_target_group)} total markers assigned to the specified muscle group.")

    print("Breakdown by muscle:")
    for muscle_name in target_muscles:
        # Count how many times each muscle name appears in the map values
        count = sum(1 for assigned_muscle in marker_to_muscle_map.values() if assigned_muscle == muscle_name)
        print(f"  - '{muscle_name}': {count} markers")

    if not markers_on_target_group:
        print(f"Warning: No markers found for muscle group {target_muscles}. Skipping analysis.")
        return None

    # Calculate the average displacement magnitude for each frame
    results = {}
    sorted_frame_keys = sorted(displacements_by_frame.keys(), key=int)
    for frame_str in sorted_frame_keys:
        frame_data = displacements_by_frame[frame_str]
        magnitudes = [np.linalg.norm(frame_data.get(key, [0, 0, 0])) for key in markers_on_target_group]
        if magnitudes:
            results[frame_str] = {"average_magnitude": np.mean(magnitudes)}

    output_json = OUTPUT_METRICS_JSON_PATH + f'deformation_metrics_{os.path.basename(displacements_path)}'
    print(f"\nSaving metrics in JSON format to: {output_json}")
    try:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=4)
        print("Metrics file saved successfully.")
    except Exception as e:
        print(f"ERROR: Could not save metrics file. {e}")

    return results

def compare_and_plot_clips():
    """Main function to analyze multiple clips and plot their results together."""

    print("--- Comparing Muscle Deformations Across Multiple Clips ---")

    # Load the shared marker-to-muscle map
    try:
        with open(MARKER_TO_MUSCLE_MAP_JSON, 'r') as f:
            marker_to_muscle_map = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load the marker map file. Aborting. {e}")
        return

    # Analyze each clip specified in the configuration
    all_clip_results = {}
    for clip_label, displacement_path in CLIPS_TO_COMPARE.items():

        # Construct the expected path for the cached metrics file
        base_filename = os.path.basename(displacement_path)
        metrics_filename = f"metrics_{base_filename}"
        metrics_filepath = os.path.join(OUTPUT_METRICS_JSON_PATH, metrics_filename)

        # Check if the cache file exists
        if os.path.exists(metrics_filepath):
            print(f"  -> Found pre-computed metrics file. Loading from cache...")
            try:
                with open(metrics_filepath, 'r') as f:
                    clip_metrics = json.load(f)
            except Exception as e:
                print(f"  Warning: Could not load cache file. Re-computing. Error: {e}")
                clip_metrics = None  # Force re-computation
        else:
            clip_metrics = None

        # If cache was not loaded, run the analysis from scratch
        if clip_metrics is None:
            print(f"  -> No cache found. Analyzing from scratch...")
            clip_metrics = analyze_single_clip(displacement_path, marker_to_muscle_map, TARGET_MUSCLE_LIST)

            # 4. If analysis was successful, save the new cache file
            if clip_metrics:
                print(f"     Saving new metrics to: {metrics_filepath}")
                # Ensure the cache directory exists
                os.makedirs(OUTPUT_METRICS_JSON_PATH, exist_ok=True)
                try:
                    with open(metrics_filepath, 'w') as f:
                        json.dump(clip_metrics, f, indent=4)
                    print("     Metrics file saved successfully.")
                except Exception as e:
                    print(f"     Warning: Could not save new metrics file. {e}")

        if clip_metrics:
            all_clip_results[clip_label] = clip_metrics

    if not all_clip_results:
        print("No data was successfully analyzed. Cannot create plot.");
        return

    # Plot the results from all clips on a single graph
    print("\nGenerating comparison plot...")
    plt.figure(figsize=(15, 8))  # Create a figure for the plot

    for clip_label, results in all_clip_results.items():
        # Extract data for plotting
        sorted_frames = sorted(results.keys(), key=int)
        frames = [int(f) for f in sorted_frames]
        avg_magnitudes = [results[f]['average_magnitude'] for f in sorted_frames]

        # --- THIS IS THE KEY CHANGE ---
        if APPLY_SMOOTHING and len(avg_magnitudes) > SMOOTHING_WINDOW_SIZE:
            # Apply the Savitzky-Golay filter
            smoothed_magnitudes = savgol_filter(avg_magnitudes, SMOOTHING_WINDOW_SIZE, SMOOTHING_POLY_ORDER)
            # Plot the smoothed data
            plt.plot(frames, smoothed_magnitudes, label=f"{clip_label}", linewidth=2.5, alpha=0.9)
            # Optionally, plot the original noisy data underneath with transparency
            # plt.plot(frames, avg_magnitudes, label=f"_{clip_label} (Raw)", color=plt.gca().lines[-1].get_color(),
            #          linestyle=':', alpha=0.3)
        else:
            # Plot the original data if not smoothing
            plt.plot(frames, avg_magnitudes, label=clip_label, linewidth=2)
        # --- END OF CHANGE ---

        # Plot this clip's data as a line on the graph
        # plt.plot(frames, avg_magnitudes, label=clip_label, linewidth=2, alpha=0.9)

    # --- Add labels, title, and other formatting ---
    group_name_title = ", ".join(TARGET_MUSCLE_LIST)
    plt.xlabel("Frame Number")
    plt.ylabel("Average Displacement Magnitude (mm)")
    plt.title(f"Muscle Group Deformation Comparison for: {group_name_title}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # --- 4. Save and show the plot ---
    try:
        plt.savefig(PLOT_OUTPUT_PATH, dpi=300)
        print(f"Comparison plot saved successfully to: {PLOT_OUTPUT_PATH}")
    except Exception as e:
        print(f"ERROR: Could not save plot. {e}")

    plt.show()

if __name__ == "__main__":
    compare_and_plot_clips()
