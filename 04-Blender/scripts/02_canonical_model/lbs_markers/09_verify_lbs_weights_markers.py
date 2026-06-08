"""
Script: verify_lbs_weights_markers.py
Goal:
"""

import json
import numpy as np

# --- USER: SET THIS ---
PATH_TO_MARKER_LBS_WEIGHTS = "C:/Users/ealvarad/00-Local/02-Python/Blender/data/weights/canonical_model/lbs_markers/markers_lbs_weights_exported_tpose.json"
# ----------------------

try:
    with open(PATH_TO_MARKER_LBS_WEIGHTS, 'r') as f:
        marker_lbs_weights_data = json.load(f)
except Exception as e:
    print(f"Error loading marker LBS weights: {e}"); exit()

print(f"Verifying weights for {len(marker_lbs_weights_data)} marker points...")

epsilon = 1e-5 # Tolerance for sum check
points_with_sum_issues = 0
points_with_negative_weights = 0
points_with_zero_influences = 0

for marker_id_str, data in marker_lbs_weights_data.items():
    weights = data.get("weights", [])
    bone_indices = data.get("bone_indices", [])

    if not weights or not bone_indices:
        points_with_zero_influences += 1
        print(f"Warning: Marker '{marker_id_str}' has zero bone influences.")
        continue

    sum_w = sum(weights)
    if abs(sum_w - 1.0) > epsilon:
        points_with_sum_issues += 1
        print(f"Warning: Marker '{marker_id_str}' weights sum to {sum_w:.6f} (not 1.0). Weights: {weights}")

    for w_val in weights:
        if w_val < -epsilon: # Allow for very slightly negative due to float precision
            points_with_negative_weights += 1
            print(f"Warning: Marker '{marker_id_str}' has negative weight: {w_val:.6f}. Weights: {weights}")
            break # Only report once per marker

print("\n--- Numerical Verification Summary ---")
if points_with_sum_issues == 0:
    print("All marker points have weights summing to ~1.0: OK")
else:
    print(f"{points_with_sum_issues} marker points have weights that DO NOT sum to ~1.0.")

if points_with_negative_weights == 0:
    print("No significantly negative weights found: OK")
else:
    print(f"{points_with_negative_weights} marker points have one or more negative weights.")

if points_with_zero_influences == 0:
    print("All marker points have at least one bone influence: OK")
else:
    print(f"{points_with_zero_influences} marker points have zero bone influences.")