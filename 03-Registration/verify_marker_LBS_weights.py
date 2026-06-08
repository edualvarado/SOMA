"""
Script: verify_marker_LBS_weights.py
Goal:   Sanity-check the output of create_marker_LBS_weights_final.py.

        For each marker corner it verifies:
          - Weights sum to ~1.0
          - No negative weights
          - At least one bone influence

Usage:
    python verify_marker_LBS_weights.py --input registration/S1/canonical_model/S1_marker_lbs_weights_exported.json
"""

import json
import argparse
from pathlib import Path
from loguru import logger


def main():
    parser = argparse.ArgumentParser(
        description="Verify that marker LBS weights are valid (normalized, non-negative, non-empty)."
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to the marker LBS weights JSON produced by create_marker_LBS_weights_final.py.")
    args = parser.parse_args()

    logger.info("Loading marker LBS weights from: {}", args.input)
    try:
        with open(args.input, "r") as f:
            marker_lbs_weights_data = json.load(f)
    except Exception as e:
        logger.error("Failed to load file: {}", e)
        return

    logger.info("Verifying weights for {} marker points...", len(marker_lbs_weights_data))

    epsilon = 1e-5
    sum_issues   = 0
    negative     = 0
    zero_influence = 0

    for marker_id, data in marker_lbs_weights_data.items():
        weights      = data.get("weights", [])
        bone_indices = data.get("bone_indices", [])

        if not weights or not bone_indices:
            zero_influence += 1
            logger.warning("Marker '{}' has zero bone influences.", marker_id)
            continue

        total = sum(weights)
        if abs(total - 1.0) > epsilon:
            sum_issues += 1
            logger.warning("Marker '{}' weights sum to {:.6f} (expected 1.0).", marker_id, total)

        if any(w < -epsilon for w in weights):
            negative += 1
            logger.warning("Marker '{}' has a negative weight.", marker_id)

    logger.info("--- Verification Summary ---")
    logger.info("Weights sum to ~1.0:          {}", "OK" if sum_issues    == 0 else f"{sum_issues} issues")
    logger.info("No negative weights:          {}", "OK" if negative       == 0 else f"{negative} issues")
    logger.info("All markers have influences:  {}", "OK" if zero_influence == 0 else f"{zero_influence} issues")

    if sum_issues == 0 and negative == 0 and zero_influence == 0:
        logger.success("All checks passed.")
    else:
        logger.error("Some checks failed — review warnings above.")


if __name__ == "__main__":
    main()
