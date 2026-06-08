"""
Script: merge_marker_annotations.py
Goal:   Merge two annotation JSON files (e.g. auto-detected + manually labelled)
        into a single output JSON.

Used twice in the annotation pipeline:

  Step 4 — merge automatic (fixed) + manual 4-point annotations:
    python merge_marker_annotations.py \\
        --json1  S1/uv_detections_charuco-suit/markers-skin-fixed.json \\
        --json2  S1/uv_detections_charuco-suit/markers-skin-manual.json \\
        --output S1/uv_detections_charuco-suit/markers-skin-final.json

  Step 7 — merge previous final + manual 1-point edge annotations:
    python merge_marker_annotations.py \\
        --json1  S1/uv_detections_charuco-suit/markers-skin-final.json \\
        --json2  S1/uv_detections_charuco-suit/markers-skin-manual-1-point-corrected.json \\
        --output S1/uv_detections_charuco-suit/markers-skin-final-corrected.json
"""

import json
import argparse
from pathlib import Path
from loguru import logger


def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description="Merge two marker annotation JSON files into one."
    )
    parser.add_argument("--json1", type=Path, required=True,
                        help="Primary annotation JSON (base file, kept as-is).")
    parser.add_argument("--json2", type=Path, required=True,
                        help="Secondary annotation JSON (appended to the primary).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output path for the merged JSON.")
    args = parser.parse_args()

    logger.info("Loading primary annotations from:   {}", args.json1)
    with open(args.json1, "r") as f:
        data_1 = json.load(f)

    logger.info("Loading secondary annotations from: {}", args.json2)
    with open(args.json2, "r") as f:
        data_2 = json.load(f)

    # Append secondary markers into a copy of the primary data
    merged = data_1.copy()
    merged["0"]["corners_markers"].extend(data_2["0"]["corners_markers"])
    merged["0"]["id_markers"].extend(data_2["0"]["id_markers"])

    n1 = len(data_1["0"]["id_markers"])
    n2 = len(data_2["0"]["id_markers"])
    logger.info("Merged {} (primary) + {} (secondary) = {} total markers.", n1, n2, n1 + n2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(merged, args.output)
    logger.success("Saved merged annotations to: {}", args.output)


if __name__ == "__main__":
    main()
