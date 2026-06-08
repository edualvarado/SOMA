"""
Script: fix_manual_annotations.py
Goal:   Fix the format produced by manual_marker_annotator_1point.py.

        The 1-point annotator saves each corner click as a separate entry,
        resulting in one flat list of single-point entries and repeated IDs:

            "corners_markers": [ [[x,y]], [[x,y]], [[x,y]], [[x,y]], ... ]
            "id_markers":      [ [166], [166], [166], [166], [165], ... ]

        This script groups them back into the standard 4-corners-per-marker
        format used by the rest of the pipeline:

            "corners_markers": [ [[x,y],[x,y],[x,y],[x,y]], ... ]
            "id_markers":      [ [166], [165], ... ]

Usage:
    python fix_manual_annotations.py \\
        --input  S1/uv_detections_charuco-suit/markers-skin-manual-1-point.json \\
        --output S1/uv_detections_charuco-suit/markers-skin-manual-1-point-corrected.json
"""

import json
import argparse
from pathlib import Path
from loguru import logger


def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def group_markers(input_data):
    """
    Groups flat single-point entries into 4-corners-per-marker groups.
    Assumes the annotator always produces exactly 4 clicks per marker,
    in order, with the same ID repeated 4 times.
    """
    grouped_data = {"0": {"corners_markers": [], "id_markers": []}}
    corners = input_data["0"]["corners_markers"]
    ids     = input_data["0"]["id_markers"]

    # Each marker is represented by 4 consecutive single-point entries
    for i in range(0, len(corners), 4):
        grouped_data["0"]["corners_markers"].append(
            [corners[i][0], corners[i+1][0], corners[i+2][0], corners[i+3][0]]
        )
        grouped_data["0"]["id_markers"].append(ids[i])  # ID is repeated 4x; keep one

    return grouped_data


def main():
    parser = argparse.ArgumentParser(
        description="Reformat 1-point manual annotations into 4-corners-per-marker format."
    )
    parser.add_argument("--input",  type=Path, required=True,
                        help="Input JSON from manual_marker_annotator_1point.py.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output path for the corrected JSON.")
    args = parser.parse_args()

    logger.info("Reading annotations from: {}", args.input)
    with open(args.input, "r") as f:
        data = json.load(f)

    n_raw = len(data["0"]["id_markers"])
    transformed_data = group_markers(data)
    n_grouped = len(transformed_data["0"]["id_markers"])

    logger.info("Grouped {} entries into {} markers.", n_raw, n_grouped)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(transformed_data, args.output)
    logger.success("Saved corrected annotations to: {}", args.output)


if __name__ == "__main__":
    main()
