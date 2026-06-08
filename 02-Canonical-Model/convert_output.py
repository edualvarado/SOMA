"""
Script: convert_output.py
Goal:   Convert the outputs from the Blender canonical model step and the 3D
        triangulation pipeline into the unified format required for registration.

        Handles two conversions (both triggered by default):

        1. Canonical model  (run once per subject)
           Input:  {base_path}/source/canonical_model/output.json
           Output: {base_path}/registration/canonical_model/canonical_data.json

        2. Triangulation sequence  (run once per shot)
           Input:  {base_path}/source/{shot}/triangulation_markers_processed.json
           Output: {base_path}/registration/{shot}/{subject}_triangulated_sequence_{shot}.json

        Both conversions rename marker keys from the Blender/tracking format:
            marker_{id}_corner_{n}  /  {id}: [[x,y,z], ...]
        to the unified registration format:
            marker_{id}_{instance}_{corner}: [[x,y,z]]
        where 'instance' handles markers that appear in multiple mesh locations.

Usage:
    # Convert both canonical model and one shot (most common):
    python convert_output.py --subject S1 --shot shot_001

    # Convert only the canonical model (first time for a new subject):
    python convert_output.py --subject S1

    # Convert only a new shot (canonical already done):
    python convert_output.py --subject S1 --shot shot_002 --skip_canonical

    # Override the default base path:
    python convert_output.py --subject S1 --shot shot_001 --base_path /custom/path/S1/data
"""

import json
import argparse
import gc
from pathlib import Path
from loguru import logger


# ---------------------------------------------------------------------------
# Canonical model conversion
# ---------------------------------------------------------------------------

def convert_canonical_model(base_path: Path, subject: str):
    """
    Converts Blender's output.json to canonical_data.json.

    Blender produces one entry per marker-corner, where a marker placed on
    multiple mesh locations yields a list with more than one 3D point:
        "marker_93_corner_0": [[x,y,z], [x,y,z]]   <- two instances
        "marker_459_corner_0": [[x,y,z]]             <- one instance

    We flatten this into one key per (marker, instance, corner):
        "marker_93_0_0": [[x,y,z]]
        "marker_93_1_0": [[x,y,z]]
        "marker_459_0_0": [[x,y,z]]
    """
    input_path = base_path / "source/canonical_model/output.json"
    output_dir  = base_path / "registration/canonical_model"
    output_path = output_dir / "canonical_data.json"

    logger.info("[Canonical] Reading: {}", input_path)
    with open(input_path, "r") as f:
        data = json.load(f)

    canonical_data = {"0": {}}

    for key, value in data.items():
        # Key format: marker_{id}_corner_{n}
        parts        = key.split('_')
        marker_id    = parts[1]
        corner_index = parts[-1]

        if len(value) > 1:
            # Marker appears on multiple mesh locations
            logger.warning("[Canonical] Duplicate marker {}: {} instances.", marker_id, len(value))
            for instance_idx, corner_coord in enumerate(value):
                new_key = f"marker_{marker_id}_{instance_idx}_{corner_index}"
                canonical_data["0"][new_key] = [corner_coord]
        else:
            new_key = f"marker_{marker_id}_0_{corner_index}"
            canonical_data["0"][new_key] = value  # already [[x, y, z]]

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(canonical_data, f, indent=4)
    logger.success("[Canonical] Saved to: {}", output_path)


# ---------------------------------------------------------------------------
# Triangulation sequence conversion
# ---------------------------------------------------------------------------

def convert_triangulation(base_path: Path, subject: str, shot: str):
    """
    Converts triangulation_markers_processed.json to the registration format.

    Uses streaming writes (frame-by-frame) to avoid memory errors on long
    sequences. The input is loaded into memory once; only the output is streamed.

    Input format:
        { "0": { "451": [[x,y,z], [x,y,z], [x,y,z], [x,y,z]], ... }, ... }

    Output format:
        { "0": { "marker_451_0_0": [[x,y,z]], "marker_451_0_1": [[x,y,z]], ... }, ... }

    Duplicate markers (nested list-of-corner-sets) are split into separate
    instances following the same marker_{id}_{instance}_{corner} convention.
    """
    input_path = base_path / f"source/{shot}/triangulation_markers_processed.json"
    output_dir  = base_path / f"registration/{shot}"
    output_path = output_dir / f"{subject}_triangulated_sequence_{shot}.json"

    logger.info("[Triangulation] Reading: {}", input_path)
    try:
        with open(input_path, "r") as f:
            triangulation_data = json.load(f)
    except Exception as e:
        logger.error("[Triangulation] Failed to load input: {}", e)
        return

    sorted_frames = sorted([k for k in triangulation_data.keys() if k.isdigit()], key=int)
    total_frames  = len(sorted_frames)
    logger.info("[Triangulation] Loaded {} frames. Starting stream conversion...", total_frames)

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "w") as f_out:
            f_out.write('{\n')

            for i, frame in enumerate(sorted_frames):
                markers              = triangulation_data[frame]
                transformed_frame    = {}

                for marker_id, corners_data in markers.items():
                    # Detect duplicates: corners_data is a list-of-corner-sets
                    # (extra nesting level) instead of a flat list of corners.
                    is_duplicate = (
                        corners_data
                        and isinstance(corners_data[0], list)
                        and corners_data[0]
                        and isinstance(corners_data[0][0], list)
                    )

                    if is_duplicate:
                        logger.warning("[Triangulation] Duplicate: marker {} in frame {}.", marker_id, frame)
                        for instance_idx, corner_set in enumerate(corners_data):
                            for corner_idx, corner_coord in enumerate(corner_set):
                                new_key = f"marker_{marker_id}_{instance_idx}_{corner_idx}"
                                transformed_frame[new_key] = [corner_coord]
                    else:
                        for corner_idx, corner_coord in enumerate(corners_data):
                            new_key = f"marker_{marker_id}_0_{corner_idx}"
                            transformed_frame[new_key] = [corner_coord]

                # Write frame immediately to disk (streaming)
                f_out.write(f'    "{frame}": {json.dumps(transformed_frame)}')
                f_out.write(',\n' if i < total_frames - 1 else '\n')

                if i % 100 == 0:
                    print(f"  Converted frame {frame}/{sorted_frames[-1]}...", end='\r')

            f_out.write('}')

        logger.success("[Triangulation] Saved to: {}", output_path)

    except Exception as e:
        logger.error("[Triangulation] Critical error during write: {}", e)

    # Free the input data explicitly to help GC on large sequences
    del triangulation_data
    gc.collect()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert Blender/triangulation outputs to the registration format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--subject", type=str, required=True,
        help="Subject identifier (e.g. S1, S2). Used to build default paths and output filenames."
    )
    parser.add_argument(
        "--shot", type=str, default=None,
        help="Shot identifier (e.g. shot_001). If omitted, only the canonical model is converted."
    )
    parser.add_argument(
        "--base_path", type=Path, default=None,
        help="Override the base path. Defaults to '{subject}/uv_detections_charuco-suit'."
    )
    parser.add_argument(
        "--skip_canonical", action="store_true",
        help="Skip the canonical model conversion (useful when it was already done for this subject)."
    )
    args = parser.parse_args()

    base_path = args.base_path or Path(f"{args.subject}/uv_detections_charuco-suit")

    logger.info("===================================")
    logger.info("==       convert_output.py       ==")
    logger.info("===================================")
    logger.info("Subject:        {}", args.subject)
    logger.info("Shot:           {}", args.shot or "(not provided)")
    logger.info("Base path:      {}", base_path)
    logger.info("Skip canonical: {}", args.skip_canonical)

    if not args.skip_canonical:
        convert_canonical_model(base_path, args.subject)
    else:
        logger.info("[Canonical] Skipped via --skip_canonical.")

    if args.shot:
        convert_triangulation(base_path, args.subject, args.shot)
    else:
        logger.info("[Triangulation] Skipped — no --shot provided.")


if __name__ == "__main__":
    main()
