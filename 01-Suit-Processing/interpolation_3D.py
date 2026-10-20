import json
import numpy as np
import os
import glob
import argparse
from pathlib import Path
from loguru import logger
import toolkit.loading as tl
from tqdm import tqdm


def load_json(file_path):
    """Loads data from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data, file_path):
    """Saves data to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def linear_interpolation_3d(point1, point2, alpha):
    """
    Performs linear interpolation between two sets of 3D points.

    Args:
        point1: A list of 3D coordinates (e.g., [[x1, y1, z1], [x2, y2, z2]]).
        point2: A list of 3D coordinates of the same structure as point1.
        alpha: Interpolation factor (a float between 0 and 1).

    Returns:
        A list of 3D coordinates representing the interpolated points.
    """
    return [[p1 + alpha * (p2 - p1) for p1, p2 in zip(coord1, coord2)] for coord1, coord2 in zip(point1, point2)]


def interpolate_missing_3D(data, max_gap):
    """
    Interpolates missing sets of 3D points in the data.

    Args:
        data: Dictionary where keys are frame numbers (as strings) and values are nested dictionaries of 3D points.
        max_gap: Maximum number of frames for which missing data will be interpolated.

    Returns:
        The modified data dictionary with interpolated points added.
    """
    frames = sorted([int(frame) for frame in data.keys()])  # Sorted frame numbers
    all_ids = {id_key for frame_data in data.values() for id_key in frame_data.keys()}

    for obj_id in all_ids:
        frame_indices = [frame for frame in frames if obj_id in data[str(frame)]]

        for i in range(len(frame_indices) - 1):
            start_frame = frame_indices[i]
            end_frame = frame_indices[i + 1]
            gap = end_frame - start_frame - 1

            if 0 < gap <= max_gap:
                # Interpolate positions for the gap
                start_point = data[str(start_frame)][obj_id]
                end_point = data[str(end_frame)][obj_id]

                for j in range(1, gap + 1):
                    missing_frame = start_frame + j
                    alpha = j / (gap + 1)  # Calculate interpolation factor
                    interpolated_point = linear_interpolation_3d(start_point, end_point, alpha)

                    if str(missing_frame) not in data:
                        data[str(missing_frame)] = {}

                    data[str(missing_frame)][obj_id] = interpolated_point

    return data



def main():
    parser = argparse.ArgumentParser(description="Interpolate N detections in 2D videos")
    parser.add_argument("--frames_3D_int", type=int, required=True, help="Number of missing frames required to interpolate 3D")
    parser.add_argument("--folder", type=Path, required=True, help="Folder with MP4 videos")
    parser.add_argument("--keyword", type=str, default="stream", help="Keyword that must be in the name of the jsons")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    logger.info("=========================")
    logger.info("== 4. 3D Interpolation ==")
    logger.info("=========================")

    logger.info(f"[INFO] Interpolated Frames: {args.frames_3D_int}")
    logger.info(f"[INFO] Folder {args.folder}")

    triangulation_folder = args.folder / f"tracking_charuco-suit/triangulation"

    # Load the triangulations
    markers = glob.glob(os.path.join(triangulation_folder, 'triangulation_markers*.json'))
    markers = sorted(markers)

    # Define the output directory
    output_directory_path = args.folder / f"tracking_charuco-suit/triangulation/3D-interpolated-N{args.frames_3D_int}"  # Replace with the path to your output directory

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory_path, exist_ok=True)

    # Load the input data
    data = load_json(markers[0])

    if args.verbose:
        logger.debug(f"[DEBUG] Interpolating 3D missing frames in {markers[0]}...")

    # Interpolate missing points
    interpolated_data = interpolate_missing_3D(data, args.frames_3D_int)

    # Create the output file path
    base_name = os.path.basename(markers[0])
    output_file_path = os.path.join(output_directory_path, base_name)

    # Save the interpolated data
    save_json(interpolated_data, output_file_path)
    logger.info(f"[INFO] Interpolated data saved to {output_file_path}")


if __name__ == "__main__":
    main()
