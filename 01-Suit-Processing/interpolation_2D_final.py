"""
To interpolate for a given frame distance (preliminary) - TODO: Fix!
"""

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
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def linear_interpolation_charucos(point1, point2, alpha):
    return [p1 + alpha * (p2 - p1) for p1, p2 in zip(point1, point2)]

def linear_interpolation_markers(point1, point2, alpha):
    return [p1 + alpha * (p2 - p1) for p1, p2 in zip(point1, point2)]

def interpolate_missing_charucos(data, N):
    frames = sorted([int(frame) for frame in data.keys()])  # List of sorted frame keys as integers
    interpolation_info = {}  # Dictionary to track interpolation ranges for each ID

    for id in range(1001):  # IDs from 0 to 1000
        id_positions = []

        # Find frames containing the current ID and store their indices
        for i, frame in enumerate(frames):
            if id in [marker[0] for marker in data[str(frame)]['id_charuco']]:
                id_positions.append(i)

        for i in range(len(id_positions) - 1):
            start_frame_idx = id_positions[i]
            end_frame_idx = id_positions[i + 1]

            # Calculate the number of actual frames between start and end frames
            actual_missing_frames = frames[end_frame_idx] - frames[start_frame_idx] - 1

            # Debugging statement for frame gap
            # print(f"Checking ID {id} between frames {frames[start_frame_idx]} and {frames[end_frame_idx]}: {actual_missing_frames} missing frames.")

            # Check if the number of missing frames is exactly N
            if actual_missing_frames > 0 and actual_missing_frames <= N:
                # print(f"Interpolating ID {id} between frames {frames[start_frame_idx]} and {frames[end_frame_idx]}")

                start_frame = start_frame_idx
                end_frame = end_frame_idx

                # Fetch the starting and ending coordinates for the current ID
                start_index = [marker[0] for marker in data[str(frames[start_frame])]['id_charuco']].index(id)
                end_index = [marker[0] for marker in data[str(frames[end_frame])]['id_charuco']].index(id)
                start_coords = data[str(frames[start_frame])]['corners_charuco'][start_index]
                end_coords = data[str(frames[end_frame])]['corners_charuco'][end_index]

                for j in range(1, frames[end_frame] - frames[start_frame]):
                    alpha = j / (frames[end_frame] - frames[start_frame])
                    interpolated_coords = linear_interpolation_markers(start_coords, end_coords, alpha)
                    frame_to_update = str(frames[start_frame] + j)

                    # Check if the frame actually exists in the data before attempting to update
                    if frame_to_update not in data:
                        # Create the missing frame with empty lists for 'corners_charuco' and 'id_charuco'
                        data[frame_to_update] = {'corners_charuco': [], 'id_charuco': []}
                        # print(f"Created missing frame {frame_to_update} for interpolation.")

                    # Find the correct index to insert the interpolated id and coordinates
                    insert_index = next(
                        (k for k, marker in enumerate(data[frame_to_update]['id_charuco']) if marker[0] > id),
                        len(data[frame_to_update]['id_charuco'])
                    )

                    data[frame_to_update]['corners_charuco'].insert(insert_index, interpolated_coords)
                    data[frame_to_update]['id_charuco'].insert(insert_index, [id])

                    # Add the ID and the range to the interpolation info
                    if id not in interpolation_info:
                        interpolation_info[id] = []
                    if (start_frame, end_frame) not in interpolation_info[id]:
                        interpolation_info[id].append((start_frame, end_frame))

                    # Debugging statement for interpolation insertion
                    # print(f"Inserted interpolated ID {id} at frame {frame_to_update} between {frames[start_frame]} and {frames[end_frame]}.")

    # Print debug message with interpolation info
    # for id, ranges in interpolation_info.items():
    #     for start_frame, end_frame in ranges:
    #         print(f"[CHARUCOS] ID {id} interpolated from frame {frames[start_frame]} to {frames[end_frame]}")

    # Sort the data dictionary by its keys after interpolation
    sorted_data = {str(frame): data[str(frame)] for frame in sorted(int(key) for key in data.keys())}

    return sorted_data

def interpolate_missing_markers(data, N):
    frames = list(data.keys())
    interpolation_info = {}  # Dictionary to track interpolation ranges for each ID

    for id in range(1001):  # IDs from 0 to 1000
        id_positions = []

        for i, frame in enumerate(frames):
            if id in [marker[0] for marker in data[frame]['id_markers']]:
                id_positions.append(i)

        for i in range(len(id_positions) - 1):
            if 1 <= (id_positions[i + 1] - id_positions[i] - 1) <= N:
                start_frame = id_positions[i]
                end_frame = id_positions[i + 1]
                start_coords = data[frames[start_frame]]['corners_markers'][
                    [marker[0] for marker in data[frames[start_frame]]['id_markers']].index(id)]
                end_coords = data[frames[end_frame]]['corners_markers'][
                    [marker[0] for marker in data[frames[end_frame]]['id_markers']].index(id)]

                for j in range(1, end_frame - start_frame):
                    alpha = j / (end_frame - start_frame)
                    interpolated_coords = [linear_interpolation_markers(start, end, alpha) for start, end in
                                           zip(start_coords, end_coords)]
                    frame_to_update = frames[start_frame + j]

                    # Find the correct index to insert the interpolated id and coordinates
                    insert_index = next(
                        (k for k, marker in enumerate(data[frame_to_update]['id_markers']) if marker[0] > id),
                        len(data[frame_to_update]['id_markers']))

                    data[frame_to_update]['corners_markers'].insert(insert_index, interpolated_coords)
                    data[frame_to_update]['id_markers'].insert(insert_index, [id])

                    # Add the ID and the range to the interpolation info
                    if id not in interpolation_info:
                        interpolation_info[id] = []
                    if (start_frame, end_frame) not in interpolation_info[id]:
                        interpolation_info[id].append((start_frame, end_frame))

    # Print debug message with interpolation info
    # for id, ranges in interpolation_info.items():
    #     for start_frame, end_frame in ranges:
    #         print(f"[MARKERS] ID {id} interpolated from frame {frames[start_frame]} to {frames[end_frame]}")

    return data

def main():
    parser = argparse.ArgumentParser(description="Interpolate N detections in 2D videos")
    parser.add_argument("--frames_2D_int", type=int, required=True, help="Number of missing frames required to interpolate 2D")
    parser.add_argument("--folder", type=Path, required=True, help="Folder with MP4 videos")
    parser.add_argument("--keyword", type=str, default="stream", help="Keyword that must be in the name of the jsons")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    logger.info("=========================")
    logger.info("== 2. 2D Interpolation ==")
    logger.info("=========================")

    logger.info(f"[INFO] Interpolated Frames: {args.frames_2D_int}")
    logger.info(f"[INFO] Folder {args.folder}")

    detection_folder = args.folder / f"detections_charuco-suit"

    # Load the detections
    charucos = glob.glob(os.path.join(detection_folder, 'corners_charuco*.json'))
    charucos = sorted(charucos)

    # Load the detections
    markers = glob.glob(os.path.join(detection_folder, 'corners_markers*.json'))
    markers = sorted(markers)

    # Define the output directory
    output_directory_path = args.folder/f"detections_charuco-suit/2D-interpolated-N{args.frames_2D_int}"  # Replace with the path to your output directory

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory_path, exist_ok=True)

    # Loop over the list of JSON files
    """
    for charucos in tqdm(charucos[:], desc="Processing charucos"):
        # Load the JSON data
        data = load_json(charucos)

        # Interpolate missing frames
        if args.verbose:
            logger.debug(f"[DEBUG] Interpolating missing frames in {charucos}...")
        interpolated_data = interpolate_missing_charucos(data, args.frames_2D_int)

        # Create the output file path
        base_name = os.path.basename(charucos)
        output_file_path = os.path.join(output_directory_path, base_name)

        # Save the interpolated data
        save_json(interpolated_data, output_file_path)

        logger.info(f"[INFO] Interpolated data saved to {output_file_path}")
    """
    
    # Loop over the list of JSON files
    for markers in tqdm(markers[:], desc="Processing markers"):
        # Load the JSON data
        data = load_json(markers)

        # Interpolate missing frames
        if args.verbose:
            logger.debug(f"[DEBUG] Interpolating missing frames in {markers}...")
        interpolated_data = interpolate_missing_markers(data, args.frames_2D_int)

        # Create the output file path
        base_name = os.path.basename(markers)
        output_file_path = os.path.join(output_directory_path, base_name)

        # Save the interpolated data
        save_json(interpolated_data, output_file_path)

        logger.info(f"[INFO] Interpolated data saved to {output_file_path}")

if __name__ == "__main__":
    main()
