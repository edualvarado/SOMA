"""
Script: detect_2D_uv.py
Goal: It creates uv_detections_charuco-suit/markers-skin.json using the input image skin.png
Then it also saves the debug image skin_charuco-suit in /debug

This script makes the 2D detection from the suit (v1, v2) from the UV map and saves the results to a JSON file.
The detections are not complete - must be processed.

JSON structure (corners_markers and corners_charuco):

{
  "0": {
    "corners_markers": [
      [[X,Y],[X,Y],[X,Y],[X,Y]]],
    "id_markers": [["ID"]]
    }
}

{
  "0": {
    "corners_charuco": [
      [X,Y]],
    "id_charuco": [["ID"]]
    }
}

Example usage:
    python detect_2D_uv.py --folder /CT/MUSK/static00/280424-testbench/v2-scans/inside-humans/ --board configs/suits/charuco-suit.json --debug

We run the detection in the skin image. Problems:
- Sparse detections (fixed with manual annotation)
- Four IDs (925 -> 256, 731 -> 141, 995 -> 940, 262 -> 598) are wrong. -> Was fixed manually checking the .json

Run the script with --help to see all available options.
"""

import itertools
import argparse
import json
import multiprocessing
import time
import cv2
import numpy as np
import random
import re
import cv2.aruco as aruco
from multiprocessing import Pool
from pathlib import Path
from typing import Callable

from caliboards import BoardFactory
from pathlib import Path
from loguru import logger
from collections import defaultdict
from functools import singledispatch

def detect_board_in_uv(
    image_path: Path,
    board_path: Path,
    output_folder: Path,
    debug: bool = False,  # TODO: DEFINE to save or not debug images
    update_fn: Callable = None
):
    """
    Detect a `board` (or pattern in the suit) in a video and save the results to a JSON file.

    Args:
        image_path: Path to the image file.
        board_path: Path to the board JSON file.
        output_folder: Path to the output folder.
        roi_params: Tuple with the sliding window parameters.
        debug: Whether to save debug images.
        update_fn: Function to call to update the progress bar.
    """

    # Define the output file paths for the ChArUco detection .json (for saving)
    detection_file_path = output_folder / f"charuco-{image_path.stem}.json"

    # Define the output file paths for the ArUco marker detection .json (for saving)
    detection_markers_file_path = output_folder / f"markers-{image_path.stem}.json"

    # Dictionary to store the detected corner markers and ids for each frame
    frame_to_corners_markers = {}

    # Load the board from the JSON file
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    board = aruco.CharucoBoard((49, 65), 0.03, 0.022, aruco_dict)
    board.setLegacyPattern(True)

    frame = cv2.imread(image_path)

    image_width = 4096  # Image width in pixels
    image_height = 4096  # Image height in pixels

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Aruco markers (directly, no instance needed)
    parameters = cv2.aruco.DetectorParameters()
    corners_markers, id_markers, _ = aruco.detectMarkers(gray, board.getDictionary(),
                                                  parameters=parameters)

    # If markers are detected, try to detect the Charuco board
    if id_markers is not None:
        logger.info("[INFO] Charuco Board Detected!")

        if debug:
            draw_debug_image(corners_markers, id_markers, image_path, board_path, frame)
            save_debug_image(output_folder, board_path, frame)
        # --

        corners_markers_np = np.concatenate(corners_markers, axis=0)

        frame_dict_markers = dict(corners_markers=corners_markers_np.tolist(), id_markers=id_markers.tolist())
        frame_to_corners_markers["0"] = frame_dict_markers

        save_result_to_json(frame_to_corners_markers, detection_markers_file_path)

    else:
        print("No markers detected in the image")

def save_result_to_json(frame_to_corners, path):
    with open(path, "w") as file:
        json.dump(frame_to_corners, file, indent=2, default=to_serializable)

@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)

def save_debug_image(image_path, board_path, frame):
    debug_folder = image_path.parent / "debug"
    debug_folder.mkdir(exist_ok=True)
    path = debug_folder / f"skin_charuco-suit.jpg"
    cv2.imwrite(str(path), frame)

def draw_debug_image(corners_markers, id_markers, image_path, board_path, frame):
    # Pass 1: Fill all the markers
    for marker in corners_markers:
        # Reshape the corner array to integers for drawing
        pts = marker.reshape((-1, 1, 2)).astype(np.int32)

        # Fill the inside of the marker with a custom color (e.g., red)
        cv2.fillPoly(frame, [pts], color=(0, 0, 255))  # Fill with red color

        # Pass 2: Draw all the text
        for i, corner in enumerate(corners_markers):
            # Calculate the center of the marker
            marker_center = tuple(np.mean(corner.reshape((-1, 2)), axis=0).astype(int))

            # Get the ID of the marker and convert it to a string
            marker_id = str(id_markers[i][0])  # Convert numpy.int32 to string

            # Define font properties
            fontFace = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            textThickness = 2  # Thickness of the actual text

            # Draw the border (text outline) with a larger size in black
            cv2.putText(
                frame,
                marker_id,
                marker_center,
                fontFace,
                fontScale,
                color=(0, 0, 0),  # Outer color (e.g., black for contour)
                thickness=textThickness + 2,  # Make the border thicker
                lineType=cv2.LINE_AA
            )

            # Draw the actual text on top in white
            cv2.putText(
                frame,
                marker_id,
                marker_center,
                fontFace,
                fontScale,
                color=(255, 255, 255),  # Inner color (text color, e.g., white)
                thickness=textThickness,
                lineType=cv2.LINE_AA
            )

def average_corners(input_dict):
    # Create a dictionary to accumulate the sum of corners and count of occurrences for each id
    id_corner_sum = defaultdict(lambda: [0, 0])
    id_count = defaultdict(int)

    corners = input_dict["corners"]
    ids = input_dict["id"]

    for corner, id_val in zip(corners, ids):
        id_val = id_val[0]  # Extract the single id value from the list
        id_corner_sum[id_val][0] += corner[0]
        id_corner_sum[id_val][1] += corner[1]
        id_count[id_val] += 1

    # Calculate the average corners for each id
    averaged_corners = []
    unique_ids = []
    for id_val, sum_corner in id_corner_sum.items():
        avg_corner = [sum_corner[0] / id_count[id_val], sum_corner[1] / id_count[id_val]]
        averaged_corners.append(avg_corner)
        unique_ids.append([id_val])

    # Construct the output dictionary
    output_dict = {
        "corners": averaged_corners,
        "id": unique_ids
    }

    return output_dict



def average_corners_markers(input_dict):
    # Initialize a dictionary to accumulate the sum of corners and count of occurrences for each id
    id_corner_sum = defaultdict(lambda: [[0, 0], [0, 0], [0, 0], [0, 0], 0])

    corners = input_dict["cornersMarkers"]
    ids = input_dict["idMarkers"]

    # Iterate over cornersMarkers and idMarkers
    for corner, id_val in zip(corners, ids):
        id_val = id_val[0]  # Extract the single id value from the list
        for i in range(4):  # For each of the four corners
            id_corner_sum[id_val][i][0] += corner[i][0]
            id_corner_sum[id_val][i][1] += corner[i][1]
        id_corner_sum[id_val][4] += 1  # Increment the count

    # Calculate the average corners for each id
    averaged_corners = []
    unique_ids = []
    for id_val, sum_corner in id_corner_sum.items():
        avg_corner = [[sum_corner[i][0] / sum_corner[4], sum_corner[i][1] / sum_corner[4]] for i in range(4)]
        averaged_corners.append(avg_corner)
        unique_ids.append([id_val])

    # Construct the output dictionary
    output_dict = {
        "cornersMarkers": averaged_corners,
        "idMarkers": unique_ids
    }

    return output_dict

# Helper wrapper function to handle exceptions appropriately in a multiprocessing pool
def try_to_detect(*args):
    try:
        detect_board_in_video(*args[0])
    except Exception as e:
        logger.error(e)
        raise e

# Helper function to define the global variable `counter` in the multiprocessing pool to update the progress bar
def init(c=None):
    global counter
    counter = c

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description="Create canonical model from scan")
    parser.add_argument("--folder", type=Path, required=True, help="Folder with UV texture")
    parser.add_argument("--board", type=Path, required=True, help="Path to the board JSON configuration")
    parser.add_argument("--debug", action="store_true", help="Save 2D detections debug frames")

    args = parser.parse_args()

    logger.info("==============================")
    logger.info("== 1. Starting UV detection ==")
    logger.info("==============================")

    logger.info(f"[INFO] Loading board from {args.board.absolute()}")

    # Find all video files in the folder and sort in ascending order
    image_files = list(args.folder.glob("skin.jpg"))
    image_files.sort()

    if not image_files:
        logger.error(f"No images found in {args.folder}")
        return

    logger.info(f"[INFO] Found {len(image_files)} image/s")

    # Create output folder alongside the input folder
    out_folder = args.folder / f"uv_detections_{args.board.stem}"
    out_folder.mkdir(exist_ok=True)

    for image_file in image_files:
        detect_board_in_uv(image_file, args.board, out_folder, debug=args.debug)

if __name__ == "__main__":
    main()
