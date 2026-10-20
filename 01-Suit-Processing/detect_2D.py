"""
This script makes the 2D detection from the suit (v1, v2) from videos and saves the results to a JSON file.

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
    python detect_2D.py --board board.json --folder videos --parallel --static --debug --mask --setup 1

- The script will look for all MP4 videos in the folder and process them in parallel.
- The results will be saved to a JSON file with the same name as the video file.
- The JSON file will contain a dictionary with KEY: frame index and VALUES: dicts of detections with ids.
- We define corners_markers and corners_charuco as the corners detected by the board and the corners detected by the ChArUco board, respectively.
- corners_markers is (N,4,2) and corners_charuco is (N,2).
- We define id_markers and id_charuco as the ids of the corners detected by the board and the corners detected by the ChArUco board, respectively.
- Both, id_markers and id_charuco, are lists of integers.
- If no corners were detected in a frame, the frame index will not be present in the dictionary.

Run the script with --help to see all available options.
"""

import itertools
import argparse
import json
import multiprocessing
import time
import random
import toolkit.loading as tl
import re
import os
import cv2
import numpy as np
from multiprocessing import Pool
from pathlib import Path
from typing import Callable
from caliboards import BoardFactory
from loguru import logger
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

def detect_suit_in_video(
        video_path: Path,
        mask_path: Path,
        board_path: Path,
        output_folder: Path,
        calibrations: dict,
        max_frame_count: int,
        debug: bool,
        apply_mask: bool,
        setup: int,
        verbose: bool,
        update_fn: Callable = None
):
    """
    Detect a `board` (or pattern in the suit) in a video and save the results to a JSON file.

    Args:
        video_path: Path to the video file.
        mask_path: Path to the mask file.
        board_path: Path to the board JSON file.
        output_folder: Path to the output folder.
        calibrations: Dict containing video calibrations,
        frames_count: Max. number of frames to process,
        debug: Whether to save debug images.
        setup: Parameter setup
        apply_mask: Whether to apply the mask to the video frames.
        update_fn: Function to call to update the progress bar.
    """

    logger.info(f"[INFO] Processing video {video_path.name}")

    # Define the output file paths for the ChArUco detection .json (for saving)
    detection_corners_charuco_file_path = output_folder / f"corners_charuco_{video_path.stem}.json"

    # Define the output file paths for the ArUco marker detection .json (for saving)
    detection_corners_markers_file_path = output_folder / f"corners_markers_{video_path.stem}.json"

    # Load the board from the JSON file
    board = BoardFactory.from_json(board_path)

    # Open the video file
    cap = cv2.VideoCapture(str(video_path))

    mcap = None
    if apply_mask:
        mcap = cv2.VideoCapture(str(mask_path))

    # Dictionary to store the detected charuco corners and ids for each frame
    frame_to_corners_charuco = {}

    # Dictionary to store the detected marker corner and ids for each frame
    frame_to_corners_markers = {}

    # Loop through the video frames
    frame_idx = 0
    while cap.isOpened():
        ret_cap, frame = cap.read()
        if not ret_cap:
            logger.error(f"[ERROR] Video {str(video_path)} could not be read.")
            break

        # Handle mask processing only if apply_mask is True
        if apply_mask:
            ret_mask, mask = mcap.read()
            if not ret_mask:
                logger.error(f"[ERROR] Mask {str(mask_path)} could not be read.")
                break

            # Find bounding box of mask
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                continue

            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            frame = frame[y: y + h, x: x + w]

        # If it's the static scan, we just do one frame (if frames_count = 1, we will do just frame 0)
        if frame_idx == max_frame_count:
            logger.info(f"[INFO] Maximum number of frames reached: {max_frame_count}.")
            break

        # Initialize empty lists for the corners and ids for the current frame
        frame_corners_charuco_list = []
        frame_id_charuco_list = []

        # Initialize empty lists for the markers and ids for the current frame
        frame_corners_markers_list = []
        frame_id_markers_list = []

        # Detect ChArUco corners (N,2) w/ IDs AND markers (N,4,2) w/ IDs
        corners_charuco, id_charuco, corners_markers, id_markers = board.detect_all(frame, frame_idx, video_path, calibrations,
                                                                                    draw=debug, setup=setup, verbose=verbose)

        # If corners were detected, save them to the dictionary
        if corners_charuco is not None:
            corners_charuco_np = np.concatenate(corners_charuco, axis=0)
            frame_dict_corners_charuco = dict(corners_charuco=corners_charuco_np.tolist(), id_charuco=id_charuco.tolist())
            frame_to_corners_charuco[frame_idx] = frame_dict_corners_charuco

        """Explanation
        - corners_charuco = [array([[100.2, 200.3], [150.8, 220.4]]), array([[300.5, 400.1]])]
        - ids_charucos = [[1, 2], [3]]
        - corners_np = array([[100.2, 200.3], [150.8, 220.4], [300.5, 400.1]])
        - frame_dict = {"corners": [[100.2, 200.3], [150.8, 220.4], [300.5, 400.1]],"id": [1, 2, 3]}
        - frame_to_corners_charuco = {2: {"corners": [[100.2, 200.3], [150.8, 220.4], [300.5, 400.1]],"id": [1, 2, 3]}}       
        """

        # If ArUcos were detected, save them to the dictionary
        if corners_markers is not None and len(corners_markers) > 0:
            corners_markers_np = np.concatenate(corners_markers, axis=0)

            # Add the offset to the corners
            if apply_mask:
                offset = np.array([x, y])
                corners_markers_np += offset

            frame_dict_corners_markers = dict(corners_markers=corners_markers_np.tolist(), id_markers=id_markers.tolist())
            frame_to_corners_markers[frame_idx] = frame_dict_corners_markers

        # ---

        # Pair the elements from the two lists (ChArUco corners) together and sort
        paired_list_corners_charuco = list(zip(frame_corners_charuco_list, frame_id_charuco_list))
        sorted_pairs_corners_charuco = sorted(paired_list_corners_charuco, key=lambda entry: entry[1])

        # Pair the elements from the two lists (Marker corners) together and sort
        paired_list_corners_markers = list(zip(frame_corners_markers_list, frame_id_markers_list))
        sorted_pairs_corners_markers = sorted(paired_list_corners_markers, key=lambda entry: entry[1])

        """ Explanation
        - zip(frame_corners_charuco_list, frame_id_charuco_list) produces an iterator of tuples.
        - paired_list_corners_charuco = [("corner1", 3), ("corner2", 1), ("corner3", 2)] is a list of tuples
        - sorted_pairs_corners_charuco = [("corner2", 1), ("corner3", 2), ("corner1", 3)]
        """

        # ---

        # Sorting ChArUco corners
        if sorted_pairs_corners_charuco:
            # Unzip the sorted pairs
            frame_corners_charuco_list, frame_id_charuco_list = zip(*sorted_pairs_corners_charuco)

            # Convert the tuples back to lists
            frame_corners_charuco_list = list(frame_corners_charuco_list)
            frame_id_charuco_list = list(frame_id_charuco_list)

            """ Explanation
            - if sorted_pairs_corners_charuco = [(1, 'a'), (2, 'b'), (3, 'c')], then:
            - frame_charucos_list = (1, 2, 3) -> [1, 2, 3]
            - frame_ids_charucos_list = ('a', 'b', 'c') -> ['a', 'b', 'c']         
            """

        # Sorting Markers corners
        if sorted_pairs_corners_markers:
            # Unzip the sorted pairs
            frame_corners_markers_list, frame_id_markers_list = zip(*sorted_pairs_corners_markers)

            # Convert the tuples back to lists
            frame_corners_markers_list = list(frame_corners_markers_list)
            frame_id_markers_list = list(frame_id_markers_list)

        # ---

        # Save the corners and ids for the current frame
        if frame_corners_charuco_list and frame_id_charuco_list:

            # Create dictionary with corners and ids (can be duplicated)
            frame_dict_corners_charuco_unzip = dict(corners_charuco=frame_corners_charuco_list,
                                                    id_charuco=frame_id_charuco_list)

            # Detect duplicates id and do average for corners
            frame_dict_corners_charuco_avg = average_corners_charuco(frame_dict_corners_charuco_unzip)
            frame_to_corners_charuco[frame_idx] = frame_dict_corners_charuco_avg

        # Save the corners and ids for the current frame
        if frame_corners_markers_list and frame_id_markers_list:
            # Create dictionary with corners and ids (can be duplicated)
            frame_dict_corners_markers_unzip = dict(corners_markers=frame_corners_markers_list,
                                                    id_markers=frame_id_markers_list)

            # Detect duplicates id and do average for corners
            frame_dict_corners_markers_avg = average_corners_markers(frame_dict_corners_markers_unzip)
            frame_to_corners_markers[frame_idx] = frame_dict_corners_markers_avg

        # ---

        # Call the update function to update the progress bar
        if update_fn is not None:
            x = 10  # Update every 10 iterations
            if frame_idx % x == 0:
                update_fn()

        # Update the counter to update the progress bar in the multiprocessing pool
        if counter is not None:
            with counter.get_lock():
                counter.value += 1

        # if debug and corners is not None:
        if debug:
            save_debug_image(video_path, frame_idx, board_path, frame, setup)

        frame_idx += 1

    cap.release()

    save_result_to_json(frame_to_corners_charuco, detection_corners_charuco_file_path)
    save_result_to_json(frame_to_corners_markers, detection_corners_markers_file_path)
    logger.info(f"[INFO] Finished processing {video_path.name}")

    return frame_to_corners_charuco, frame_to_corners_markers

# TODO: Careful with this, it could lead to wrong 2D detections. Debug!
def average_corners_charuco(input_dict):
    # Create a dictionary to accumulate the sum of corners and count of occurrences for each id
    id_corner_sum = defaultdict(lambda: [0, 0])
    id_count = defaultdict(int)

    corners = input_dict["corners_charuco"]
    ids = input_dict["id_charuco"]

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
        "corners_charuco": averaged_corners,
        "id_charuco": unique_ids
    }

    return output_dict

# TODO: Careful with this, it could lead to wrong 2D detections. Debug!
def average_corners_markers(input_dict):
    # Create a dictionary to accumulate the sum of corners and count of occurrences for each id
    id_corner_sum = defaultdict(lambda: [[0, 0], [0, 0], [0, 0], [0, 0], 0])
    id_count = defaultdict(int)

    corners = input_dict["corners_markers"]
    ids = input_dict["id_markers"]

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
        "corners_markers": averaged_corners,
        "id_markers": unique_ids
    }

    return output_dict

def save_result_to_json(frame_to_corners, path):
    with open(path, "w") as file:
        json.dump(frame_to_corners, file, indent=2)

def save_debug_image(video_file, frame_idx, board_path, frame, setup):
    debug_folder = video_file.parent / "debug_frames"
    debug_folder.mkdir(exist_ok=True)
    path = debug_folder / f"{video_file.stem}_fr{frame_idx}_{board_path.stem}_setup{setup}.jpg"
    cv2.imwrite(str(path), frame)
    logger.info(f"[INFO] Saved frame {frame_idx} to {path.absolute()}")

# Helper wrapper function to handle exceptions appropriately in a multiprocessing pool
def try_to_detect(*args):
    try:
        detect_suit_in_video(*args[0])
    except Exception as e:
        logger.error(f"[ERROR] {e}")
        raise e

# Helper function to define the global variable `counter` in the multiprocessing pool to update the progress bar
def init(c=None):
    global counter
    counter = c

# Convert any string to bools
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
    parser = argparse.ArgumentParser(description="Detect calibration board in videos")
    parser.add_argument("--modulus", type=int, default=None, help="Modulus of video indexes to process")
    parser.add_argument("--board", type=Path, required=True, help="Path to the board JSON configuration")
    parser.add_argument("--folder", type=Path, required=True, help="Folder with MP4 videos")
    parser.add_argument("--parallel", action="store_true", help="Process videos in parallel")
    parser.add_argument("--static", action="store_true", help="Detect on first frame")
    parser.add_argument("--debug", action="store_true", help="Save 2D detections debug frames")
    parser.add_argument("--max_frames", type=int, default=1, help="Maximum number of frames to process")
    parser.add_argument("--mask", action="store_true", help="Apply mask to video frames")
    parser.add_argument("--setup", type=int, default=1, help="Parameter setup for 2D detections")
    parser.add_argument("--remainder", type=int, default=0, help="Remainder of the modulus of video indexes to process")
    parser.add_argument("--keyword", type=str, default=None, help="Keyword that must be present in the video filename")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    logger.info("==============================")
    logger.info("== 1. Starting 2D detection ==")
    logger.info("==============================")

    logger.info(f"[INFO] Loading board from {args.board.absolute()}")
    logger.info(f"[INFO] Loading videos from {args.folder.absolute()}")
    logger.info(f"[INFO] Modulus: {args.modulus}")
    logger.info(f"[INFO] Parallel: {args.parallel}")
    logger.info(f"[INFO] Debugging frames: {args.debug}")
    logger.info(f"[INFO] Parameter setup: {args.setup}")
    if args.max_frames:
        logger.info(f"[INFO] Limited frames only: {args.max_frames}")

    # Find all video files in the folder and sort in ascending order
    video_files = list(args.folder.glob("*.mp4"))
    video_files.sort()

    # Filter videos based on keyword and modulus
    if args.keyword:
        video_files = [path for path in video_files if args.keyword in path.name]
    if args.modulus is not None:
        video_files = [path for index, path in enumerate(video_files) if index % args.modulus == args.remainder]

    if not video_files:
        logger.error(f"[ERROR] No videos found in {args.folder}")
        return

    # Retrieve camera resolutions and calibrations
    resolutions = tl.load_resolutions(args.folder)
    width = resolutions[0][0]
    calibrations = {}
    calibs = tl.load_calibrations(args.folder / "cameras.calib", width)
    logger.info(f"[INFO] Found {len(video_files)} video/s and {len(calibs)} calibrations")

    for video in video_files:
        file_name = video.name # "stream001.mp4"

        # Extract the index (digits) from the file name
        match = re.search(r'(\d+)', file_name)
        if match:
            index_with_zeros = match.group(1)  # "001"
            video_index = int(index_with_zeros)  # "1"
        else:
            video_index = None  # No index found

        # TODO: Review if this is correct or we need to convert intrinsics
        calibrations[file_name] = {
                "K1": calibs[video_index]["intrinsic"].matrix,
                "d1": calibs[video_index]["intrinsic"].distortion,
        }

    # =========================================================================
    # NEW: OPTIMIZATION START: Determine Available Cores Correctly
    # =========================================================================
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        available_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
        logger.info(f"[OPTIMIZATION] Running in Slurm. Cores allocated: {available_cores}")
    else:
        # Fallback for local machine
        try:
            available_cores = len(os.sched_getaffinity(0))
        except AttributeError:
            available_cores = multiprocessing.cpu_count()
        logger.info(f"[OPTIMIZATION] Running locally. Cores detected: {available_cores}")

    if args.parallel:
        # If we have 32 cores and 10 videos, each video gets 3 threads.
        # If we have 4 cores and 100 videos, each video gets 1 thread.
        # We ensure at least 1 thread per process.
        
        num_videos = len(video_files)
        if num_videos >= available_cores:
            threads = 1 # Saturation: 1 thread per video
        else:
            threads = max(1, available_cores // num_videos)
        
        cv2.setNumThreads(threads)
        logger.info(f"[INFO] Using {threads} OpenCV threads per video process.")
    # =========================================================================

    total_frames = 0
    latest_frames_count = 0
    frames_count = 0
    flag_different_frame_count = False
    for path in tqdm(video_files, desc="Counting frames"):
        total_frames += int(cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FRAME_COUNT))
        frames_count = int(cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FRAME_COUNT))
        if frames_count != latest_frames_count:
            if not flag_different_frame_count:
                latest_frames_count = frames_count
                flag_different_frame_count = True

    max_frame_count = frames_count
    if args.max_frames:
        max_frame_count = args.max_frames

    logger.info(f"[INFO] Processing {len(video_files)} video/s for a total of {total_frames} frames")

    out_folder = args.folder / f"detections_{args.board.stem}"
    out_folder.mkdir(exist_ok=True)

    pbar = tqdm(total=total_frames, unit="frames", colour="green", desc="Processing 2D Detections over frames")

    if not args.parallel:
        init()
        logger.info(f"[INFO] Using single thread")

        def update_fn(pbar=pbar):
            pbar.update(10)

        for video_file in tqdm(video_files):
            mask_dir = Path(video_file).parent / "segmentations"
            if os.path.exists(mask_dir):
                mask_path = Path(video_file).parent / "segmentations" / video_file.name
            else:
                mask_path = video_file.name
            detect_suit_in_video(video_file, mask_path, args.board, out_folder, calibrations, max_frame_count,
                                 args.debug, args.mask, args.setup, args.verbose, update_fn)

        exit()
    else:
        counter_cpu = multiprocessing.Value("i", 0)
        
        # Use the allocated core count, NOT cpu_count()
        num_cores = available_cores 
        logger.info(f"[INFO] Using multiple threads - {num_cores} worker processes")

        mask_dir = Path(video_files[0]).parent / "segmentations"
        if os.path.exists(mask_dir):
            mask_files = [Path(video_file).parent / "segmentations" / video_file.name for video_file in video_files]
        else:
            mask_files = [Path(video_file).parent / video_file.name for video_file in video_files]

        tasks = [(video_file, mask_file, args.board, out_folder, calibrations, max_frame_count,
                  args.debug, args.mask, args.setup, args.verbose)
                 for video_file, mask_file in zip(video_files, mask_files)]

        with Pool(initializer=init, initargs=(counter_cpu,), processes=num_cores) as pool:
            result = pool.map_async(try_to_detect, tasks)

            while not result.ready():
                pbar.n = counter_cpu.value
                pbar.refresh()
                time.sleep(0.1)

            logger.info("[INFO] 2D Detections finished")

            pbar.close()

    """ 
    # Estimating threads
    if args.parallel:
        # threads = multiprocessing.cpu_count() // len(video_files) or 1
        threads = max(1, (multiprocessing.cpu_count() * 2) // len(video_files))
        cv2.setNumThreads(threads)
        logger.info(f"[INFO] Using {threads} threads for each video processing in OpenCV")

    # Count total frames
    total_frames = 0
    latest_frames_count = 0
    frames_count = 0
    flag_different_frame_count = False
    for path in tqdm(video_files, desc="Counting frames"):
        total_frames += int(cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FRAME_COUNT))
        frames_count = int(cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FRAME_COUNT))
        if frames_count != latest_frames_count:
            if not flag_different_frame_count:
                logger.info(f"[INFO] Video {path.name} has {frames_count} frames")
                latest_frames_count = frames_count
                flag_different_frame_count = True
            else:
                logger.error(f"[ERROR] Mismatch - Video {path.name} has {frames_count} frames, while the previous video has {latest_frames_count} frames")

    # Process only N frames
    max_frame_count = frames_count
    if args.max_frames:
        max_frame_count = args.max_frames

    logger.info(f"[INFO] Processing {len(video_files)} video/s for a total of {total_frames} frames")

    # Create output folder with detection file
    out_folder = args.folder / f"detections_{args.board.stem}"
    out_folder.mkdir(exist_ok=True)

    # Setup centralized progress bar
    pbar = tqdm(total=total_frames, unit="frames", colour="green", desc="Processing 2D Detections over frames")

    # Process videos in a single or multiple threads
    if not args.parallel:
        init()
        logger.info(f"[INFO] Using single thread")

        def update_fn(pbar=pbar):
            pbar.update(x)

        for video_file in tqdm(video_files):
            # Check if segmentations exists. If not, we just take the video's name (will not be used)
            mask_dir = Path(video_file).parent / "segmentations"
            if os.path.exists(mask_dir):
                mask_path = Path(video_file).parent / "segmentations" / video_file.name
            else:
                mask_path = video_file.name
            # logger.debug(f"[DEBUG] Processing {video_file.name} with mask {mask_path.name}")
            detect_suit_in_video(video_file, mask_path, args.board, out_folder, calibrations, max_frame_count,
                                 update_fn, apply_mask=args.mask, setup=args.setup, verbose=args.verbose)

        exit()
    else:
        # Setup multiprocessing pool
        counter_cpu = multiprocessing.Value("i", 0)
        num_cores = multiprocessing.cpu_count()
        logger.info(f"[INFO] Using multiple threads - {num_cores} cores")

        mask_dir = Path(video_files[0]).parent / "segmentations"
        if os.path.exists(mask_dir):
            mask_files = [Path(video_file).parent / "segmentations" / video_file.name for video_file in video_files]
        else:
            mask_files = [Path(video_file).parent / video_file.name for video_file in video_files]

        tasks = [(video_file, mask_file, args.board, out_folder, calibrations, max_frame_count,
                  args.debug, args.mask, args.setup, args.verbose)
                 for video_file, mask_file in zip(video_files, mask_files)]

        with Pool(initializer=init, initargs=(counter_cpu,), processes=num_cores) as pool:
            result = pool.map_async(try_to_detect, tasks)

            while not result.ready():
                pbar.n = counter_cpu.value
                pbar.refresh()
                time.sleep(0.1)

            logger.info("[INFO] 2D Detections finished")

            pbar.close()
    """ 
if __name__ == "__main__":
    main()
