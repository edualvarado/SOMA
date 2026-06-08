"""
This script makes the 3D tracking from the suit (v1, v2) from 2D detections and saves the results to a JSON file.

---

JSON structure (detections_corners_markers and detections_corners_charuco):

{
  "0": {
        "1" : {
                "corners_markers": [
                  [[X,Y],[X,Y],[X,Y],[X,Y]]],
                "id_markers": [["ID"]]
                }
        }
}

JSON structure (detections_corners_charuco):

{
  "0": {
        "1" : {
                "corners_charuco": [
                  [X,Y]],
                "id_charuco": [["ID"]]
                }
        }
}

First key is frame idx. Second key is video idx.

---

JSON structure (resolutions):

{0: (4112, 3008), 1: (4112, 3008), 2: (4112, 3008)...}

Key is camera idx. Value is (width, height).

---

JSON structure (marker_id_to_video_indices):

{177: [1, 4], 372: [1, 4, 6],...}

Key is marker id. Value is a list of video indices where the marker appears.

---

JSON structure (triangulation_attempts):

If a marker appears in e.g. 3 videos, we have 3 different combinations of videos. Therefore, this dictionary contains
all the possible combinations of videos and the corresponding 3D points.

{
       ((1, np.int64(1)), (4, np.int64(46))): array([[2.5180147 , 1.3015634 , 0.40403748],
       [2.5420702 , 1.30529   , 0.3851208 ],
       [2.5552015 , 1.2898782 , 0.39628217],
       [2.5311196 , 1.2847252 , 0.41451216]], dtype=float32),
       ((1, np.int64(1)), (6, np.int64(49))): array([[2.513219  , 1.2969337 , 0.39813876],
       [2.5371382 , 1.3006902 , 0.3791093 ],
       [2.5506842 , 1.2855623 , 0.39074874],
       [2.527123  , 1.2805809 , 0.40943533]], dtype=float32),
       ((4, np.int64(46)), (6, np.int64(49))): array([[2.5097268 , 1.2955296 , 0.38942823],
       [2.5336306 , 1.2992561 , 0.37010616],
       [2.5477095 , 1.2838926 , 0.38233405],
       [2.5244675 , 1.2789495 , 0.4018405 ]], dtype=float32)
}

They key is (video_index_1, corner_idx_1), (video_index_2, corner_idx_2) and the value is the 3D points.

---

{
  "0": {
    "372": [
      [
        X,
        Y,
        Z
      ],
      [
        X,
        Y,
        Z
      ],
      [
        X,
        Y,
        Z
      ],
      [
        X,
        Y,
        Z
      ]
    ],
    "427": [
      [
        X,
        Y,
        Z
      ],
      [
        X,
        Y,
        Z
      ],
      [
        X,
        Y,
        Z
      ],
      [
        X,
        Y,
        Z
      ]
    ]
    }
}

---

Example usage:
    python tracking_3D.py --folder detections

- The script will look for all the 2D detections in a folder and create the triangulated markers
- For each camera, this script will look for the corners_charuco_streamXXX and corners_markers_streamXXX detections
- The results will be saved to a JSON.
- The JSON file will contain a dictionary with KEY: frame index and VALUES: dicts of 3D detections with ids.
- We define 3d_corners_markers and 3d_corners_charuco as the corners detected by the board and the corners detected by the ChArUco board, respectively.
- corners_markers is (N,4,3) and corners_charuco is (N,3).
- We define 3d_id_markers and 3d_id_charuco as the ids of the corners detected by the board and the corners detected by the ChArUco board, respectively.
- Both, id_markers and id_charuco, are lists of integers.
- If no corners were detected in a frame, the frame index will not be present in the dictionary.

Run the script with --help to see all available options.
"""

import argparse
import itertools
import cv2
import matplotlib.pyplot as plt
import numpy as np
import toolkit.loading as tl
import json
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from scipy.optimize import least_squares
import multiprocessing
import os

MEDIAN_THRESHOLD = 0.1
Z_SCORE_THRESHOLD = 1


def remove_duplicate_camera_indices(marker_id_to_video_indices):
    """
    We remove the views if there is the same ID more than once in one camera. Example:

    We convert:

    {177: [1, 4], 372: [1, 1, 6],...}

    into:

    {177: [1, 4], 372: [6],...}
    """

    # New dictionary to store the modified data
    marker_id_to_video_indices_modified = {}

    for marker_id, video_indices in marker_id_to_video_indices.items():
        # Count occurrences of each video index
        video_index_counts = {}
        for video_index in video_indices:
            video_index_counts[video_index] = video_index_counts.get(video_index, 0) + 1

        # Filter out video indices that appear more than once
        filtered_video_indices = [video_index for video_index, count in video_index_counts.items() if count == 1]

        # Add the filtered list to the modified dictionary
        marker_id_to_video_indices_modified[marker_id] = filtered_video_indices

    return marker_id_to_video_indices_modified


def save_result_to_json(frame_to_corners, path):
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError

    with open(path, "w") as file:
        json.dump(frame_to_corners, file, indent=2, default=default)
    logger.info(f"[INFO] Saved {len(frame_to_corners)} frames to {path}")


def objective(avg_points, valid_corner_keys, detections, calibs):
    # Reshape to have 3 columns
    avg_points = avg_points.reshape(-1, 3)  # (N, 3)

    # Compute the reprojection error
    reprojection_errors = []

    for video_index, corner_index in valid_corner_keys:
        # logger.info(f"[INFO] OBJ - Computing reprojection error for video {video_index} and corner_index {corner_index}")
        calibration = calibs[video_index]
        intrinsic, pose = calibration["intrinsic"], calibration["extrinsic"]

        K = intrinsic.matrix
        rvec, tvec = pose.rvec, pose.tvec

        corners_markers = detections[video_index]["corners_markers"]
        corner = corners_markers[corner_index]

        # Reproject the 3D point into the camera frame
        reprojected_points = cv2.projectPoints(avg_points, rvec, tvec, K, None)[0].squeeze()

        # Minimize the error between the reprojected 3D point into the camera AND the actual 2D detection
        reprojection_error = reprojected_points - corner
        reprojection_errors.append(reprojection_error.flatten())

    reprojection_errors = np.concatenate(reprojection_errors)

    return reprojection_errors


def process_frame(task_data):
    """
    Performs 3D triangulation for all visible markers in a single frame.
    Designed to be called by a multiprocessing Pool.

    Args:
        task_data (tuple): A tuple containing all necessary data for one frame:
            (frame_idx, detections_for_frame, calibs, projection_matrices,
             args, MEDIAN_THRESHOLD, Z_SCORE_THRESHOLD)

    Returns:
        tuple: A tuple containing (frame_idx, frame_results_dict).
               Returns (frame_idx, {}) if no markers could be triangulated.
    """
    # 1. Unpack all the data for this task
    (frame_idx,
     detections,
     calibs,
     projection_matrices,
     args,
     MEDIAN_THRESHOLD,
     Z_SCORE_THRESHOLD) = task_data

    # This dictionary will store the results for ONLY this frame
    frame_results = {}

    # 2. Create dict with marker ids and videos where they appear (for this frame)
    marker_id_to_video_indices = {}
    for video_idx, corners_data in detections.items():
        for marker_id in corners_data["id_markers"]:
            marker_id = marker_id.item()
            if marker_id not in marker_id_to_video_indices:
                marker_id_to_video_indices[marker_id] = []
            marker_id_to_video_indices[marker_id].append(video_idx)

    marker_id_to_video_indices = remove_duplicate_camera_indices(marker_id_to_video_indices)

    # 3. Loop through each unique marker found in this frame
    # Note: The inner tqdm is removed as it would create garbled output in parallel mode.
    # The main process will have a single tqdm for frames.
    for marker_id, video_indices in marker_id_to_video_indices.items():
        if len(video_indices) < 2:
            continue

        triangulation_attempts = {}
        # 4. Triangulate using all combinations of camera pairs
        for video_index_1, video_index_2 in itertools.combinations(video_indices, 2):
            P1, P2 = projection_matrices[video_index_1], projection_matrices[video_index_2]

            corners_markers_1 = detections[video_index_1]["corners_markers"]
            id_markers_1 = detections[video_index_1]["id_markers"]
            corners_markers_2 = detections[video_index_2]["corners_markers"]
            id_markers_2 = detections[video_index_2]["id_markers"]

            corner_indices_1 = np.where(id_markers_1 == marker_id)[0]
            corner_indices_2 = np.where(id_markers_2 == marker_id)[0]
            # corner_indices_1 = np.where(id_markers_1.flatten() == marker_id)[0]
            # corner_indices_2 = np.where(id_markers_2.flatten() == marker_id)[0]

            for corner_idx_1, corner_idx_2 in itertools.product(corner_indices_1, corner_indices_2):
                corner_1, corner_2 = corners_markers_1[corner_idx_1], corners_markers_2[corner_idx_2]
                points_4d = cv2.triangulatePoints(P1, P2, corner_1.T, corner_2.T)
                points_3d = (points_4d[:3] / points_4d[3]).T

                attempt_key = ((video_index_1, corner_idx_1), (video_index_2, corner_idx_2))

                triangulation_attempts[attempt_key] = points_3d

        if not triangulation_attempts:
            continue

        # 5. Outlier Rejection
        all_triangulated_points = np.array(list(triangulation_attempts.values()))
        if len(all_triangulated_points) < 3:  # Need at least 3 pairs for robust median/std
            continue

        # Median-based filtering
        median_points = np.median(all_triangulated_points, axis=0)
        distances_to_median = np.linalg.norm(all_triangulated_points - median_points, axis=-1)
        median_mask = np.any(distances_to_median < MEDIAN_THRESHOLD, axis=1)
        filtered_points = all_triangulated_points[median_mask]

        if filtered_points.size > 0:
            # Z-score-based filtering
            median_filtered = np.median(filtered_points, axis=0)
            std_filtered = np.std(filtered_points, axis=0)
            epsilon = 1e-8
            z_scores = (filtered_points - median_filtered) / (std_filtered + epsilon)
            combined_z_scores = np.linalg.norm(z_scores, axis=-1)
            outliers_mask = np.any(combined_z_scores > Z_SCORE_THRESHOLD, axis=1)
            filtered_points = filtered_points[~outliers_mask]

        # Filter the attempts dictionary
        filtered_attempt_keys = [key for key, value in zip(triangulation_attempts, all_triangulated_points) if
                                 value in filtered_points]
        filtered_attempts = {key: triangulation_attempts[key] for key in filtered_attempt_keys}

        # NEW
        if not filtered_attempts:
            # This logs if verbose mode is on, helping you debug your filter thresholds
            if args.verbose:
                logger.debug(f"For marker {marker_id}, all {len(all_triangulated_points)} triangulation attempts were filtered out. Skipping.")
            continue  # Skip to the next marker

        valid_corner_keys = []
        for corner_key_1, corner_key_2 in filtered_attempts.keys():
            if corner_key_1 not in valid_corner_keys:
                valid_corner_keys.append(corner_key_1)
            if corner_key_2 not in valid_corner_keys:
                valid_corner_keys.append(corner_key_2)

        avg_points = np.mean(filtered_points, axis=0)

        if len(filtered_attempts) < 3:
            continue


        """
        # 6. Optimization (Conditional)
        avg_points = np.mean(final_filtered_points, axis=0)

        # Get the keys for the valid corners that survived filtering
        valid_triangulation_keys = np.array(list(triangulation_attempts.keys()), dtype=object)[median_mask][
            ~outliers_mask]

        valid_corner_keys = list(set(key for pair in valid_triangulation_keys for key in pair))
        # Ensure all keys are hashable by converting numpy arrays to tuples
        # valid_corner_keys = list(
        #     set(
        #         tuple(key) if isinstance(key, np.ndarray) else key
        #         for pair in valid_triangulation_keys
        #         for key in pair
        #     )
        # )

        if not valid_corner_keys:
            continue
            
        # Default to the simple average
        best_points = avg_points
        
        """

        # (n, 2) where n = 4 * number of attempts to triangulate one ID. Without the reshape, it would be 1D.
        initial_reprojection_errors = objective(avg_points.flatten(), valid_corner_keys, detections, calibs).reshape(-1,
                                                                                                                     2)
        # Errors are defined by 2D vectors which we calculate the norm to estimate the magnitude of the error
        initial_reprojection_error = np.linalg.norm(initial_reprojection_errors, axis=-1)  # (1, n)

        # Wrap the objective function to include additional parameters
        result = least_squares(
            lambda x: objective(x, valid_corner_keys, detections, calibs),
            avg_points.flatten()
        )

        # Optimized 3D positions
        final_points = result.x.reshape(-1, 3)

        final_reprojection_error = objective(final_points.flatten(), valid_corner_keys, detections, calibs).reshape(-1,
                                                                                                                    2)
        final_reprojection_error = np.linalg.norm(final_reprojection_error, axis=-1)

        # Save final point, whether is the optimized one or not
        best_points = final_points if final_reprojection_error.mean() < initial_reprojection_error.mean() else avg_points

        difference = final_points - avg_points

        """
        try:
            initial_reprojection_errors = objective(avg_points.flatten(), valid_corner_keys, detections, calibs)
            initial_mean_error = np.linalg.norm(initial_reprojection_errors.reshape(-1, 2), axis=-1).mean()

            # ONLY run expensive optimization if the initial error is high
            REPROJECTION_ERROR_THRESHOLD = 0.0  # (pixels) - can be passed in args
            if initial_mean_error > REPROJECTION_ERROR_THRESHOLD:
                result = least_squares(
                    lambda x: objective(x, valid_corner_keys, detections, calibs),
                    avg_points.flatten(),
                    max_nfev=20  # Limit iterations to prevent getting stuck
                )
                final_points = result.x.reshape(-1, 3)
                final_reprojection_errors = objective(final_points.flatten(), valid_corner_keys, detections, calibs)
                final_mean_error = np.linalg.norm(final_reprojection_errors.reshape(-1, 2), axis=-1).mean()

                if final_mean_error < initial_mean_error:
                    best_points = final_points
        except Exception as e:
            # Catch potential errors in optimization or key finding
            if args.verbose:
                logger.warning(f"Could not optimize marker {marker_id} in frame {frame_idx}: {e}")
        """

        if best_points is not None:  # Assuming best_points can be None
            frame_results[str(marker_id)] = best_points

        """
        # 7. Store result for this marker (use string for JSON key)
        frame_results[str(marker_id)] = best_points
        """

    # 8. Return the frame index and its results
    return frame_idx, frame_results


def main():
    parser = argparse.ArgumentParser(description="Triangulate marker positions from multiple cameras")
    parser.add_argument("--folder", type=Path, help="The folder containing the detection data")
    parser.add_argument("--frames_2D_int", type=int, required=True,
                        help="Number of missing frames required to interpolate 2D")
    parser.add_argument("--max_frames", type=int, default=1, help="Maximum number of frames to process")
    parser.add_argument("--keyword", type=str, help="The keyword to use to filter the detection data", default="stream")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # --- 1. SETUP (same as before) ---
    logger.info("================================")
    logger.info("== 3. Starting 3D tracking V2 ==")
    logger.info("================================")
    detection_folder = args.folder / f"detections_charuco-suit/2D-interpolated-N{args.frames_2D_int}"
    tracking_folder = args.folder / f"tracking_charuco-suit/triangulation/"
    tracking_folder.mkdir(parents=True, exist_ok=True)

    detections_corners_markers = tl.load_corners_markers(detection_folder, keyword=args.keyword)
    resolutions = tl.load_resolutions(args.folder, keyword=args.keyword)
    calibs = tl.load_calibrations(args.folder / "cameras.calib", resolutions[0][0])

    # --- 2. PRE-CALCULATION (Optimization) ---
    logger.info("[INFO] Pre-calculating projection matrices...")
    projection_matrices = {
        cam_idx: cal.get("intrinsic").matrix @ cal.get("extrinsic").matrix[:3, :]
        for cam_idx, cal in calibs.items()
    }

    # --- 3. PREPARE TASKS FOR PARALLEL PROCESSING ---
    tasks = []
    # Use args.max_frames to limit the number of frames
    frame_indices = sorted(detections_corners_markers.keys())[:args.max_frames]

    for frame_idx in frame_indices:
        detections_for_frame = detections_corners_markers[frame_idx]
        task_data = (
            int(frame_idx),
            detections_for_frame,
            calibs,
            projection_matrices,
            args,
            MEDIAN_THRESHOLD,
            Z_SCORE_THRESHOLD
        )
        tasks.append(task_data)

    # --- 4. EXECUTE IN PARALLEL ---
    all_results = {}

    num_cores = os.cpu_count()

    logger.info(f"[INFO] Starting parallel processing on {num_cores} cores for {len(tasks)} frames...")

    with multiprocessing.Pool(processes=num_cores) as pool:
        # Use imap to get results as they are completed, with a progress bar
        for frame_idx, frame_results in tqdm(pool.imap_unordered(process_frame, tasks), total=len(tasks)):
            if frame_results:  # Only add frames that have valid results
                all_results[str(frame_idx)] = frame_results
            else:
                logger.warning(f"No results for frame {frame_idx}.")

    # --- 5. SAVE FINAL RESULT ---
    logger.info(f"[INFO] Processing complete. Found triangulations for {len(all_results)} frames.")
    output_path = tracking_folder / "triangulation_markers.json"
    # Ensure keys are sorted for consistent output
    sorted_results = dict(sorted(all_results.items(), key=lambda item: int(item[0])))
    save_result_to_json(sorted_results, output_path)


if __name__ == "__main__":
    main()