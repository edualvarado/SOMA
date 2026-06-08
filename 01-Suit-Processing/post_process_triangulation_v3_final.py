import json
import matplotlib.pyplot as plt
import argparse
from matplotlib.cm import viridis
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from loguru import logger
import cv2  # OpenCV is needed to create videos from images.
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import shutil  # Import for folder deletion
import numpy as np
import os
from tqdm import tqdm
from scipy.signal import medfilt
from scipy.spatial import KDTree

def save_result_to_json(data_dict, path):
    """Saves a dictionary to JSON, correctly handling NumPy arrays."""
    def default_serializer(o):
        if isinstance(o, np.ndarray):
            return o.tolist() # Convert ndarray to a standard list
        raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')

    with open(path, "w") as out_file:
        json.dump(data_dict, out_file, indent=4, default=default_serializer)
    logger.success(f"Post-processing complete. Filtered data saved to {path}")


def get_longest_run(numbers):
    """Helper function to find the longest run of consecutive integers in a list."""
    if not numbers:
        return 0
    numbers = sorted(list(set(numbers))) # Ensure sorted and unique
    if not numbers:
        return 0
    longest_run = 1
    current_run = 1
    for i in range(1, len(numbers)):
        if numbers[i] == numbers[i-1] + 1:
            current_run += 1
        else:
            longest_run = max(longest_run, current_run)
            current_run = 1
    return max(longest_run, current_run)


def get_all_segments(numbers, min_segment_len):
    """Finds all continuous segments and returns the indices of points belonging to long-enough segments."""
    if not numbers:
        return []
    numbers = sorted(list(set(numbers)))
    if not numbers:
        return []

    valid_indices = set()
    current_segment = [numbers[0]]

    for i in range(1, len(numbers)):
        if numbers[i] == numbers[i - 1] + 1:
            current_segment.append(numbers[i])
        else:
            if len(current_segment) >= min_segment_len:
                valid_indices.update(current_segment)
            current_segment = [numbers[i]]

    # Check the last segment
    if len(current_segment) >= min_segment_len:
        valid_indices.update(current_segment)

    return valid_indices

def calculate_centroid(points):
    """Calculate the centroid of a list of 3D points."""
    return np.mean(points, axis=0)

def calculate_displacement(centroid1, centroid2):
    """Calculate the Euclidean distance between two 3D points."""
    return np.linalg.norm(centroid1 - centroid2)

def z_score_filter(distances, threshold=2.5):
    """Filter out outliers based on z-score."""
    mean = np.mean(distances)
    std_dev = np.std(distances)
    if std_dev == 0:
        return [False] * len(distances)  # No outliers if no variation
    z_scores = np.abs((distances - mean) / std_dev)
    return z_scores > threshold


def filter_globally_unstable_markers(data, min_track_len):
    """STAGE 1: Removes markers whose longest continuous track is shorter than min_track_len."""
    logger.info("STAGE 1: Filtering globally unstable markers...")
    marker_appearances = {}
    frame_keys_int = sorted([int(k) for k in data.keys()])

    for frame_idx in frame_keys_int:
        for marker_id in data[str(frame_idx)].keys():
            if marker_id not in marker_appearances:
                marker_appearances[marker_id] = []
            marker_appearances[marker_id].append(frame_idx)

    ids_to_remove = set()
    for marker_id, frame_indices in tqdm(marker_appearances.items(), desc="Analyzing global track stability"):
        longest_track = get_longest_run(frame_indices)
        if longest_track < min_track_len:
            ids_to_remove.add(marker_id)

    logger.info(f"Identified {len(ids_to_remove)} globally unstable markers.")

    # Create new data object without the removed markers
    processed_data = {}
    for frame_key, frame_data in data.items():
        processed_data[frame_key] = {
            marker_id: points
            for marker_id, points in frame_data.items()
            if marker_id not in ids_to_remove
        }
    return processed_data


def filter_displacement_outliers(data, z_threshold):
    """ OLD STAGE 2: Removes markers that jump unnaturally between consecutive frames."""
    logger.info("STAGE 2: Filtering outliers based on frame-to-frame displacement...")
    processed_data = {}
    previous_centroids = {}

    frame_keys = sorted(data.keys(), key=lambda x: int(x))

    for frame_key in tqdm(frame_keys, desc="Analyzing displacement"):
        frame_data = data[frame_key]

        current_centroids = {
            object_id: calculate_centroid(np.array(points))
            for object_id, points in frame_data.items()
        }

        if not current_centroids:
            processed_data[frame] = {}
            previous_centroids = {}
            continue

        ids_in_frame = list(current_centroids.keys())
        displacements = [
            calculate_displacement(current_centroids[oid], previous_centroids[oid])
            if oid in previous_centroids else 0
            for oid in ids_in_frame
        ]

        is_outlier_mask = z_score_filter(np.array(displacements), z_threshold)

        filtered_frame_data = {}
        for i, object_id in enumerate(ids_in_frame):
            if not is_outlier_mask[i]:
                filtered_frame_data[object_id] = frame_data[object_id]

        processed_data[frame_key] = filtered_frame_data
        previous_centroids = current_centroids

    return processed_data


def smooth_trajectories_temporally(data, window_size):
    """
    NEW STAGE 2: Smooths each marker's trajectory over time to remove flickers.
    This works by applying a median filter to the time series of each corner's coordinates.
    """
    logger.info(f"STAGE 2: Smoothing trajectories with a temporal median filter (window: {window_size})...")

    # window_size must be an odd integer
    if window_size % 2 == 0:
        window_size += 1

    # First, collect all trajectories for each corner of each marker
    trajectories = {}
    frame_keys_int = sorted([int(k) for k in data.keys()])

    for frame_idx in frame_keys_int:
        frame_key = str(frame_idx)
        for marker_id, corners in data[frame_key].items():
            if marker_id not in trajectories:
                # Store as a list of frames and a list of corner points
                trajectories[marker_id] = {'frames': [], 'corners': []}
            trajectories[marker_id]['frames'].append(frame_idx)
            trajectories[marker_id]['corners'].append(corners)

    smoothed_data = {str(k): {} for k in frame_keys_int}

    for marker_id, traj in tqdm(trajectories.items(), desc="Smoothing trajectories"):
        corners_array = np.array(traj['corners'])  # Shape: (num_frames, 4, 3)

        # Apply median filter along the time axis (axis 0) for each coordinate
        # For each of the 4 corners, and each of the 3 coordinates (X,Y,Z)...
        smoothed_corners = np.zeros_like(corners_array)
        for corner_idx in range(4):  # 4 corners
            for coord_idx in range(3):  # X, Y, Z
                time_series = corners_array[:, corner_idx, coord_idx]
                smoothed_corners[:, corner_idx, coord_idx] = medfilt(time_series, kernel_size=window_size)

        # Rebuild the data structure with smoothed points
        for i, frame_idx in enumerate(traj['frames']):
            smoothed_data[str(frame_idx)][marker_id] = smoothed_corners[i]

    return smoothed_data


def filter_motion_outliers(data, distance_threshold, min_neighbors=3):
    """
    NEW STAGE 3: Filters outliers based on motion consensus with neighboring markers.
    """
    logger.info("STAGE 3: Filtering outliers based on local motion consensus...")

    processed_data = {frame: {} for frame in data}
    previous_centroids = {}
    frame_keys = sorted(data.keys(), key=lambda x: int(x))

    for frame_key in tqdm(frame_keys, desc="Analyzing motion consensus"):
        # We need data from the previous frame to calculate displacement
        prev_frame_key = str(int(frame_key) - 1)
        if prev_frame_key not in data:
            # For the first frame, or if there's a gap, keep all markers
            processed_data[frame_key] = data[frame_key]
            continue

        frame_data = data[frame_key]
        prev_frame_data = data[prev_frame_key]

        # Calculate displacements for all markers present in both frames
        displacements = {}
        prev_centroids = {mid: calculate_centroid(np.array(p)) for mid, p in prev_frame_data.items()}

        # Only consider markers present in the previous frame for building the KD-Tree
        valid_prev_ids = list(prev_centroids.keys())
        if len(valid_prev_ids) < min_neighbors + 1:
            processed_data[frame_key] = frame_data
            continue

        prev_centroid_positions = np.array([prev_centroids[mid] for mid in valid_prev_ids])
        kdtree = KDTree(prev_centroid_positions)

        outliers_in_frame = set()

        for marker_id, corners in frame_data.items():
            if marker_id not in prev_centroids:
                continue  # Marker just appeared, can't judge its motion

            current_centroid = calculate_centroid(np.array(corners))
            displacement_vec = current_centroid - prev_centroids[marker_id]

            # Find neighbors in the PREVIOUS frame
            distances, indices = kdtree.query(current_centroid, k=min_neighbors + 1)

            neighbor_displacements = []
            for idx in indices[1:]:  # Exclude the point itself
                neighbor_id = valid_prev_ids[idx]
                if neighbor_id in frame_data:  # Ensure neighbor still exists in current frame
                    neighbor_curr_centroid = calculate_centroid(np.array(data[frame_key][neighbor_id]))
                    neighbor_disp_vec = neighbor_curr_centroid - prev_centroids[neighbor_id]
                    neighbor_displacements.append(neighbor_disp_vec)

            if not neighbor_displacements:
                continue  # No valid neighbors to compare against

            # Compare this marker's motion to the median motion of its neighbors
            median_neighbor_displacement = np.median(neighbor_displacements, axis=0)
            motion_discrepancy = np.linalg.norm(displacement_vec - median_neighbor_displacement)

            if motion_discrepancy > distance_threshold:
                outliers_in_frame.add(marker_id)

        # Build the final frame data, excluding the outliers
        processed_data[frame_key] = {
            mid: p for mid, p in frame_data.items() if mid not in outliers_in_frame
        }

    return processed_data


def filter_track_islands(data, min_segment_len):
    """STAGE 4: For stable markers, removes small, disconnected track segments (islands)."""
    logger.info("STAGE 4: Filtering small, isolated track segments (islands)...")
    marker_appearances = {}
    frame_keys_int = sorted([int(k) for k in data.keys()])

    for frame_idx in frame_keys_int:
        for marker_id in data[str(frame_idx)].keys():
            if marker_id not in marker_appearances:
                marker_appearances[marker_id] = []
            marker_appearances[marker_id].append(frame_idx)

    # For each marker, determine the set of frame indices that belong to a "long enough" segment
    valid_frames_per_marker = {}
    for marker_id, frame_indices in tqdm(marker_appearances.items(), desc="Analyzing track segments"):
        valid_frames_per_marker[marker_id] = get_all_segments(frame_indices, min_segment_len)

    # Create the final data object, only keeping points that are part of a valid segment
    processed_data = {}
    for frame_idx in frame_keys_int:
        frame_key = str(frame_idx)
        processed_data[frame_key] = {}
        for marker_id, points in data[frame_key].items():
            if marker_id in valid_frames_per_marker and frame_idx in valid_frames_per_marker[marker_id]:
                processed_data[frame_key][marker_id] = points

    return processed_data

def main():
    parser = argparse.ArgumentParser(description="Triangulate marker positions from multiple cameras")
    parser.add_argument("--folder", type=Path, help="The folder containing the detection data")
    parser.add_argument("--output", type=str, help="Name of the final video file", default="output_video.mp4")
    parser.add_argument("--frames_3D_int", type=int, required=True, help="Number of missing frames required to interpolate 3D")
    parser.add_argument("--window_size", type=int, required=True, help="Number of frames to consider isolation")

    args = parser.parse_args()

    # ---

    z_threshold = 3 # v2 was 2
    # window_size = args.window_size

    window_size_global = 10
    smoothing_window = 5
    neighbor_dist_thresh = 0.03
    window_size_local = 10

    # ---

    logger.info("===========================================")
    logger.info("========== 6. Post-process v3 =============")
    logger.info("===========================================")

    # Define the directory for saving plots and video output
    tracking_dir = args.folder / f"tracking_charuco-suit/triangulation/3D-interpolated-N{args.frames_3D_int}"
    tracking_file = tracking_dir / "triangulation_markers.json"

    with open(tracking_file, "r") as f:
        data = json.load(f)

        # --- Run the New 4-Stage Filtering Pipeline ---
        data_stage1 = filter_globally_unstable_markers(data, window_size_global)
        data_stage2 = smooth_trajectories_temporally(data_stage1, smoothing_window)
        data_stage3 = filter_motion_outliers(data_stage2, neighbor_dist_thresh)
        processed_data = filter_track_islands(data_stage3, window_size_local)

        # Save the processed data to a new JSON file
        # --- THIS IS THE FIX: Use the custom saver function ---
        output_file_path = tracking_dir / "triangulation_markers_processed.json"
        save_result_to_json(processed_data, output_file_path)

        logger.success(f"Post-processing complete. Filtered data saved to {output_file_path}")


if __name__ == "__main__":
    main()