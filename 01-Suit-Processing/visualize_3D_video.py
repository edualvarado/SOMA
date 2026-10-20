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
from tqdm import tqdm


def get_all_segments(numbers, min_segment_len):
    """Finds all continuous segments and returns the set of frame indices belonging to long-enough segments."""
    if not numbers:
        return set()
    numbers = sorted(list(set(numbers)))
    if not numbers:
        return set()

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


def identify_track_islands(data, min_segment_len):
    """
    Analyzes the data and identifies all points that are part of a short, disconnected track segment.
    This is the "sister" function to the filter, designed for visualization.

    Returns:
        dict: A dictionary mapping each frame key to a set of marker IDs considered 'islands' in that frame.
    """
    logger.info("Building marker appearance history to identify islands...")
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

    # Now, identify the points that are NOT in a valid segment
    islands_per_frame = {}
    logger.info("Identifying island points for visualization...")
    for frame_idx in frame_keys_int:
        frame_key = str(frame_idx)
        islands_in_this_frame = set()
        for marker_id in data[frame_key].keys():
            # A marker at this frame is an "island" if the frame index
            # is NOT part of any of its valid, long segments.
            if marker_id in valid_frames_per_marker and frame_idx not in valid_frames_per_marker[marker_id]:
                islands_in_this_frame.add(marker_id)
        islands_per_frame[frame_key] = islands_in_this_frame

    return islands_per_frame


def calculate_centroid(points):
    """Calculate the centroid of a list of 3D points."""
    return np.mean(points, axis=0)

def main():
    parser = argparse.ArgumentParser(description="Triangulate marker positions from multiple cameras")
    parser.add_argument("--folder", type=Path, help="The folder containing the detection data")
    parser.add_argument("--output", type=str, help="Name of the final video file", default="output_video.mp4")
    parser.add_argument("--fps", type=int, help="Frames per second for the video", default=30)
    parser.add_argument("--frames_3D_int", type=int, required=True, help="Number of missing frames required to interpolate 3D")
    parser.add_argument("--debug", action="store_true", help="Save 2D detections debug frames")

    args = parser.parse_args()

    logger.info("========================================")
    logger.info("== 5. Visualizing 3D tracking (video) ==")
    logger.info("========================================")

    # Define the directory for saving plots and video output
    tracking_dir = args.folder / f"tracking_charuco-suit/triangulation/3D-interpolated-N{args.frames_3D_int}"

    # ---
    tracking_file = tracking_dir / "triangulation_markers_processed.json"
    # tracking_file = tracking_dir / "triangulation_markers.json"
    # ---

    tracking_dir_debug = args.folder / f"tracking_charuco-suit/triangulation/3D-interpolated-N{args.frames_3D_int}/debug"
    tracking_dir_debug.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    tracking_dir_debug_images = args.folder / f"tracking_charuco-suit/triangulation/3D-interpolated-N{args.frames_3D_int}/debug/images"
    tracking_dir_debug_images.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    with open(tracking_file, "r") as f:
        data = json.load(f)

    logger.info(f"[INFO] Files (plots and video) will be saved in: {tracking_dir_debug}")

    # ---

    # Set your desired window size
    window_size = 10  # Adjust this value as needed

    # Track IDs that do not appear in at least `window_size` consecutive frames
    frame_keys = sorted(data.keys(), key=lambda x: int(x))  # Sort frames numerically
    frames_with_ids = {frame_key: set(frame_data.keys()) for frame_key, frame_data in data.items()}

    # Dictionary to store the isolated markers per frame
    isolated_ids_per_frame = {}

    # Loop through all frames and calculate isolation
    for i, frame_key in enumerate(frame_keys):
        current_ids = frames_with_ids[frame_key]
        # Initialize a dictionary for tracking appearances in the sliding window
        id_appearance_count = {}

        # Slide over the frames within the window
        for j in range(max(0, i - window_size + 1), min(len(frame_keys), i + window_size)):
            for object_id in frames_with_ids[frame_keys[j]]:
                if object_id not in id_appearance_count:
                    id_appearance_count[object_id] = 0
                id_appearance_count[object_id] += 1

        # Mark IDs that appear less than `window_size` times as isolated
        isolated_ids = {object_id for object_id, count in id_appearance_count.items() if count < window_size}
        isolated_ids_per_frame[frame_key] = isolated_ids

    # ---

    # Use our new function to identify the islands based on the correct logic
    islands_to_remove = identify_track_islands(data, window_size)

    # ---

    x_limits, y_limits, z_limits = [float("inf"), float("-inf")], [float("inf"), float("-inf")], [float("inf"),
                                                                                                  float("-inf")]

    # First pass: calculate global axis limits
    for frame_key, frame_data in data.items():
        for object_id, coordinates in frame_data.items():
            swapped_coordinates = [(z, x, y) for x, y, z in coordinates]
            xs, ys, zs = zip(*swapped_coordinates)
            x_limits = [min(x_limits[0], min(xs)), max(x_limits[1], max(xs))]
            y_limits = [min(y_limits[0], min(ys)), max(y_limits[1], max(ys))]
            z_limits = [min(z_limits[0], min(zs)), max(z_limits[1], max(zs))]

    max_range = max(
        x_limits[1] - x_limits[0],
        y_limits[1] - y_limits[0],
        z_limits[1] - z_limits[0]
    ) / 2.0
    mid_x = (x_limits[0] + x_limits[1]) / 2.0
    mid_y = (y_limits[0] + y_limits[1]) / 2.0
    mid_z = (z_limits[0] + z_limits[1]) / 2.0

    frame_images = []

    # Second pass: Plot each frame and save images
    for frame_key, frame_data in data.items():
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

        for idx, (object_id, coordinates) in enumerate(frame_data.items()):
            # Swap axes: X -> Z (height), Y -> X, Z -> Y
            swapped_coordinates = [(z, x, y) for x, y, z in coordinates]
            xs, ys, zs = zip(*swapped_coordinates)

            # Scatter points with color map
            color = viridis(idx / len(frame_data))

            # ---

            # if object_id == "930":
            #     ax.scatter(xs, ys, zs, c='blue', s=0.5, marker='*', label=f"ID {object_id}")
            # else:
            #     ax.scatter(xs, ys, zs, c='blue', s=0.5, label=f"ID {object_id}")

            # ---

            # Check if the object ID is isolated in the current frame
            if object_id in isolated_ids_per_frame[frame_key]:
                ax.scatter(xs, ys, zs, c='red', s=0.5, marker='*', label=f"ID {object_id}")
            elif object_id in islands_to_remove[frame_key]:
                ax.scatter(xs, ys, zs, c='yellow', s=0.5, marker='*', label=f"ID {object_id}")
            else:
                ax.scatter(xs, ys, zs, c='blue', s=0.5, label=f"ID {object_id}")

            if object_id in islands_to_remove[frame_key]:
                ax.scatter(xs, ys, zs, c='yellow', s=0.5, marker='*', label=f"ID {object_id}")
            else:
                ax.scatter(xs, ys, zs, c='blue', s=0.5, label=f"ID {object_id}")

            # ---

            # TODO: ADD IDs
            # Add ID text near the first point of each marker (example: (xs[0], ys[0], zs[0]))
            # -----------------------
            # for x, y, z in swapped_coordinates:
            #     ax.text(x, y, z, f"{object_id}", color="black", fontsize=8)
            # -----------------------

        # Set equal aspect ratio and axis limits
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.view_init(elev=30, azim=45)  # Adjustable viewing angles
        plt.title(f"3D Coordinates (Frame {frame_key})")

        # To show also and rotate
        # -----------------------
        # plt.show()
        # -----------------------

        # Save the current plot as an image
        image_path = tracking_dir_debug_images / f"frame_{frame_key}.png"
        plt.savefig(image_path)
        plt.close(fig)  # Close the figure to free resources
        logger.info(f"Saved plot for frame {frame_key} at {image_path}")
        frame_images.append(image_path)

    # Compile the images into a video
    output_video_path = tracking_dir_debug / args.output
    frame = cv2.imread(str(frame_images[0]))
    height, width, _ = frame.shape

    video_writer = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))

    for image_path in frame_images:
        img = cv2.imread(str(image_path))
        video_writer.write(img)

    video_writer.release()
    logger.info(f"Video has been saved at {output_video_path}")

    # Delete the temporal folder `tracking_dir_debug_images`
    if not args.debug:
        if tracking_dir_debug_images.exists():
            shutil.rmtree(tracking_dir_debug_images)
            logger.info(f"Deleted temporal folder: {tracking_dir_debug_images}")
        else:
            logger.warning(f"Temporal folder {tracking_dir_debug_images} does not exist or was already deleted.")


if __name__ == "__main__":
    main()
