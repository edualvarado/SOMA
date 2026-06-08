import cv2


def cut_video(video_path, indexes, fps, suffix="", out_folder=None):
    """Cut a video based on the indexes of the frames to keep.

    Args:
        video_path: A string representing the path to the video file.
        indexes: A list of integers representing the indexes of the frames to keep.
        fps: An integer representing the frames per second of the output video.
        suffix: A string representing the suffix to add to the output video file. Defaults to "".
        out_folder: A string representing the path to the output folder. Defaults to None.
    """
    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Failed to open video file {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_folder = out_folder or video_path.parent
    filepath = out_folder / (video_path.stem + suffix + ".mp4")

    # Create the output video file
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(filepath), fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        raise IOError(f"Failed to create video writer for {filepath}")

    if not indexes:
        raise ValueError("The indexes list is empty.")

    if max(indexes) >= total_frames:
        raise ValueError("One or more indexes are out of the range of the total frames.")

    # Read and write the selected frames
    max_index = max(indexes)
    for i in range(total_frames):
        cap.grab()
        if i > max_index:
            break
        if i not in indexes:
            continue

        success, frame = cap.retrieve()
        if not success:
            raise IOError(f"Failed to retrieve frame {i} from {video_path}")

        out.write(frame)

    # Release the video files
    cap.release()
    out.release()

    # Ensure the file is written correctly
    if not filepath.exists() or filepath.stat().st_size == 0:
        raise IOError(f"Output file {filepath} is not created correctly or is empty.")

    print(f"Video saved as {filepath.resolve()}")


def find_slowest_fps(video_paths):
    slowest_fps = float("inf")
    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        slowest_fps = min(slowest_fps, fps)
        cap.release()

    return slowest_fps
