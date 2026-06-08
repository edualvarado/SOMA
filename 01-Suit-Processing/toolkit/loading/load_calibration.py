import cv2
import numpy as np
from pathlib import Path
from typing import List
from ..common import Intrinsics, Pose

def load_calibrations(camera_calibration_file: Path, image_width: int):
    """
    Loads camera model from a calibration file.

    Args:
        camera_calibration_file: Path to the cameras.calib file
        image_width: Width of the images used for calibration.
    """

    with open(camera_calibration_file, "r") as file:
        lines = file.readlines()

    camera_models = {}

    for i, line in enumerate(lines):
        if "camera\t" in line:
            idx, camera_data = parse_camera_data(lines, i, image_width)
            camera_models[idx] = camera_data

    return camera_models


def load_resolutions(folder: Path, keyword: str = "stream"):
    """
    Loads video files from a specified folder and it determines the resolution (width and height).

    Args:
        folder: Path to the directory containing video files to process
        keyword: Keyword to filter the video files by. Default is "stream".
    """

    folder = Path(folder)

    video_files = list(folder.glob("*.mp4")) + list(folder.glob("*.avi")) + list(folder.glob("*.mkv"))
    video_files.sort()
    video_files = sorted([f for f in video_files if keyword in f.stem])

    if not video_files:
        raise FileNotFoundError(f"No video files found in {folder}")

    resolutions = {}
    for idx, video in enumerate(video_files):
        cap = cv2.VideoCapture(str(video))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resolutions[idx] = (width, height)

    return resolutions

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def parse_camera_data(lines: List[str], start_line: int, image_width: int):
    """Parses camera data from the given lines starting at start_line."""
    camera_info = lines[start_line : start_line + 25]
    camera_idx = int(camera_info[0].split()[1])
    distortions = np.array(camera_info[11].split()[1:], dtype=np.float32)

    extrinsic_data = [np.array(line.split()[1:], dtype=np.float32) for line in camera_info[16:19]]
    extrinsic_matrix = np.vstack(extrinsic_data)
    extrinsic_matrix[:, 3] /= 1000
    extrinsics_4x4 = np.eye(4)
    extrinsics_4x4[:3, :] = extrinsic_matrix

    intrinsic_data = [np.array(line.split()[1:], dtype=np.float32) for line in camera_info[20:23]]
    intrinsic_matrix = np.vstack(intrinsic_data)
    intrinsic_matrix[:2, :] *= image_width

    intr = Intrinsics(matrix=intrinsic_matrix, distortion=distortions)
    extr = Pose.from_matrix(extrinsics_4x4, parent="cam", child="world")

    return camera_idx, {"intrinsic": intr, "extrinsic": extr}
