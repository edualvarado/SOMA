"""
This script provides the methods to load the detection data.

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

"""

import json
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from ..common import Pose


def load_corners_charuco(folder: Path, keyword: str = "charuco-stream"):
    """Loads corners_charuco from a folder of JSON files."""

    # List all "charuco_stream" json files
    files = list(folder.glob("*.json"))
    files = sorted([f for f in files if keyword in f.stem and "charuco" in f.stem and "- Copy" not in f.stem], key=lambda f: f.stem)

    # New dict to return
    corners_charuco_detections_dict = {}  # frame -> video idx -> corners, ids

    for path in tqdm(files, colour="blue", desc="Loading ChArUco Corners..."):
        # logger.info(f"[INFO] ChArUco corners loading from: {path}")

        # Take video index
        video_idx = int(path.stem.split("stream")[1])

        with open(path, "r") as f:
            corners_charuco_detections = json.load(f)

        for frame, values in corners_charuco_detections.items():
            frame_idx = int(frame)

            corners_charuco = np.array(values["corners_charuco"], dtype=np.float32)
            assert corners_charuco.shape[-1] == 2 # (N,2)

            if values["id_charuco"] is None:
                logger.error("[ERROR] The id field is missing in the detections")
            else:
                id_charuco = np.array(values["id_charuco"], dtype=np.int32)

            if frame_idx not in corners_charuco_detections_dict:
                corners_charuco_detections_dict[frame_idx] = {}

            corners_charuco_detections_dict[frame_idx][video_idx] = dict(corners_charuco=corners_charuco,
                                                                        id_charuco=id_charuco)

    return corners_charuco_detections_dict

def load_corners_markers(folder: Path, keyword: str = "markers-stream"):
    """Loads corners_markers from a folder of JSON files."""

    # List all "markers_stream" json files
    files = list(folder.glob("*.json"))
    files = sorted([f for f in files if keyword in f.stem and "markers" in f.stem and "- Copy" not in f.stem], key=lambda f: f.stem)

    # New dict to return
    corners_markers_detections_dict = {}  # frame -> video idx -> corners, ids

    for path in tqdm(files, colour="blue", desc="Loading Markers Corners..."):
        # logger.info(f"[INFO] Markers corners loading from: {path}")

        # Take video index
        video_idx = int(path.stem.split("stream")[1])

        with open(path, "r") as f:
            corners_markers_detections = json.load(f)

        for frame, values in corners_markers_detections.items():
            frame_idx = int(frame)

            corners_markers = np.array(values["corners_markers"], dtype=np.float32)
            assert corners_markers.shape[-1] == 2 # (N,4,2)

            if values["id_markers"] is None:
                logger.error("[ERROR] The id field is missing in the detections")
            else:
                id_markers = np.array(values["id_markers"], dtype=np.int32)

            if frame_idx not in corners_markers_detections_dict:
                corners_markers_detections_dict[frame_idx] = {}

            corners_markers_detections_dict[frame_idx][video_idx] = dict(corners_markers=corners_markers,
                                                                        id_markers=id_markers)

    return corners_markers_detections_dict
