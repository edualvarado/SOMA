import json
from pathlib import Path
from typing import Tuple, Union

import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import numpy as np

from ..board import Board
from ..utils import save_to_pdf


class ArucoBoard(Board):
    """
    A wrapper for the GridBoard class from the OpenCV library.
    This is due to provide a more user-friendly interface for the CharucoBoard class, in line with the other board classes.

    Args:
        marker_size: The size of the markers in mm.
        marker_sepation: The separation between the markers in mm.
        gridsize: The number of squares in the board in the x and y directions.
    """

    def __init__(
        self,
        marker_size: float,
        marker_spacing: float,
        gridsize: Tuple[int, int],
    ):
        self.marker_size = marker_size
        self.marker_spacing = marker_spacing
        self.gridsize = gridsize
        width_mm = gridsize[0] * marker_size + (gridsize[0] - 1) * marker_spacing
        height_mm = gridsize[1] * marker_size + (gridsize[1] - 1) * marker_spacing
        self.size_mm = (width_mm, height_mm)
        aruco_dict = get_dictionary(gridsize[0] * gridsize[1])

        self.board = aruco.GridBoard(gridsize, marker_size, marker_spacing, aruco_dict)

    def detect(self, frame, draw=False):
        aruco_dict = self.board.getDictionary()

        detect_params = cv2.aruco.DetectorParameters()
        detect_params.useAruco3Detection = False
        detect_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        refine_params = cv2.aruco.RefineParameters()

        detector = cv2.aruco.ArucoDetector(aruco_dict, detect_params, refine_params)
        corners, ids, _ = detector.detectMarkers(frame)

        if ids is None:
            return None, None

        if draw:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        return corners, ids

    def estimate_pose(self, corners, ids, camera_matrix, dist_coeffs, use_fisheye=False):
        """Estimate the pose of the board from the corners and ids detected in the frame.

        Args:
            frame: The frame where the board was detected. Shape (H, W, 3)
            corners: The corners of the markers in the board. Shape (N, 4, 2)
            ids: The ids of the markers in the board. Shape (N,)
            camera_matrix: The camera matrix. Shape (3, 3)
            dist_coeffs: The distortion coefficients. Shape (5,)
            use_fisheye: Whether to use the fisheye model for the camera.
        """
        obj_pts, img_pts = self.match_ids_to_objpts(corners, ids)

        if obj_pts is None:
            return None, None

        if len(obj_pts) < 6:
            return None, None

        if use_fisheye:
            undistorted_img_pts = cv2.fisheye.undistortPoints(img_pts, camera_matrix, dist_coeffs)
            ret, rvec, tvec = cv2.solvePnP(obj_pts, undistorted_img_pts, np.eye(3), np.zeros(5))
        else:
            ret, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs)

        if not ret:
            print("Failed to estimate pose")

        return rvec, tvec

    def match_ids_to_objpts(self, corners, ids):
        obj_pts, img_pts = self.board.matchImagePoints(corners, ids)

        if obj_pts is None:
            return None, None

        return obj_pts / 1000, img_pts

    def save(self, path: Union[str, Path]):
        path = Path(path).resolve()
        path.mkdir(parents=True, exist_ok=True)

        self.save_image(path)
        self.to_json(path / "config.json")

    @classmethod
    def from_json(cls, path: Union[str, Path]):
        with open(path, "r") as file:
            data = json.load(file)
        return cls(data["square_size"], data["marker_size"], data["gridsize"])

    def visualize(self):
        pixel_size = mm_to_px(self.size_mm[0]), mm_to_px(self.size_mm[1])
        image = self.board.generateImage(pixel_size)
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.show()

    def save_image(self, path: Union[str, Path]):
        pixel_size = mm_to_px(self.size_mm[0]), mm_to_px(self.size_mm[1])
        image = self.board.generateImage(pixel_size)

        # Check if result is bigger than A4
        if self.size_mm[0] > 297 or self.size_mm[1] > 297:
            paper_size = "A3"
            print("Board is larger than A4, saving to A3")
        else:
            paper_size = "A4"

        save_to_pdf(image, path / "board.pdf", str(self), max_size_mm=max(self.size_mm), paper_size=paper_size)
        print(f"Board image saved to {path / 'board.pdf'}")

    def to_json(self, path: Union[str, Path]):
        with open(path, "w") as file:
            json.dump(
                {
                    "board_type": "aruco",
                    "marker_size": self.marker_size,
                    "marker_spacing": self.marker_spacing,
                    "gridsize": self.gridsize,
                },
                file,
            )

    def __str__(self):
        return f"ArucoBoard(marker_size={self.marker_size}, marker_sepation={self.marker_spacing}, gridsize={self.gridsize})"


def mm_to_px(mm, ppi=300):
    return round(mm * ppi / 25.4)


def get_dictionary(n_markers: int):
    if n_markers < 50:
        dict_id = aruco.DICT_5X5_50
    elif n_markers < 100:
        dict_id = aruco.DICT_5X5_100
    elif n_markers < 250:
        dict_id = aruco.DICT_5X5_250
    elif n_markers < 1000:
        dict_id = aruco.DICT_5X5_1000
    else:
        raise ValueError(f"Invalid number of markers: {n_markers}")

    return aruco.getPredefinedDictionary(dict_id)
