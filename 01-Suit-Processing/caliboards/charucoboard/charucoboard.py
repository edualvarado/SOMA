import json
from pathlib import Path
from typing import Tuple, Union

import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import numpy as np

from ..board import Board
from ..utils import save_to_pdf
from loguru import logger


class ChArUcoBoard(Board):
    """
    A wrapper for the CharucoBoard class from the OpenCV library.
    This is due to provide a more user-friendly interface for the CharucoBoard class, in line with the other board classes.

    Args:
        square_size: The size of the squares in the board in mm.
        marker_size: The size of the markers in mm.
        gridsize: The number of squares in the board in the x and y directions.
        aruco_dict_id: The ID of the ArUco dictionary to be used for the board.
        legacy_pattern: Whether to use the legacy pattern for the board.

    """

    def __init__(
        self,
        square_size: float,
        marker_size: float,
        gridsize: Tuple[int, int],
        aruco_dict_id: int = aruco.DICT_4X4_1000,
        legacy_pattern: bool = True,
    ):
        self.square_size = square_size
        self.marker_size = marker_size
        self.gridsize = gridsize
        self.size_mm = (gridsize[0] * square_size, gridsize[1] * square_size)

        aruco_dict = aruco.getPredefinedDictionary(aruco_dict_id)
        self.board = aruco.CharucoBoard(gridsize, square_size, marker_size, aruco_dict)

        self.board.setLegacyPattern(legacy_pattern)

        # Create detector
        charuco_parameters = cv2.aruco.CharucoParameters()
        detector_parameters = cv2.aruco.DetectorParameters() # Default: Min: 3, Max: 23, Step: 10
        refine_parameters = cv2.aruco.RefineParameters()

        self.detector = cv2.aruco.CharucoDetector(self.board, charuco_parameters, detector_parameters,
                                                  refine_parameters)

    def detect(self, frame, draw=False):

        # Convert colorspace
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create detector
        charuco_parameters = cv2.aruco.CharucoParameters()
        detector_parameters = cv2.aruco.DetectorParameters()
        refine_parameters = cv2.aruco.RefineParameters()

        detector = cv2.aruco.CharucoDetector(
            self.board,
            charuco_parameters,
            detector_parameters,
            refine_parameters,
        )

        charucos, ids_charucos, _, _ = detector.detectBoard(gray)

        if draw:
            cv2.aruco.drawDetectedCornersCharuco(frame, charucos, ids_charucos)

        return charucos, ids_charucos

    def detect_all(self, frame, draw=False):

        # Convert colorspace
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect Aruco markers (directly, no instance needed)
        markers, ids_markers, _ = aruco.detectMarkers(gray, self.board.getDictionary())

        # Detect ChaArUco corners
        charucos, ids_charucos, _, _ = self.detector.detectBoard(gray)

        if len(markers) > 0:
            # Refine interpolated ChArUco corners
            res2 = cv2.aruco.interpolateCornersCharuco(markers, ids_markers, frame, self.board)

            if draw:
                cv2.aruco.drawDetectedMarkers(frame, markers, ids_markers)
                if res2[1] is not None:
                    res2[1] = np.reshape(res2[1], (len(res2[1]), 1, 2))
                    res2[2] = np.reshape(res2[2], (len(res2[2]), 1))
                    cv2.aruco.drawDetectedCornersCharuco(frame, res2[1], res2[2])

            return res2[1], res2[2], markers, ids_markers
        else:
            if draw:
                if charucos is not None:
                    charucos = np.reshape(charucos, (len(charucos), 1, 2))
                    ids_charucos = np.reshape(ids_charucos, (len(ids_charucos), 1))
                    cv2.aruco.drawDetectedCornersCharuco(frame, charucos, ids_charucos)

            return charucos, ids_charucos, None, None

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
                    "board_type": "charuco",
                    "square_size": self.square_size,
                    "marker_size": self.marker_size,
                    "gridsize": self.gridsize,
                },
                file,
            )

    def __str__(self):
        return f"ChArUcoBoard(square_size={self.square_size}, marker_size={self.marker_size}, gridsize={self.gridsize})"


def mm_to_px(mm, ppi=300):
    return round(mm * ppi / 25.4)


def get_dictionary(n_markers: int):
    if n_markers < 50:
        dict_id = aruco.DICT_4X4_50
    elif n_markers < 100:
        dict_id = aruco.DICT_4X4_100
    elif n_markers < 250:
        dict_id = aruco.DICT_4X4_250
    elif n_markers < 1000:
        dict_id = aruco.DICT_4X4_1000
    else:
        raise ValueError(f"Invalid number of markers: {n_markers}")

    return aruco.getPredefinedDictionary(dict_id)
