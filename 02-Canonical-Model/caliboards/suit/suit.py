import json
from pathlib import Path
from typing import Tuple, Union

import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ..board import Board

from loguru import logger

class Suit(Board):
    """
    A wrapper for the Board class from the OpenCV library.
    This is due to provide a more user-friendly interface to detect the ChArUcos in the suit.

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


        # TODO: Hard-coding board - Review
        # ---------
        # self.board = aruco.CharucoBoard(gridsize, square_size, marker_size, aruco_dict)
        # self.board.setLegacyPattern(legacy_pattern)

        self.board = aruco.CharucoBoard((49, 65), 0.03, 0.022, aruco_dict)
        self.board.setLegacyPattern(True)
        # ---------

        # Create detector
        detector_parameters = cv2.aruco.DetectorParameters()
        charuco_parameters = cv2.aruco.CharucoParameters()

        # TODO Parameter tuning - FIRST WORKING VERSION
        # ---------
        """
        # Parameters for suit - TODO: TUNING IMPORTANT
        detector_parameters.adaptiveThreshWinSizeMin = 3
        detector_parameters.adaptiveThreshWinSizeMax = 23
        detector_parameters.adaptiveThreshWinSizeStep = 1

        # TODO: TUNING IMPORTANT
        # Min and max parameters control the expected size of the marker perimeter relative to the image size.
        # Increase the min value (e.g., 0.05 to 0.1) to filter out smaller, potentially noisy detections.
        detector_parameters.minMarkerPerimeterRate = 0.005

        # Minimum distance of any marker corner to the image border.
        # Increase this value if detections near the edges of the image are unstable or flickering.
        detector_parameters.minDistanceToBorder = 3

        # Corners
        detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        # Controls the margin ignored in the perspective transformation step.
        # Decrease this value if the markers are not being detected due to partial occlusion or noise near the edges.
        detector_parameters.perspectiveRemoveIgnoredMarginPerCell = 0.1
        """
        # ---------

        # TODO Parameter tuning - SECOND WORKING VERSION
        # ---------
        # Parameters for suit - TODO: TUNING IMPORTANT
        detector_parameters.adaptiveThreshWinSizeMin = 3 # (DEFAULT 3)
        detector_parameters.adaptiveThreshWinSizeMax = 23 # (DEFAULT 23)
        detector_parameters.adaptiveThreshWinSizeStep = 1 # (DEFAULT 10)

        detector_parameters.adaptiveThreshConstant = 5 # (DEFAULT 7)

        detector_parameters.perspectiveRemovePixelPerCell = 10 # (DEFAULT 4)

        detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG

        detector_parameters.cornerRefinementWinSize = 7 # (DEFAULT 5)


        detector_parameters.errorCorrectionRate = 1 # (DEFAULT 0.6)

        detector_parameters.minMarkerDistanceRate = 0.1 # (DEFAULT 0.125)
        detector_parameters.minMarkerPerimeterRate = 0.02 # (DEFAULT 0.03)
        detector_parameters.polygonalApproxAccuracyRate = 0.05 # (DEFAULT 0.03)
        # ---------

        refine_parameters = cv2.aruco.RefineParameters()

        self.detector = cv2.aruco.CharucoDetector(
                self.board,
                charuco_parameters,
                detector_parameters,
                refine_parameters
        )


        logger.info(f"-- Suit board created: {self} --")
        logger.info(f"-------------------- Parametrization --------------------")
        logger.info(f"Adaptive Threshold Window (min, max, step): "
                    f"({self.detector.getDetectorParameters().adaptiveThreshWinSizeMin},"
                    f"{self.detector.getDetectorParameters().adaptiveThreshWinSizeMax}, "
                    f"{self.detector.getDetectorParameters().adaptiveThreshWinSizeStep})")
        logger.info(f"Min Marker Perimeter Rate: {self.detector.getDetectorParameters().minMarkerPerimeterRate}")
        logger.info(f"Min Distance to Border: {self.detector.getDetectorParameters().minDistanceToBorder}")
        logger.info(f"Corner Refinement: {self.detector.getDetectorParameters().cornerRefinementMethod}")
        logger.info(f"Remove Ignored Margin per Cell: "
                     f"{self.detector.getDetectorParameters().perspectiveRemoveIgnoredMarginPerCell}")
        logger.info(f"---------------------------------------------------------")

    def detect(self, frame, draw=False):

        # Convert colorspace
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ChaArUco corners
        charucos, ids_charucos, _, _ = self.detector.detectBoard(gray)

        if draw:
            cv2.aruco.drawDetectedCornersCharuco(frame, charucos, ids_charucos)

        return charucos, ids_charucos

    def detect_all(self, frame, frame_idx, video_path, calibrations, draw=False):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect Aruco markers (directly, no instance needed)
        markers, ids_markers, _ = aruco.detectMarkers(gray, self.board.getDictionary(),
                                                      parameters=self.detector.getDetectorParameters())

        logger.info(f"Video {video_path.name} - f{frame_idx}: Detected Markers: ({len(markers)})")

        if len(markers) > 0:
            # Without camera parameters, done by homography - unstable for our case
            # With camera parameters, done by pose prediction

            # TODO - Giving calibrations for changing the way corners are detected
            # ---------
            # charucoRetval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(markers, ids_markers, gray, self.board)
            retval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(markers, ids_markers, gray,
                                                                                        self.board,
                                                                                        calibrations[video_path.name]["K1"],
                                                                                        calibrations[video_path.name]["d1"])
            # ---------

            logger.debug(f"Video {video_path.name}: retval: ({retval})")

            if charucoCorners is not None:
                logger.info(f"Video {video_path.name} - f{frame_idx}: Detected Corners: ({len(charucoCorners)})")
            else:
                logger.info(f"Video {video_path.name} - f{frame_idx}: Detected Corners: (None)")

            if draw:
                cv2.aruco.drawDetectedMarkers(frame, markers, ids_markers)
                if charucoCorners is not None:
                    charucoCorners = np.reshape(charucoCorners, (len(charucoCorners), 1, 2))
                    charucoIds = np.reshape(charucoIds, (len(charucoIds), 1))
                    cv2.aruco.drawDetectedCornersCharuco(frame, charucoCorners, charucoIds)

            return charucoCorners, charucoIds, markers, ids_markers
        else:

            logger.debug(f"Markers not detected")
            return None, None, None, None

    # TODO REVIEW
    def draw_all(self, frame, marker_corners):
        logger.info(f"Not implemented yet")

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
        return f"ChArUco-Suit(square_size={self.square_size}, marker_size={self.marker_size}, gridsize={self.gridsize})"


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

