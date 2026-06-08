from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ..board import Board
from ..utils import save_to_pdf
from .utils import build_T, deg2rad, get_face_image, get_marker_in_plane, plot_image_3D

np.set_printoptions(precision=3, suppress=True)


class QuestBoard(Board):
    """
    A QuestBoard is a 3D board with 4 faces: front, left, right, and top.
    The front face is the main face and the other faces are placed at 90 degrees from the front face.
    By default the side faces have the same size as the front face and the top face has the same width as the front face and the same height as the left face.

    Args:
        marker_size: The size of the markers in mm.
        front_size: The size of the front face in mm.
        side_size: The size of the side faces in mm. If not specified, the side faces will be half the size of the front face.
        top_size: The size of the top face in mm. If not specified, the top face will have the same size as the front face.
    """

    def __init__(
        self,
        marker_size: float,
        front_size: Tuple[int, int],
        side_size: Tuple[int, int] = None,
        top_size: Tuple[int, int] = None,
    ):
        self.marker_size = marker_size
        self.side_angle = deg2rad(90)
        self.top_angle = deg2rad(90)
        side_size = side_size if side_size else (front_size[0], front_size[0])
        top_size = top_size if top_size else (front_size[0], front_size[1])
        self.face_sizes = dict(front=front_size, left=side_size, right=side_size, top=top_size)

        self._validate_dimensions()
        self.T = self._calculate_transformations()
        self.global_corners, self.local_corners = self._place_markers()
        self.ids = self._build_ids()
        self.board = self._build_board()

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

    def visualize(self, ppi=300, fig=None, ax=None):
        if fig is None or ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        # Draw each face
        for face in ["front", "left", "right", "top"]:
            # Calculate the size of the face in pixels
            canvas = self._get_face_image(face, ppi)

            # Draw the face
            plot_image_3D(ax, canvas, self.face_sizes[face], self.T[face])

        plt.axis("off")
        plt.show()

    def save(self, path: Union[str, Path], ppi: int = 300):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.save_images(ppi=ppi, path=path)
        self.to_json(path / "config.json")

    @property
    def all_markers(self):
        return np.concatenate(list(self.global_corners.values()), axis=0).astype(np.float32)

    @property
    def all_ids(self):
        return np.concatenate(list(self.ids.values()), axis=0).astype(np.int32)

    def to_json(self, path: str):
        data = {
            "board_type": "quest",
            "marker_size": self.marker_size,
            "front_size": self.face_sizes["front"],
            "side_size": self.face_sizes["left"],
            "top_size": self.face_sizes["top"],
        }
        with open(path, "w") as file:
            json.dump(data, file, indent=2)
        print(f"Saved QuestBoard configuration to {path}")

    @classmethod
    def from_json(cls, path: str):
        with open(path, "r") as file:
            config = json.load(file)

        return cls(
            config["marker_size"],
            config["front_size"],
            config["side_size"],
            config["top_size"],
        )

    def save_images(self, ppi=300, path="faces/"):
        path = Path(path).resolve()
        path.mkdir(parents=True, exist_ok=True)

        for face in ["front", "left", "right", "top"]:
            # Adding 1 pixel black border
            img = self._get_face_image(face, ppi)
            img = np.pad(img, 1, mode="constant", constant_values=0)
            description = f"↑ {face.capitalize()} face ↑"

            save_to_pdf(img, path / f"{face}.pdf", description, ppi=ppi)
            print(f"Face {face} saved to {path / f'{face}.pdf'}")

    def _build_board(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        return cv2.aruco.Board(self.all_markers, dictionary, self.all_ids)

    def _validate_dimensions(self):
        assert self.face_sizes["left"][0] == self.face_sizes["front"][0]
        assert self.face_sizes["right"][0] == self.face_sizes["front"][0]
        assert self.face_sizes["top"][1] == self.face_sizes["front"][1]

    def _calculate_transformations(self):
        T_f_ls = build_T("left", self.face_sizes["front"], self.face_sizes["left"], self.side_angle)
        T_f_rs = build_T("right", self.face_sizes["front"], self.face_sizes["right"], self.side_angle)
        T_f_t = build_T("top", self.face_sizes["front"], self.face_sizes["top"], self.top_angle)
        return dict(left=T_f_ls, right=T_f_rs, top=T_f_t, front=np.eye(4))

    def _place_markers(self):
        local_markers = {}
        global_markers = {}
        for face in ["front", "left", "right", "top"]:
            # Getting the size of the plane
            size = self.face_sizes[face]
            # Calculating the marker positions in local coordinates
            face_markers = get_marker_in_plane(self.marker_size, size)
            local_markers[face] = face_markers
            # Transforming the markers to global coordinates
            T = self.T[face]
            transformed_markers = self._transform_markers(face_markers, T)
            global_markers[face] = transformed_markers

        return global_markers, local_markers

    def _transform_markers(self, markers, T):
        n = markers.shape[0]
        homomarkers = np.concatenate([markers, np.ones((n, 4, 1))], axis=2)
        global_markers = np.matmul(T, homomarkers.transpose(0, 2, 1)).transpose(0, 2, 1)
        return global_markers[:, :, :3]

    def _build_ids(self):
        ids = {}
        id_counter = 0
        for face in ["front", "left", "right", "top"]:
            corners = self.local_corners[face]
            n = corners.shape[0]
            face_ids = np.arange(id_counter, id_counter + n)
            ids[face] = face_ids
            id_counter += n
        return ids

    def _get_face_image(self, face: str, ppi: int):
        dictionary = self.board.getDictionary()
        return get_face_image(
            self.face_sizes[face],
            self.local_corners[face],
            self.ids[face],
            self.marker_size,
            dictionary,
            ppi,
        )
