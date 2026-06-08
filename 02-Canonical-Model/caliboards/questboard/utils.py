from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def deg2rad(deg: float) -> float:
    return deg * np.pi / 180


def mm_to_pdf_pixels(mm):
    return int(mm * (72 / 25.4))  # 72 DPI / 25.4 mm per inch


def get_marker_in_plane(marker_size: float, board_size: Tuple[int], border_size: float = None) -> np.ndarray:
    """Generate the positions of the corners of the marker in a plane of size board_size.

    Args:
        marker_size: The size of the marker in millimeters.
        board_size: The size of the board in millimeters. We assume the ordering is (height, width).
        border_size: The size of the border in millimeters. If None, the border size is calculated automatically.

    Returns:
        The positions of the corners of the marker in a plane of size board_size. A 3D array of shape (n, 4, 3) where n is the number of markers.
        The ordering of the corners is (top left, top right, bottom right, bottom left).
    """
    h, w = board_size
    assert h >= marker_size and w >= marker_size

    # Number of markers fitting in each dimension
    h_markers = int(h / marker_size)
    w_markers = int(w / marker_size)

    # Determine border and spacing sizes
    if border_size is None:
        border_size_h = (h - h_markers * marker_size) / (h_markers + 1)
        border_size_w = (w - w_markers * marker_size) / (w_markers + 1)
        bh, bw = border_size_h, border_size_w
    else:
        assert border_size * 2 + marker_size <= h
        assert border_size * 2 + marker_size <= w
        bh, bw = border_size, border_size

    # Spacing calculation for more than one marker
    h_spacing = (h - 2 * bh - h_markers * marker_size) / max(1, h_markers - 1)
    w_spacing = (w - 2 * bw - w_markers * marker_size) / max(1, w_markers - 1)

    # Calculate the initial offset to center the origin
    initial_offset_x = -w / 2 + bw
    initial_offset_y = +h / 2 - bh

    # Generate marker positions
    corners = []
    for i in range(h_markers):
        for j in range(w_markers):
            top_left_x = initial_offset_x + j * (marker_size + w_spacing)
            top_left_y = initial_offset_y - i * (marker_size + h_spacing)
            corners.append(
                np.array(
                    [
                        [top_left_x, top_left_y],  # Top left
                        [top_left_x + marker_size, top_left_y],  # Top right
                        [
                            top_left_x + marker_size,
                            top_left_y - marker_size,
                        ],  # Bottom right
                        [top_left_x, top_left_y - marker_size],  # Bottom left
                    ]
                )
            )

    # Adding the Z dimension
    corners = [np.hstack([corner, np.zeros((4, 1))]) for corner in corners]

    return np.array(corners)


# Visualization function
def visualize_markers_3d(corners: np.ndarray, idx: Optional[List[int]] = None, fig=None, ax=None):
    if idx is None:
        idx = list(range(corners.shape[0]))
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    # Iterate over each set of corners
    for i, marker in zip(idx, corners):
        # Ensure the marker has 3 dimensions, add a z-coordinate of 0 if missing
        if marker.shape[1] == 2:
            marker = np.hstack([marker, np.zeros((4, 1))])

        # Define a square (quadrilateral) for each set of corners
        sq = [[marker[i] for i in range(4)]]

        # Create a Poly3DCollection object for each square
        pc = Poly3DCollection(sq, alpha=0.5)
        ax.add_collection3d(pc)

        # Draw the ID in the center of the marker
        x = np.mean(marker[:, 0])
        y = np.mean(marker[:, 1])
        z = np.mean(marker[:, 2])
        ax.text(
            x,
            y,
            z,
            str(i),
            color="black",
            fontsize=10,
            horizontalalignment="center",
            verticalalignment="center",
        )

    # Setting the limits of the plot. Adjust as needed
    ax.set_xlim([np.min(corners[:, :, 0]), np.max(corners[:, :, 0])])
    ax.set_ylim([np.min(corners[:, :, 1]), np.max(corners[:, :, 1])])
    ax.set_zlim([np.min(corners[:, :, 2]), np.max(corners[:, :, 2])])

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")

    ax.set_aspect("equal")

    return fig, ax


def build_T(face: str, front_size: Tuple[int, int], face_size: Tuple[int, int], angle: float) -> np.ndarray:
    """
    Creates a 4x4 transformation matrix for a specified face of the board.

    Args:
    - face: A string specifying the face ('left', 'right', or 'top').
    - front_size: A tuple (height, width) specifying the size of the front face.
    - face_size: A tuple (height, width) specifying the size of the face.
    - angle: The angle in radians between the front face and the specified face.

    Returns:
    - A 4x4 numpy array representing the transformation matrix.
    """
    hf, wf = front_size
    hf_s, wf_s = face_size

    if face == "left":
        tx = wf / 2 + wf_s / 2 * np.cos(angle)
        tz = -wf_s / 2 * np.sin(angle)
        t = np.array([tx, 0, tz])
        R = np.array(
            [
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)],
            ]
        )

    elif face == "right":
        tx = wf / 2 + wf_s / 2 * np.cos(angle)
        tz = -wf_s / 2 * np.sin(angle)
        t = np.array([-tx, 0, tz])
        R = np.array(
            [
                [np.cos(angle), 0, -np.sin(angle)],
                [0, 1, 0],
                [np.sin(angle), 0, np.cos(angle)],
            ]
        )

    elif face == "top":
        ty = hf / 2 + hf_s / 2 * np.cos(angle)
        tz = -hf_s / 2 * np.sin(angle)
        t = np.array([0, ty, tz])
        R = np.array(
            [
                [1, 0, 0],
                [0, np.cos(-angle), -np.sin(-angle)],
                [0, np.sin(-angle), np.cos(-angle)],
            ]
        )

    else:
        raise ValueError("Invalid face name. Choose from 'left', 'right', or 'top'.")

    T = np.eye(4)
    T[:3, 3] = t
    T[:3, :3] = R
    return T


def get_face_image(face_size_mm, local_corners_mm, ids, marker_size, dictionary, ppi):
    face_size_px = (
        int(ppi * face_size_mm[0] / 25.4),
        int(ppi * face_size_mm[1] / 25.4),
    )  # (height, width)

    # Calculate the size of the markers in pixels
    marker_size_px = int(ppi * marker_size / 25.4)

    # Convert the corners to pixels
    offset = np.array([face_size_mm[1] / 2, face_size_mm[0] / 2, 0])
    corners_bottomleft = local_corners_mm + offset
    corners_px = corners_bottomleft * ppi / 25.4
    corners_px[:, :, 1] = face_size_px[0] - corners_px[:, :, 1]
    corners_px = corners_px.astype(np.int32)

    canvas = np.ones(face_size_px, dtype=np.uint8) * 255

    for id_, corner in zip(ids, corners_px):
        img_marker = cv2.aruco.generateImageMarker(dictionary, id_, marker_size_px)

        tlx, tly = round(corner[0, 0]), round(corner[0, 1])
        canvas[tly : tly + marker_size_px, tlx : tlx + marker_size_px] = img_marker

    return canvas


def plot_image_3D(ax, img, size, T):
    """Plot an image in 3D.

    Args:
        ax: The matplotlib axis to plot on.
        img: The image to plot, as a numpy array.
        size: The size of the image in millimeters. A tuple (height, width).
        T: The transformation matrix to apply to the image, as a 4x4 numpy array.
    """
    # Get the corners in global coordinates
    size = np.array(size)

    # Rescale image to be 5x lower resolution
    img = cv2.resize(img, (0, 0), fx=0.05, fy=0.05)

    # Assuming the image is a 2D grayscale image
    height, width = img.shape

    # Create a grid of coordinates for the image
    U, V = np.meshgrid(
        np.linspace(-size[1] / 2, size[1] / 2, width),
        np.linspace(-size[0] / 2, size[0] / 2, height),
    )

    # Flatten the coordinates and append zeros and ones for homogeneous transformation
    coords = np.vstack(
        (
            U.flatten(),
            V.flatten(),
            np.zeros_like(U.flatten()),
            np.ones_like(U.flatten()),
        )
    )

    # Apply the transformation matrix
    transformed_coords = T @ coords

    # Reshape and extract 3D coordinates
    X_t, Y_t, Z_t = transformed_coords[:3, :].reshape(3, height, width)

    # Invert the Y axis
    img = np.flipud(img)

    # Plot the image
    ax.plot_surface(X_t, Y_t, Z_t, facecolors=plt.cm.gray(img / img.max()), rstride=1, cstride=1)
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_aspect("equal")

    return ax
