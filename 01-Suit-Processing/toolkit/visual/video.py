from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.colors as mcolors
import numpy as np
import trimesh
from loguru import logger
from tqdm import tqdm

from ..common import Intrinsics, Pose
from .mesh import draw_mesh
from .text import draw_rounded_rectangle_with_text


def draw_poses_dets_on_video(
    video_path: Path,
    intrinsics: Intrinsics,
    extrinsics: Pose | Dict[int, Pose],
    list_of_frame_to_poses: List[Dict[int, Pose]],
    list_of_frame_to_dets: List[Dict[int, Dict[str, np.ndarray]]],
    list_of_frame_to_arucos: List[Dict[int, Dict[str, np.ndarray]]],
    frame_to_body_poses: Dict[int, Dict[str, Pose]] = None,
    skeleton: Dict[str, str] = None,
    fixed_mesh_path: Path = None,
    pose_names: List[str] = None,
    pose_colors: List[str] = None,
    pose_lengths: List[float] = None,
    limit: int = None,
):
    """Draws the estimated poses on the video as a generator.

    Args:
        video_path: Path to the video file.
        intrinsics: Camera intrinsics.
        extrinsics: Camera extrinsics, or dictionary containing the extrinsics for each frame.
        list_of_frame_to_poses: List of dictionaries containing the poses for each frame.
        list_of_frame_to_dets: List of dictionaries containing the detections for each frame.
        list_of_frame_to_arucos: List of dictionaries containing the arucos for each frame.
        frame_to_body_poses: Dictionary containing the body poses for each frame.
        skeleton: Skeleton hierarchy.
        fixed_mesh_path: Path to the fixed mesh to draw on the video.
        pose_names: List of names of the poses.
        pose_colors: List of colors for the poses.
        limit: Limit the number of frames to process.

    Yields:
        np.ndarray: The frame with the poses drawn.
    """
    cap = cv2.VideoCapture(str(video_path))

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if limit is None else limit
    pbar = tqdm(total=n_frames, desc="Drawing frames on video")

    is_fisheye = len(intrinsics.distortion) == 4

    if fixed_mesh_path is not None:
        mesh = trimesh.load(str(fixed_mesh_path))
        V, F = mesh.vertices, mesh.faces
        logger.info(f"Loaded mesh with {V.shape[0]} vertices and {F.shape[0]} faces")

        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        overlay = draw_mesh(V, F, extrinsics, intrinsics, np.zeros((h, w, 3), dtype=np.uint8), return_overlay=True)

        # Split the BGRA image into its channels
        alpha = overlay[:, :, 3]

        # Create a mask from the alpha channel
        mask = np.expand_dims(alpha / 255.0, axis=-1)

        # Multiply the BGR image by the inverse of the mask
        foreground = (overlay[:, :, :3] * mask).astype(np.uint8)

        inv_mask = ((1 - mask) * 255).astype(np.uint16)

    counter = 0
    while True:
        ret, img = cap.read()

        if not ret:
            break

        if limit is not None and counter >= limit:
            break

        # Compute the length and thickness of the axes
        width, height = img.shape[1], img.shape[0]
        length = min(width, height) / 25
        thickness = min(width, height) // 500
        radius = max(min(width, height) // 400, 5)
        skeleton_thickness = max(min(width, height) // 1000, 3)
        text_height = max(min(width, height) // 80, 5)
        text_offset = max(min(width, height) // 35, 5)

        # Drawing the fixed mesh
        if fixed_mesh_path is not None:
            # Add the foregroun computed previously
            weighted_img = (img * inv_mask // 255).astype(np.uint8)
            img = foreground + weighted_img

        # Drawing Poses
        extrinsic = extrinsics.get(counter) if isinstance(extrinsics, dict) else extrinsics

        '''
        # Drawing poses
        for jdx, frame_to_poses in enumerate(list_of_frame_to_poses):
            pose_world_coords = frame_to_poses.get(counter)

            if pose_world_coords is None:
                continue

            cTw, wTp = extrinsic, pose_world_coords
            cTp = cTw @ wTp  # Pose in camera frame

            rvec, tvec = cTp.rvec.flatten(), cTp.tvec.flatten() * 1000

            # Draw the board for the new camera frame
            K, d = intrinsics.matrix, intrinsics.distortion
            if not is_fisheye:
                length_scaled = length if pose_lengths is None else length * pose_lengths[jdx]
                img = draw_frame_axes(img, K, d, rvec, tvec, length_scaled, thickness)

                # If pose names are provided, draw them under the origin
                if pose_names is not None:
                    # Project the origin to the image
                    origin = np.array([0, 0, 0], dtype=np.float32).reshape(-1, 1, 3)
                    coord2D = cv2.projectPoints(origin, rvec, tvec, K, d)[0].flatten()
                    coord2D = (int(coord2D[0]), int(coord2D[1]) + text_offset)

                    if pose_colors is not None:
                        color_rgb = mcolors.to_rgb(pose_colors[jdx])
                        color = (int(color_rgb[2] * 255), int(color_rgb[1] * 255), int(color_rgb[0] * 255))
                    else:
                        color = (182, 119, 0)

                    name = pose_names[jdx].replace("_", " ")

                    img = draw_rounded_rectangle_with_text(img, coord2D, text_height, color, name, (0, 0, 0))
            else:
                print("Drawing the axes is not supported for fisheye cameras")
        '''

        # TODO - Draw detections
        for jdx, frame_to_dets in enumerate(list_of_frame_to_dets):

            # Get the ChArUco detections for each frame
            dets_world_coords = frame_to_dets.get(counter)

            if dets_world_coords is not None:
                # Access to the detections of the video to visualize
                video_idx = int(video_path.stem.split("stream")[1])
                nested_dict = dets_world_coords.get(video_idx)  # Replace 1 (video) with the correct key if necessary

                if nested_dict is not None:
                    corners = nested_dict.get('corners')
                    ids = nested_dict.get('ids')

                    if len(corners) == len(ids):
                        corners = np.reshape(corners, (len(corners), 1, 2))
                        ids = np.reshape(ids, (len(ids), 1))
                        img = cv2.aruco.drawDetectedCornersCharuco(img, corners, ids, (0, 0, 255))
                    else:
                        print(f"Skipping frame {counter} due to mismatch in number of corners and IDs")

        # TODO - Draw arucos
        for jdx, frame_to_arucos in enumerate(list_of_frame_to_arucos):

            # Get the ChArUco detections for each frame
            arucos_world_coords = frame_to_arucos.get(counter)

            if arucos_world_coords is not None:
                # Access to the detections of the video to visualize
                video_idx = int(video_path.stem.split("stream")[1])
                nested_dict = arucos_world_coords.get(video_idx)  # Replace 1 (video) with the correct key if necessary

                if nested_dict is not None:
                    cornersMarkers = nested_dict.get('cornersMarkers')
                    idsMarkers = nested_dict.get('idMarkers')

                    if len(cornersMarkers) == len(idsMarkers):
                        cornersMarkers = [cornersMarkers[i:i + 1] for i in range(cornersMarkers.shape[0])]
                        idsMarkers = idsMarkers.flatten()

                        #img = cv2.aruco.drawDetectedMarkers(img, cornersMarkers, idsMarkers)

                        # To print larger IDs
                        img = cv2.aruco.drawDetectedMarkers(img, cornersMarkers)
                        # Define the font scale and thickness
                        font_scale = 2.0
                        font_thickness = 8
                        # Iterate over each marker
                        for i in range(len(cornersMarkers)):
                            # Get the center of the marker
                            c = np.average(cornersMarkers[i][0], axis=0)

                            # Draw the ID at the center of the marker
                            cv2.putText(img, str(idsMarkers[i]), (int(c[0]), int(c[1])), cv2.FONT_HERSHEY_SIMPLEX,
                                        font_scale, (0, 0, 255), font_thickness)


                    else:
                        print(f"Skipping frame {counter} due to mismatch in number of corners and IDs")

        # Drawing Body
        pos2D = {}
        if frame_to_body_poses is not None:
            body_poses = frame_to_body_poses.get(counter)
            if body_poses is None:
                continue

            for name, pose in body_poses.items():
                cTw, wTj = extrinsic, pose

                # Compute the pose in the camera frame, joint <- camera
                cTj = cTw @ wTj

                rvec, tvec = cTj.rvec.flatten(), cTj.tvec.flatten() * 1000

                # Check if the z coordinate is too small (less than 10 mm)
                if tvec[2] < 10:
                    continue

                # Draw the board for the new camera frame
                K, d = intrinsics.matrix, intrinsics.distortion

                # Also draw only the position with a marker
                origin = np.array([0, 0, 0], dtype=np.float32).reshape(-1, 1, 3)

                if not is_fisheye:
                    coord2D = cv2.projectPoints(origin, rvec, tvec, K, d)[0].flatten()
                else:
                    coord2D = cv2.fisheye.projectPoints(origin, rvec, tvec, K, d)[0].flatten()

                # Check if coord2D is a NaN
                if np.isnan(coord2D).any():
                    continue

                if coord2D.min() < -1e4 or coord2D.max() > 1e4:
                    continue

                center = coord2D.astype(np.int32).flatten()

                cv2.circle(img, center, radius, (0, 0, 0), -1)

                # Store the 2D position to draw the skeleton
                pos2D[name] = center

            if skeleton is not None:
                for joint, parent in skeleton.items():
                    if joint not in pos2D or parent not in pos2D:
                        continue

                    # If both points are outside the image, skip
                    h, w = img.shape[:2]
                    jcoord, pcoord = pos2D[joint], pos2D[parent]

                    is_j_outside = jcoord[0] < 0 or jcoord[0] >= w or jcoord[1] < 0 or jcoord[1] >= h
                    is_p_outside = pcoord[0] < 0 or pcoord[0] >= w or pcoord[1] < 0 or pcoord[1] >= h

                    if is_j_outside and is_p_outside:
                        continue

                    cv2.line(img, pcoord, jcoord, (100, 0, 200), skeleton_thickness)

        yield img

        counter += 1
        pbar.update(1)

    pbar.close()


def draw_frame_axes(image, cameraMatrix, distCoeffs, rvec, tvec, length, thickness):
    """Custom version of cv2.drawFrameAxes that plots farther axes first."""

    # Ensure image has the correct number of channels
    assert image.ndim == 2 or image.ndim == 3, "Number of channels must be 1, 3 or 4"

    # Ensure image is not empty and length is positive
    assert image.size > 0, "Image must not be empty"
    assert length > 0, "Length must be greater than 0"

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    # Define 3D points for the axes
    points = np.array(
        [
            [0, 0, 0],  # Origin
            [length, 0, 0],  # X-axis
            [0, length, 0],  # Y-axis
            [0, 0, length],  # Z-axis
        ],
        dtype=np.float32,
    ).reshape(-1, 3)

    # Compute the Z coordinate in the camera frame
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(1, 3)

    point_without_origin = points[1:]
    points_camera = (R @ point_without_origin.T).T + t
    z_coords = points_camera[:, 2].reshape(-1)

    # Reorder the points so that the first axis is the one with the highest z value
    ordering = np.argsort(z_coords)[::-1]

    # Project the points to the image
    imgpts, _ = cv2.projectPoints(points, rvec, tvec, cameraMatrix, distCoeffs)

    # Draw the axes
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    p1 = tuple(imgpts[0].ravel().astype(int))
    for idx in ordering:
        color = colors[idx]
        p2 = tuple(imgpts[idx + 1].ravel().astype(int))

        cv2.line(image, p1, p2, color, thickness)

    return image


def draw_frame_corners(image, cameraMatrix, distCoeffs, rvec, tvec, thickness):

    # Ensure image has the correct number of channels
    assert image.ndim == 2 or image.ndim == 3, "Number of channels must be 1, 3 or 4"

    # Ensure image is not empty and length is positive
    assert image.size > 0, "Image must not be empty"

    # Define 3D points for the corners
    points = np.array(
        [
            [-0.5, -0.5, 0],  # Bottom-left corner
            [0.5, -0.5, 0],  # Bottom-right corner
            [0.5, 0.5, 0],  # Top-right corner
            [-0.5, 0.5, 0],  # Top-left corner
        ],
        dtype=np.float32,
    ).reshape(-1, 3)

    # Project the points to the image
    imgpts, _ = cv2.projectPoints(points, rvec, tvec, cameraMatrix, distCoeffs)

    # Draw the corners
    color = (100, 100, 100)
    for i in range(4):
        p1 = tuple(imgpts[i].ravel().astype(int))
        p2 = tuple(imgpts[(i+1)%4].ravel().astype(int))
        cv2.line(image, p1, p2, color, thickness)

    return image
