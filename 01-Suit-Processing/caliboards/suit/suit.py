import json
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from ..board import Board
from loguru import logger
from pathlib import Path
from typing import Tuple, Union

# Example distance threshold (pixels)
DISTANCE_THRESHOLD = 10

# Radius to find neighboring markers
RADIUS = 50

# Example distance threshold (pixels) for def remove_conflicts
DISTANCE_THRESHOLD_REMOVAL = 10

# Example distance threshold (pixels) for def remove_isolated
RADIUS_REMOVAL = 250

# Maximum allowed difference in IDs to consider them "close"
MAX_ALLOWED_ID_DIFF = 50

def find_neighbors(centroids, current_centroid, radius):
    """
    Find the indices of centroids that are within the specified radius from the current marker.
    """
    neighbors = []
    for i, centroid in enumerate(centroids):
        distance = np.linalg.norm(centroid - current_centroid)
        if distance < radius:
            neighbors.append(i)
    return neighbors

def calculate_centroid(corners):
    """Calculate the centroid of a marker given its corner points."""
    # Average the X and Y coordinates of the corners
    x_coords = corners[:, 0]
    y_coords = corners[:, 1]
    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)
    return np.array([centroid_x, centroid_y])

def refine_outlier_ids(corners, ids, radius, max_allowed_id_diff, verbose):
    """
    DEPRECATED
    --
    Final pass to remove outlier markers based on their surrounding IDs,
    ensuring independent evaluation for each marker.
    """
    # Calculate centroids for all markers
    centroids = [calculate_centroid(corner[0]) for corner in corners]

    # To track indices marked for removal
    remove_indices = set()

    # Iterate through each marker
    for i, centroid in enumerate(centroids):
        # Find neighbors of the current marker
        neighbors = find_neighbors(centroids, centroid, radius)

        # Exclude the current marker itself from neighbor list
        neighbor_ids = [
            ids[j] for j in neighbors if j != i and j not in remove_indices
        ]

        # Skip processing if no valid neighbors are found
        if not neighbor_ids:
            if verbose:
                logger.debug(f"[DEBUG] Marker {i} (ID: {ids[i]}) has no neighbors — marked for removal.")
            remove_indices.add(i)
            continue

        # Check if at least one valid close neighbor exists
        has_close_neighbor = any(
            abs(ids[i] - neighbor_id) <= max_allowed_id_diff for neighbor_id in neighbor_ids
        )

        if not has_close_neighbor:
            if verbose:
                logger.debug(f"[DEBUG] Marker {i} (ID: {ids[i]}) has no close enough neighbors — marking for removal.")
            remove_indices.add(i)
        else:
            if verbose:
                logger.debug(f"[DEBUG] Marker {i} (ID: {ids[i]}) has at least one close neighbor — keeping it.")

    # Remove markers marked for removal
    final_corners = [corner for i, corner in enumerate(corners) if i not in remove_indices]
    final_ids = [id_val for i, id_val in enumerate(ids) if i not in remove_indices]

    # Debugging Output
    if remove_indices:
        if verbose:
            logger.debug(f"[DEBUG] Removed Marker IDs: {[ids[i] for i in remove_indices]}")
    return final_corners, final_ids

def resolve_conflicts(corners, ids, verbose):
    """
    DEPRECATED
    --
    Resolve positional conflicts by removing markers that have IDs far from the surrounding IDs.
    """
    # Calculate centroids for all markers
    centroids = [calculate_centroid(corner[0]) for corner in corners]

    remove_indices = set()

    # For each pair of markers, check for positional coincidence
    for i, centroid1 in enumerate(centroids):
        for j, centroid2 in enumerate(centroids):
            if i >= j:  # Avoid duplicate comparisons and compare only distinct pairs
                continue

            # Check if two markers share the same (or close) position
            distance = np.linalg.norm(centroid1 - centroid2)
            if distance < DISTANCE_THRESHOLD:
                # Found two markers in the same position

                # Find neighbors for the first marker
                neighbors1 = find_neighbors(centroids, centroid1, RADIUS)
                ids1 = [ids[k] for k in neighbors1 if k != i]  # Exclude self

                # Find neighbors for the second marker
                neighbors2 = find_neighbors(centroids, centroid2, RADIUS)
                ids2 = [ids[k] for k in neighbors2 if k != j]  # Exclude self

                # If no neighbors are found, we skip this conflict
                if not ids1 or not ids2:
                    continue

                # Calculate the average difference in IDs for both markers
                diff1 = np.mean([abs(ids[i] - id_neighbor) for id_neighbor in ids1])
                diff2 = np.mean([abs(ids[j] - id_neighbor) for id_neighbor in ids2])

                # Compare the differences and decide which marker to remove
                if diff1 > diff2:
                    # Marker i seems incorrect, mark it for removal
                    remove_indices.add(i)
                else:
                    # Marker j seems incorrect, mark it for removal
                    remove_indices.add(j)

    # Remove any markers that were marked for removal
    final_corners = [corner for i, corner in enumerate(corners) if i not in remove_indices]
    final_ids = [id_val for i, id_val in enumerate(ids) if i not in remove_indices]

    return final_corners, final_ids

def remove_conflicts(corners, ids, verbose):
    """
    Overlapping markers will be removed
    """

    # Calculate centroids for all markers
    centroids = [calculate_centroid(corner[0]) for corner in corners]

    # Keep track of indices of markers to be removed
    remove_indices = set()

    # Check distance between every pair of centroids
    for i, centroid1 in enumerate(centroids):
        if i in remove_indices:
            continue  # Skip markers already marked for removal
        for j, centroid2 in enumerate(centroids):
            if i >= j or j in remove_indices:  # Avoid duplicate comparisons
                continue

            # Calculate the distance between the two centroids
            distance = np.linalg.norm(centroid1 - centroid2)
            if distance < DISTANCE_THRESHOLD_REMOVAL:
                if verbose:
                    logger.debug(f"[DEBUG] Conflict detected between Marker {i} (ID: {ids[i]}) "
                          f"and Marker {j} (ID: {ids[j]}). Removing both.")
                # Mark both markers for removal
                remove_indices.add(i)
                remove_indices.add(j)

    # Remove the markers marked for removal
    final_corners = [corner for idx, corner in enumerate(corners) if idx not in remove_indices]
    final_ids = [id_val for idx, id_val in enumerate(ids) if idx not in remove_indices]

    # Debugging Output
    if remove_indices:
        removed_ids = [ids[idx] for idx in remove_indices]
        if verbose:
            logger.debug(f"[DEBUG] REMOVING CONFLICTS: Marker IDs deleted: {removed_ids}")

    return final_corners, final_ids

def remove_isolated(corners, ids, verbose):
    """
    Remove markers that are isolated, i.e., not having any other markers nearby within the specified radius.
    TODO: Could be better (markers do not have same size for all videos).

    Args:
        corners: List of marker corner points.
        ids: List of marker IDs corresponding to each marker corner.
        radius: Radius around each marker within which to search for neighbors.

    Returns:
        final_corners: Filtered list of corners with isolated markers removed.
        final_ids: Filtered list of IDs with isolated markers removed.
    """
    # Calculate centroids for all markers
    centroids = [calculate_centroid(corner[0]) for corner in corners]

    # Keep track of indices of markers to be removed
    remove_indices = set()

    # Check for neighbors for each marker centroid
    for i, current_centroid in enumerate(centroids):
        neighbors = find_neighbors(centroids, current_centroid, RADIUS_REMOVAL)

        # Exclude the current marker itself from the neighbors
        neighbors = [j for j in neighbors if j != i]

        if not neighbors:  # No neighbors found
            if verbose:
                logger.debug(f"[DEBUG] Marker {i} (ID: {ids[i]}) is isolated — marking for removal.")
            remove_indices.add(i)

    # Remove the markers that are marked for removal
    final_corners = [corner for idx, corner in enumerate(corners) if idx not in remove_indices]
    final_ids = [id_val for idx, id_val in enumerate(ids) if idx not in remove_indices]

    # Debugging Output
    if remove_indices:
        removed_ids = [ids[idx] for idx in remove_indices]
        if verbose:
            logger.debug(f"[DEBUG] Removed Isolated Marker IDs: {removed_ids}")

    return final_corners, final_ids

class Suit(Board):
    """
    A wrapper for the Board class from the OpenCV library.
    This is due to provide a more user-friendly interface to detect the suit.

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
            detector_setup: int = 1,
    ):
        self.square_size = square_size
        self.marker_size = marker_size
        self.gridsize = gridsize
        self.size_mm = (gridsize[0] * square_size, gridsize[1] * square_size)

        aruco_dict = aruco.getPredefinedDictionary(aruco_dict_id)

        # Suit is non-mutable
        self.board = aruco.CharucoBoard((49, 65), 0.03, 0.022, aruco_dict)
        self.board.setLegacyPattern(True)


        # TODO Parameter tuning - Replace to make it a more elegant solution
        # ---------

        # Create detector (setup 1)
        detector_parameters_setup_1 = cv2.aruco.DetectorParameters()
        charuco_parameters_setup_1 = cv2.aruco.CharucoParameters()

        # Minimum window size for adaptive thresholding before finding contours
        detector_parameters_setup_1.adaptiveThreshWinSizeMin = 3 # (DEFAULT 3)

        # Maximum window size for adaptive thresholding before finding contours
        detector_parameters_setup_1.adaptiveThreshWinSizeMax = 23 # (DEFAULT 23)

        # Increments from adaptiveThreshWinSizeMin to adaptiveThreshWinSizeMax during the thresholding
        detector_parameters_setup_1.adaptiveThreshWinSizeStep = 1 # (DEFAULT 10) TODO: CHANGED

        # Constant for adaptive thresholding before finding contours (default 7)
        detector_parameters_setup_1.adaptiveThreshConstant = 7 # (DEFAULT 7)

        # Number of bits (per dimension) for each cell of the marker when removing the perspective
        detector_parameters_setup_1.perspectiveRemovePixelPerCell = 4 # (DEFAULT 4)

        # Corners Refinement Method
        detector_parameters_setup_1.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        # Maximum window size for the corner refinement process (in pixels)
        detector_parameters_setup_1.cornerRefinementWinSize = 5 # (DEFAULT 5)

        # Error correction rate respect to the maximum error correction capability for each dictionary
        detector_parameters_setup_1.errorCorrectionRate = 0.6 # (DEFAULT 0.6)

        # Min and max parameters control the expected size of the marker perimeter relative to the image size.
        # Increase the min value (e.g., 0.05 to 0.1) to filter out smaller, potentially noisy detections.
        detector_parameters_setup_1.minMarkerPerimeterRate = 0.005 # (DEFAULT 0.03) TODO: CHANGED

        # Determine maximum perimeter for marker contour to be detected.
        detector_parameters_setup_1.maxMarkerPerimeterRate = 4 # (DEFAULT 4)

        # Minimum distance of any marker corner to the image border.
        # Increase this value if detections near the edges of the image are unstable or flickering.
        detector_parameters_setup_1.minDistanceToBorder = 3 # (DEFAULT 3)

        # Controls the margin ignored in the perspective transformation step.
        # Decrease this value if the markers are not being detected due to partial occlusion or noise near the edges.
        detector_parameters_setup_1.perspectiveRemoveIgnoredMarginPerCell = 0.13 # (DEFAULT 0.13)

        # Minimum average distance between the corners of the two markers to be grouped
        detector_parameters_setup_1.minMarkerDistanceRate = 0.125 # (DEFAULT 0.125)

        # Minimum accuracy during the polygonal approximation process to determine which contours are squares
        detector_parameters_setup_1.polygonalApproxAccuracyRate = 0.03 # (DEFAULT 0.03)

        # ---------

        # Create detector (setup 2)
        detector_parameters_setup_2 = cv2.aruco.DetectorParameters()
        charuco_parameters_setup_2 = cv2.aruco.CharucoParameters()

        # Minimum window size for adaptive thresholding before finding contours
        detector_parameters_setup_2.adaptiveThreshWinSizeMin = 3 # (DEFAULT 3)

        # Maximum window size for adaptive thresholding before finding contours
        detector_parameters_setup_2.adaptiveThreshWinSizeMax = 23 # (DEFAULT 23)

        # Increments from adaptiveThreshWinSizeMin to adaptiveThreshWinSizeMax during the thresholding
        detector_parameters_setup_2.adaptiveThreshWinSizeStep = 1 # (DEFAULT 10) TODO: CHANGED

        # Constant for adaptive thresholding before finding contours (default 7)
        detector_parameters_setup_2.adaptiveThreshConstant = 7 # (DEFAULT 7)

        # Number of bits (per dimension) for each cell of the marker when removing the perspective
        detector_parameters_setup_2.perspectiveRemovePixelPerCell = 4 # (DEFAULT 4)

        # Corners Refinement Method
        detector_parameters_setup_2.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG

        # Maximum window size for the corner refinement process (in pixels)
        detector_parameters_setup_2.cornerRefinementWinSize = 5 # (DEFAULT 5)

        # Error correction rate respect to the maximum error correction capability for each dictionary
        detector_parameters_setup_2.errorCorrectionRate = 0.6 # (DEFAULT 0.6)

        # Min and max parameters control the expected size of the marker perimeter relative to the image size.
        # Increase the min value (e.g., 0.05 to 0.1) to filter out smaller, potentially noisy detections.
        detector_parameters_setup_2.minMarkerPerimeterRate = 0.005 # (DEFAULT 0.03) TODO: CHANGED

        # Determine maximum perimeter for marker contour to be detected.
        detector_parameters_setup_2.maxMarkerPerimeterRate = 4 # (DEFAULT 4)

        # Minimum distance of any marker corner to the image border.
        # Increase this value if detections near the edges of the image are unstable or flickering.
        detector_parameters_setup_2.minDistanceToBorder = 3 # (DEFAULT 3)

        # Controls the margin ignored in the perspective transformation step.
        # Decrease this value if the markers are not being detected due to partial occlusion or noise near the edges.
        detector_parameters_setup_2.perspectiveRemoveIgnoredMarginPerCell = 0.13 # (DEFAULT 0.13)

        # Minimum average distance between the corners of the two markers to be grouped
        detector_parameters_setup_2.minMarkerDistanceRate = 0.125 # (DEFAULT 0.125)

        # Minimum accuracy during the polygonal approximation process to determine which contours are squares
        detector_parameters_setup_2.polygonalApproxAccuracyRate = 0.03 # (DEFAULT 0.03)

        # ---------

        # Create detector (setup 2)
        detector_parameters_setup_3 = cv2.aruco.DetectorParameters()
        charuco_parameters_setup_3 = cv2.aruco.CharucoParameters()

        # Minimum window size for adaptive thresholding before finding contours
        detector_parameters_setup_3.adaptiveThreshWinSizeMin = 3 # (DEFAULT 3)

        # Maximum window size for adaptive thresholding before finding contours
        detector_parameters_setup_3.adaptiveThreshWinSizeMax = 23 # (DEFAULT 23)

        # Increments from adaptiveThreshWinSizeMin to adaptiveThreshWinSizeMax during the thresholding
        detector_parameters_setup_3.adaptiveThreshWinSizeStep = 1 # (DEFAULT 10) TODO: CHANGED

        # Constant for adaptive thresholding before finding contours (default 7)
        detector_parameters_setup_3.adaptiveThreshConstant = 7 # (DEFAULT 7)

        # Number of bits (per dimension) for each cell of the marker when removing the perspective
        detector_parameters_setup_3.perspectiveRemovePixelPerCell = 4 # (DEFAULT 4)

        # Corners Refinement Method
        detector_parameters_setup_3.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR

        # Maximum window size for the corner refinement process (in pixels)
        detector_parameters_setup_3.cornerRefinementWinSize = 5 # (DEFAULT 5)

        # Error correction rate respect to the maximum error correction capability for each dictionary
        detector_parameters_setup_3.errorCorrectionRate = 0.6 # (DEFAULT 0.6)

        # Min and max parameters control the expected size of the marker perimeter relative to the image size.
        # Increase the min value (e.g., 0.05 to 0.1) to filter out smaller, potentially noisy detections.
        detector_parameters_setup_3.minMarkerPerimeterRate = 0.005 # (DEFAULT 0.03) TODO: CHANGED

        # Determine maximum perimeter for marker contour to be detected.
        detector_parameters_setup_3.maxMarkerPerimeterRate = 4 # (DEFAULT 4)

        # Minimum distance of any marker corner to the image border.
        # Increase this value if detections near the edges of the image are unstable or flickering.
        detector_parameters_setup_3.minDistanceToBorder = 3 # (DEFAULT 3)

        # Controls the margin ignored in the perspective transformation step.
        # Decrease this value if the markers are not being detected due to partial occlusion or noise near the edges.
        detector_parameters_setup_3.perspectiveRemoveIgnoredMarginPerCell = 0.13 # (DEFAULT 0.13)

        # Minimum average distance between the corners of the two markers to be grouped
        detector_parameters_setup_3.minMarkerDistanceRate = 0.125 # (DEFAULT 0.125)

        # Minimum accuracy during the polygonal approximation process to determine which contours are squares
        detector_parameters_setup_3.polygonalApproxAccuracyRate = 0.03 # (DEFAULT 0.03)

        # ---------

        refine_parameters = cv2.aruco.RefineParameters()

        self.detector_setup_1 = cv2.aruco.CharucoDetector(
                self.board,
                charuco_parameters_setup_1,
                detector_parameters_setup_1,
                refine_parameters
        )

        self.detector_setup_2 = cv2.aruco.CharucoDetector(
                self.board,
                charuco_parameters_setup_2,
                detector_parameters_setup_2,
                refine_parameters
        )

        self.detector_setup_3 = cv2.aruco.CharucoDetector(
                self.board,
                charuco_parameters_setup_3,
                detector_parameters_setup_3,
                refine_parameters
        )

    def detect(self, frame, draw=False):
        # Convert colorspace
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ChaArUco corners
        charucos, ids_charucos, _, _ = self.detector.detectBoard(gray)

        if draw:
            cv2.aruco.drawDetectedCornersCharuco(frame, charucos, ids_charucos)

        return charucos, ids_charucos

    def detect_all(self, frame, frame_idx, video_path, calibrations, draw=False, setup=1, verbose=False):
        # # Choose parametrization
        # if setup == 1:
        #     self.detector = self.detector_setup_1
        # elif setup == 2:
        #     self.detector = self.detector_setup_2
        # elif setup == 3:
        #     raise ValueError(f"Invalid detector setup: {setup}")

        if verbose:
            logger.debug(f"[DEBUG] Using 2D detect parametrization {setup}")

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # We use the traditional detector for Setup 1 and 3
        if setup == 1 or setup == 3:
            # Detect Aruco corners_markers (directly, no instance needed) TODO (SETUP 3) - Make more elegant
            corners_markers, id_markers, _ = aruco.detectMarkers(gray, self.board.getDictionary(),
                                                                 parameters=self.detector_setup_1.getDetectorParameters())

            corners_markers_final = list(corners_markers) if corners_markers is not None else []
            id_markers_final = list(id_markers) if id_markers is not None else []

        # We use the April detector for Setup 2 and 3
        if setup == 2 or setup == 3:
            corners_markers_april, id_markers_april, _ = aruco.detectMarkers(gray, self.board.getDictionary(),
                                                                             parameters=self.detector_setup_2.getDetectorParameters())

            # If we use Setup 2, we only use April detector
            if setup == 2:
                corners_markers_final = list(corners_markers_april) if corners_markers_april is not None else []
                id_markers_final = list(id_markers_april) if id_markers_april is not None else []

        # If we use Setup 3, we combine both traditional and April detectors
        if setup == 3:
            # Compute centroids for the first method
            centroids1 = [calculate_centroid(corner[0]) for corner in corners_markers]

            if corners_markers_april is not None and id_markers_april is not None:
                for i, corner in enumerate(corners_markers_april):

                    # TODO: AUTOMATIZE - HOW TO COMBINE THEM?
                    # ---

                    """
                    # Overlapping IDs (code: before)
                    if id_markers_april[i] not in id_markers_final:
                        corners_markers_final.append(corner)
                        id_markers_final.append(id_markers_april[i])
                    """

                    # ---

                    """
                    # To preserve AprilTag if there is a coincidence
                    centroid2 = calculate_centroid(corner[0])
                    id2 = id_markers_april[i]

                    # Check if this marker is close to any marker in the first method
                    matched = False

                    for j, centroid1 in enumerate(centroids1):
                        distance = np.linalg.norm(centroid2 - centroid1)
                        if distance < DISTANCE_THRESHOLD:
                            # If close enough, replace the ID with the more "correct" one (APRILTAG)
                            matched = True
                            id_markers_final[j] = id2  # Overwrite incorrect ID
                            break

                    if not matched:
                        # If no match, add this marker to the final results
                        corners_markers_final.append(corner)
                        id_markers_final.append(id2)
                    """

                    # ---

                    """
                    # To preserve RefinePix if there is a coincidence
                    centroid2 = calculate_centroid(corner[0])
                    id2 = id_markers_april[i]

                    # Check if this marker is close to any marker in the first method
                    matched = False

                    for j, centroid1 in enumerate(centroids1):
                        distance = np.linalg.norm(centroid2 - centroid1)
                        if distance < DISTANCE_THRESHOLD:
                            # If close enough, replace the ID with the more "correct" one (APRILTAG)
                            matched = True
                            break

                    if not matched:
                        # If no match, add this marker to the final results
                        corners_markers_final.append(corner)
                        id_markers_final.append(id2)
                    """

                    # ---

                    """
                    Pretty good estimate (no overlappings, correct ones)
                    However, wrong markers might appear from first detection
                    We can use the canonical data to correct the ones that are wrong afterwards
                    """

                    """
                    # Overlapping IDs + Refinement
                    if id_markers_april[i] not in id_markers_final:
                        corners_markers_final.append(corner)
                        id_markers_final.append(id_markers_april[i])

                    # Check positional information for coincidence markers and solve conflicts (code: (smart_rem))
                    corners_markers_final, id_markers_final = resolve_conflicts(corners_markers_final, id_markers_final)

                    # Refine all parameters based on surroundings - TODO: Replace by canonical model (code: (smart_rem)_2)
                    # corners_markers_final, id_markers_final = refine_outlier_ids(corners_markers_final, id_markers_final, 100, 120)
                    """

                    # ---

                    """
                    Final option: If the centroids are very close to each other, we just delete both detections
                    as we are not sure which one is correct.
                    In the later stage with the canonical data, we will fill this. 
                    In the end, better to remove now and fill later, than to have incorrect detections and to fix 
                    them later.
                    """

                    # Overlapping IDs + Removal
                    if id_markers_april[i] not in id_markers_final:
                        corners_markers_final.append(corner)
                        id_markers_final.append(id_markers_april[i])

                    # Check position information of markers and remove overlapping ones (code: (hard_rem))
                    corners_markers_final, id_markers_final = remove_conflicts(corners_markers_final, id_markers_final, verbose)

                    # Check for isolated markers. If around there is no other centroid, is isolated, so we remove them (code: (hard_rem))
                    corners_markers_final, id_markers_final = remove_isolated(corners_markers_final, id_markers_final, verbose)

                    # ---

        id_markers_final = np.array(id_markers_final).reshape(-1, 1)

        if verbose:
            if setup == 1:
                logger.debug(f"[DEBUG - SETUP 1] Video {video_path.name} - f{frame_idx} - ({len(corners_markers)}) markers")
            elif setup == 2:
                logger.debug(f"[DEBUG - SETUP 2] Video {video_path.name} - f{frame_idx} - ({len(corners_markers_april)}) markers")
            elif setup == 3:
                logger.debug(f"[DEBUG - SETUP 3] Video {video_path.name} - f{frame_idx} - ({len(corners_markers_final)}) markers")

        if len(corners_markers_final) > 0:
            """
            Without camera parameters, done by homography - unstable for our case
            With camera parameters, done by pose prediction
            """

            retval, corners_charuco, id_charuco = aruco.interpolateCornersCharuco(corners_markers_final, id_markers_final, gray,
                                                                                        self.board,
                                                                                        calibrations[video_path.name]["K1"],
                                                                                        calibrations[video_path.name]["d1"])

            if verbose:
                if corners_charuco is not None:
                    logger.debug(f"[DEBUG] Video {video_path.name} - f{frame_idx} - ({len(corners_charuco)}) corners")
                else:
                    logger.debug(f"[DEBUG] Video {video_path.name} - f{frame_idx} - (NONE) corners")

            if draw:
                """
                cv2.aruco.drawDetectedMarkers(frame, corners_markers, id_markers)
                if corners_charuco is not None:
                    corners_charuco = np.reshape(corners_charuco, (len(corners_charuco), 1, 2))
                    id_charuco = np.reshape(id_charuco, (len(id_charuco), 1))
                    cv2.aruco.drawDetectedCornersCharuco(frame, corners_charuco, id_charuco)
                """

                if id_charuco is not None:
                    # Pass 1: Fill all the markers
                    for corner in corners_charuco:

                        # Get the x, y coordinates as integers
                        x, y = int(corner[0][0]), int(corner[0][1])

                        # Draw the point as a small circle (radius=2)
                        cv2.circle(
                            frame,  # Image on which to draw
                            (x, y),  # (x, y) center of the point
                            radius=3,  # Small circle with radius = 2
                            color=(0, 255, 0),  # Green color in BGR
                            thickness=-1  # -1 fills the circle (makes it a solid point)
                        )

                if id_markers_final is not None:
                    # Pass 1: Fill all the markers
                    for marker in corners_markers_final:
                        # Reshape the corner array to integers for drawing
                        pts = marker.reshape((-1, 1, 2)).astype(np.int32)

                        # Fill the inside of the marker with a custom color (e.g., red)
                        cv2.fillPoly(frame, [pts], color=(0, 0, 255))  # Fill with red color

                    # Pass 2: Draw all the text
                    for i, corner in enumerate(corners_markers_final):
                        # Calculate the center of the marker
                        marker_center = tuple(np.mean(corner.reshape((-1, 2)), axis=0).astype(int))

                        # Get the ID of the marker and convert it to a string
                        marker_id = str(id_markers_final[i][0])  # Convert numpy.int32 to string

                        # Define font properties
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 0.5
                        textThickness = 2  # Thickness of the actual text

                        # Draw the border (text outline) with a larger size in black
                        cv2.putText(
                            frame,
                            marker_id,
                            marker_center,
                            fontFace,
                            fontScale,
                            color=(0, 0, 0),  # Outer color (e.g., black for contour)
                            thickness=textThickness + 2,  # Make the border thicker
                            lineType=cv2.LINE_AA
                        )

                        # Draw the actual text on top in white
                        cv2.putText(
                            frame,
                            marker_id,
                            marker_center,
                            fontFace,
                            fontScale,
                            color=(255, 255, 255),  # Inner color (text color, e.g., white)
                            thickness=textThickness,
                            lineType=cv2.LINE_AA
                        )

            return corners_charuco, id_charuco, corners_markers_final, id_markers_final
        else:
            logger.info(f"[INFO] Markers not detected")
            return None, None, None, None

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
