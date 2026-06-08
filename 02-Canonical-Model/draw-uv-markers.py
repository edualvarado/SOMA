"""
Plot UV markers in the UV map
"""

import json
import matplotlib.pyplot as plt
import argparse
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from loguru import logger
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2
import numpy as np
import os
from pathlib import Path

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def save_debug_image(image_path, frame):
    cv2.imwrite(str(image_path), frame)

def draw_debug_image_fixed(corners_markers, id_markers, frame):
    # Pass 1: Fill all the markers
    for marker in corners_markers:

        # print(corners_markers)
        # print(marker)

        # Reshape the corner array to integers for drawing
        pts = np.array(marker).reshape((-1, 1, 2)).astype(np.int32)

        # Fill the inside of the marker with a custom color (e.g., red)
        cv2.fillPoly(frame, [pts], color=(0, 0, 255))  # Fill with red color

        # Pass 2: Draw all the text
        for i, corner in enumerate(corners_markers):
            # Calculate the center of the marker
            marker_center = tuple(np.mean(np.array(corner).reshape((-1, 2)), axis=0).astype(int))

            # Get the ID of the marker and convert it to a string
            marker_id = str(id_markers[i][0])  # Convert numpy.int32 to string

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

def draw_debug_image_manual(corners_markers, id_markers, frame):
    # Pass 1: Fill all the markers
    for marker in corners_markers:

        # print(corners_markers)
        # print(marker)

        # Reshape the corner array to integers for drawing
        pts = np.array(marker).reshape((-1, 1, 2)).astype(np.int32)

        # Fill the inside of the marker with a custom color (e.g., red)
        cv2.fillPoly(frame, [pts], color=(0, 255, 0))  # Fill with green color

        # Pass 2: Draw all the text
        for i, corner in enumerate(corners_markers):
            # Calculate the center of the marker
            marker_center = tuple(np.mean(np.array(corner).reshape((-1, 2)), axis=0).astype(int))

            # Get the ID of the marker and convert it to a string
            marker_id = str(id_markers[i][0])  # Convert numpy.int32 to string

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

# input_image = "skin.png"
input_image = "/CT/SOMA/static00/S5/layers/original/skin.jpg"

# --

# json_fixed = "S5/uv_detections_charuco-suit/markers-skin-fixed.json"
# with open(json_fixed, "r") as f:
#     data_fixed = json.load(f)

# frame_fixed = cv2.imread(input_image)

# # Red markers
# output_image = Path("S5/debug/skin_charuco-suit_fixed.jpg")
# draw_debug_image_fixed(data_fixed["0"]["corners_markers"], data_fixed["0"]["id_markers"], frame_fixed)
# save_debug_image(output_image, frame_fixed)

# --

# json_manual = "S5/uv_detections_charuco-suit/markers-skin-manual.json"
# with open(json_manual, "r") as f:
#     data_manual = json.load(f)

# frame_manual = cv2.imread(input_image)

# # Green markers
# output_image = Path("S5/debug/skin_charuco-suit_manual.jpg")
# draw_debug_image_manual(data_manual["0"]["corners_markers"], data_manual["0"]["id_markers"], frame_manual)
# save_debug_image(output_image, frame_manual)

# --

# json_final_v1 = "S5/uv_detections_charuco-suit/markers-skin-final.json"
# with open(json_final_v1, "r") as f:
#     data_final_v1 = json.load(f)

# frame_final_v1 = cv2.imread(input_image)

# # Red markers
# output_image = Path("S5/debug/skin_charuco-suit_final.jpg")
# draw_debug_image_fixed(data_final_v1["0"]["corners_markers"], data_final_v1["0"]["id_markers"], frame_final_v1)
# save_debug_image(output_image, frame_final_v1)

# --

# json_manual_1_point = "uv_detections_charuco-suit/markers-skin-manual-1-point-corrected.json"
# with open(json_manual_1_point, "r") as f:
#     data_manual_1_point = json.load(f)

# frame_manual_1_point = cv2.imread(input_image)

# # Green markers for manual 1 point
# output_image = Path("debug/skin_charuco-suit_manual_1_point.jpg")
# draw_debug_image_manual(data_manual_1_point["0"]["corners_markers"], data_manual_1_point["0"]["id_markers"], frame_manual_1_point)
# save_debug_image(output_image, frame_manual_1_point)

# --

json_final_corrected = "S5/uv_detections_charuco-suit/markers-skin-final-corrected.json"
with open(json_final_corrected, "r") as f:
    data_final_corrected = json.load(f)

frame_final_corrected = cv2.imread(input_image)

# Green markers for manual 1 point
output_image = Path("S5/debug/skin_charuco-suit_final_corrected.jpg")
draw_debug_image_fixed(data_final_corrected["0"]["corners_markers"], data_final_corrected["0"]["id_markers"], frame_final_corrected)
save_debug_image(output_image, frame_final_corrected)