"""
Script: manual_marker_annotator_1point.py
Goal: Interactive tool to manually annotate markers in an image.
"""

from tkinter import image_names

import cv2
import json
import numpy as np
import glob
import os

# --- Global Variables ---
current_marker_corners = []
all_markers_data = {"corners_markers": [], "id_markers": []}
image_display = None
drawing_image = None  # A copy to draw on
FRAME_ID = "0"  # Or prompt for it, or use image filename

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def mouse_callback(event, x, y, flags, param):
    global current_marker_corners, drawing_image

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(current_marker_corners) <= 1:
            current_marker_corners.append([x, y])  # Store as [X, Y]
            cv2.circle(drawing_image, (x, y), 5, (0, 255, 0), -1)  # Draw a green dot
            cv2.imshow("Annotate Markers", drawing_image)
            print(f"Corner {len(current_marker_corners)} added: ({x}, {y})")

            if len(current_marker_corners) == 1:
                print("1 corners selected. Press 'n' to enter ID, or 'c' to clear these 4 points.")
        else:
            print("Already 4 corners selected for the current marker. Press 'n' or 'c'.")

def main():
    print("================================")
    print("== 3. Manual Marker Annotator ==")
    print("================================")

    input_image = "S2/debug/skin_charuco-suit_final.jpg"
    output_json = "S2/uv_detections_charuco-suit/markers-skin-manual-1-point.json"
    # output_json = "S2/uv_detections_charuco-suit/markers-skin-manual-1-point-again.json"

    global current_marker_corners, all_markers_data, image_display, drawing_image


    image_display = cv2.imread(input_image)
    if image_display is None:
        print(f"Error: Could not load image from {input_image}")
        return

    drawing_image = image_display.copy()
    cv2.namedWindow("Annotate Markers")
    cv2.setMouseCallback("Annotate Markers", mouse_callback)

    print("--- Manual Annotation Tool ---")
    print("Click 4 points to define a marker's corners in order.")
    print("Press 'n' after 4 points to assign an ID and save the marker.")
    print("Press 'c' to clear the currently selected points for this marker.")
    print("Press 's' to save all annotated markers to JSON.")
    print("Press 'q' to quit (and discard unsaved annotations).")

    while True:
        cv2.imshow("Annotate Markers", drawing_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('c'): # Clear current marker's points
            current_marker_corners = []
            drawing_image = image_display.copy() # Reset drawing
            cv2.imshow("Annotate Markers", drawing_image)
            print("Current marker points cleared.")

        elif key == ord('n'):  # Finalize current marker
            if len(current_marker_corners) == 1:
                try:
                    marker_id_str = input("Enter Marker ID for the selected 4 corners: ")
                    marker_id = int(marker_id_str)  # Assuming ID is an integer

                    all_markers_data["corners_markers"].append(list(current_marker_corners))  # Save a copy
                    all_markers_data["id_markers"].append([marker_id])  # Match your [[ID]] format

                    # Draw the completed marker polygon (optional)
                    pts = np.array(current_marker_corners, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(drawing_image, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
                    cv2.putText(drawing_image, f"ID:{marker_id}",
                                (current_marker_corners[0][0], current_marker_corners[0][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    image_display = drawing_image.copy()  # Make drawn marker permanent for this session

                    print(f"Marker with ID {marker_id} and corners {current_marker_corners} saved locally.")
                    current_marker_corners = []  # Reset for next marker
                except ValueError:
                    print("Invalid ID entered. Please enter an integer.")
                except Exception as e:
                    print(f"An error occurred: {e}")
            else:
                print("Please select 4 corners before assigning an ID.")

        elif key == ord('s'):  # Save all collected annotations to JSON
            if not all_markers_data["corners_markers"]:
                print("No markers annotated yet to save.")
                continue

            output_data = {FRAME_ID: all_markers_data}
            try:
                save_json(output_data, output_json)
                print(f"Annotations successfully saved to {output_json}")
            except Exception as e:
                print(f"Error saving annotations to JSON: {e}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()