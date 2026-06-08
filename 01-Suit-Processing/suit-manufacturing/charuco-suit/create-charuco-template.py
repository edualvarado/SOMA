"""
To make the template for the suit via sublimation.

DPI is 300
150cm width
Human size, up to 200 cm height

Example:
    python create-charuco-template.py -o suit-charuco-template-v1.pdf -ms 22 -ss 30 -n 1593 -si 0 -ei 1592 -dict DICT_4X4_1000

"""

import argparse
import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors

import os


def create_aruco_grid(output_path, marker_size_mm, square_size_mm, num_markers, start_id, end_id, aruco_dict_type):
    # Create the 'tags' directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')

    # Define the ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict_type))

    # Define page size and margins.
    # Convert mm to points (1 mm = 2.83465 points)
    # page_width, page_height = A4
    page_width = 1500 * mm
    page_height = 2000 * mm

    # Convert sizes to points
    marker_size = marker_size_mm * mm
    square_size = square_size_mm * mm

    # Calculate number of columns that can fit into the page width
    num_cols = int(page_width // square_size)

    # Calculate number of rows that can fit into the page height
    num_rows = int(page_height // square_size)

    total_width = num_cols * square_size
    total_height = num_rows * square_size

    # Calculate margins
    margin_horizontal = (page_width - total_width) / 2
    margin_vertical = (page_height - total_height) / 2

    # Calculate number of rows and columns based on the number of markers
    num_squares = num_markers * 2 - 1  # In a ChArUco board, the number of squares is twice the number of markers minus one
    # num_cols = int((page_width - 2 * margin) // square_size)
    num_rows = int(num_squares / num_cols) + (num_squares % num_cols > 0)

    # Create PDF
    c = canvas.Canvas(f"templates/{output_path}", pagesize=(page_width, page_height))

    current_id = start_id
    marker_count = 0
    for row in range(num_rows):
        for col in range(num_cols):
            if marker_count >= num_markers:
                break

            # Generate the marker image
            marker_img = np.zeros((int(marker_size / mm), int(marker_size / mm)), dtype=np.uint8)
            cv2.aruco.generateImageMarker(aruco_dict, current_id  % 1000, int(marker_size / mm), marker_img)

            # Save the marker image to a temporary file
            tmp_marker_path = f"templates/marker_{current_id % 1000}.png"
            cv2.imwrite(tmp_marker_path, marker_img)

            # Draw the square on the PDF
            square_x = margin_horizontal + col * square_size


            square_y = page_height - margin_vertical - (row + 1) * square_size

            c.rect(square_x, square_y, square_size, square_size, stroke=1, fill=(row + col) % 2)

            # Draw the marker centered inside the white squares
            if (row + col) % 2 == 0 and current_id <= end_id:
                marker_x = square_x + (square_size - marker_size) / 2
                marker_y = square_y + (square_size - marker_size) / 2
                c.drawImage(tmp_marker_path, marker_x, marker_y, marker_size, marker_size)
                current_id += 1
                marker_count += 1

        if current_id > end_id:
            break

    # Save the PDF
    c.save()

    # Cleanup temporary marker images
    for i in range(start_id, end_id + 1):
        tmp_marker_path = f"templates/marker_{i}.png"
        if os.path.exists(tmp_marker_path):
            os.remove(tmp_marker_path)

    print(f"ChArUco grid PDF saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create a grid-like ARUCO board on an A4 PDF")
    parser.add_argument("-o", "--output_path", type=str, help="Path to the output PDF file")
    parser.add_argument("-ms", "--marker_size_mm", type=float, help="Size of each ArUco marker (in millimeters)")
    parser.add_argument("-ss", "--square_size_mm", type=float,
                        help="Size of the square each marker is placed within (in millimeters)")
    parser.add_argument("-n", "--num_markers", type=int, help="Total number of ArUco markers")
    parser.add_argument("-si", "--start_id", type=int, help="Starting ID for the ArUco markers")
    parser.add_argument("-ei", "--end_id", type=int, help="Ending ID for the ArUco markers")
    parser.add_argument("-dict", "--aruco_dict_type", type=str, help="Type of ArUco dictionary (e.g., DICT_4X4_1000)")

    args = parser.parse_args()

    create_aruco_grid(
        args.output_path,
        args.marker_size_mm,
        args.square_size_mm,
        args.num_markers,
        args.start_id,
        args.end_id,
        args.aruco_dict_type
    )

if __name__ == "__main__":
    main()
