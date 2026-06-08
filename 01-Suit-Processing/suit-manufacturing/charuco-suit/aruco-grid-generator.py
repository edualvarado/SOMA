"""
To create grids for the Transfer paper.

Example:

python aruco-grid-generator.py -o transfer-0-27-new.pdf -ms 22 -ss 30 -n 28 -si 0 -ei 27 -dict DICT_4X4_1000
python aruco-grid-generator.py -o transfer-28-55-new.pdf -ms 22 -ss 30 -n 28 -si 28 -ei 55 -dict DICT_4X4_1000
python aruco-grid-generator.py -o transfer-56-83-new.pdf -ms 22 -ss 30 -n 28 -si 56 -ei 83 -dict DICT_4X4_1000
python aruco-grid-generator.py -o transfer-84-111-new.pdf -ms 22 -ss 30 -n 28 -si 84 -ei 111 -dict DICT_4X4_1000
python aruco-grid-generator.py -o transfer-112-139-new.pdf -ms 22 -ss 30 -n 28 -si 112 -ei 139 -dict DICT_4X4_1000
python aruco-grid-generator.py -o transfer-140-167-new.pdf -ms 22 -ss 30 -n 28 -si 140 -ei 167 -dict DICT_4X4_1000
python aruco-grid-generator.py -o transfer-168-195-new.pdf -ms 22 -ss 30 -n 28 -si 168 -ei 195 -dict DICT_4X4_1000
python aruco-grid-generator.py -o transfer-196-223-new.pdf -ms 22 -ss 30 -n 28 -si 196 -ei 223 -dict DICT_4X4_1000
python aruco-grid-generator.py -o transfer-224-251-new.pdf -ms 22 -ss 30 -n 28 -si 224 -ei 251 -dict DICT_4X4_1000
python aruco-grid-generator.py -o transfer-252-279-new.pdf -ms 22 -ss 30 -n 28 -si 252 -ei 279 -dict DICT_4X4_1000
python aruco-grid-generator.py -o transfer-280-307-new.pdf -ms 22 -ss 30 -n 28 -si 280 -ei 307 -dict DICT_4X4_1000
python aruco-grid-generator.py -o transfer-308-335-new.pdf -ms 22 -ss 30 -n 28 -si 308 -ei 335 -dict DICT_4X4_1000
python aruco-grid-generator.py -o transfer-336-363-new.pdf -ms 22 -ss 30 -n 28 -si 336 -ei 363 -dict DICT_4X4_1000
python aruco-grid-generator.py -o transfer-364-391-new.pdf -ms 22 -ss 30 -n 28 -si 364 -ei 391 -dict DICT_4X4_1000
python aruco-grid-generator.py -o transfer-392-419-new.pdf -ms 22 -ss 30 -n 28 -si 392 -ei 419 -dict DICT_4X4_1000
python aruco-grid-generator.py -o transfer-420-447-new.pdf -ms 22 -ss 30 -n 28 -si 420 -ei 447 -dict DICT_4X4_1000
python aruco-grid-generator.py -o transfer-448-475-new.pdf -ms 22 -ss 30 -n 28 -si 448 -ei 475 -dict DICT_4X4_1000
python aruco-grid-generator.py -o transfer-476-503-new.pdf -ms 22 -ss 30 -n 28 -si 476 -ei 503 -dict DICT_4X4_1000
python aruco-grid-generator.py -o transfer-504-531-new.pdf -ms 22 -ss 30 -n 28 -si 504 -ei 531 -dict DICT_4X4_1000
python aruco-grid-generator.py -o transfer-532-559-new.pdf -ms 22 -ss 30 -n 28 -si 532 -ei 559 -dict DICT_4X4_1000
python aruco-grid-generator.py -o transfer-560-587-new.pdf -ms 22 -ss 30 -n 28 -si 560 -ei 587 -dict DICT_4X4_1000


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
    # Validate inputs
    # assert num_markers == (end_id - start_id + 1), "Number of IDs must equal the number of ArUco markers."
    # assert marker_size_mm <= square_size_mm, "Marker size must be less than or equal to square size."

    # Create the 'tags' directory if it doesn't exist
    if not os.path.exists('tags'):
        os.makedirs('tags')

    # Define the ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict_type))

    # Define page size and margins
    page_width, page_height = A4
    margin = 5 * mm

    # Convert sizes to points
    marker_size = marker_size_mm * mm
    square_size = square_size_mm * mm
    # distance_between_markers = square_size
    rectangle_size = 40 * mm  # Size of the rectangles in points

    # Calculate number of rows and columns
    total_marker_size = square_size
    # num_cols = int((page_width - 2 * margin + distance_between_markers) // total_marker_size)
    # num_rows = int((page_height - 2 * margin + distance_between_markers) // total_marker_size)
    num_cols = int((page_width - 2 * margin) // rectangle_size)
    num_rows = int((page_height - 2 * margin) // rectangle_size)

    # Calculate total content height
    total_content_height = num_rows * rectangle_size

    # Adjust margins for centering horizontally
    # horizontal_margin = (page_width - (num_cols * total_marker_size)) / 2
    horizontal_margin = (page_width - (num_cols * rectangle_size)) / 2

    # Calculate total vertical whitespace
    total_vertical_whitespace = page_height - total_content_height

    # Calculate top and bottom margins
    top_margin = bottom_margin = total_vertical_whitespace / 2

    # Create PDF
    c = canvas.Canvas(f"tags/{output_path}", pagesize=A4)

    current_id = start_id
    for row in range(num_rows):
        for col in range(num_cols):
            if current_id > end_id:
                break

            # Generate the marker image
            marker_img = np.zeros((int(marker_size / mm), int(marker_size / mm)), dtype=np.uint8)
            cv2.aruco.generateImageMarker(aruco_dict, current_id, int(marker_size / mm), marker_img)

            # Save the marker image to a temporary file
            tmp_marker_path = f"tags/marker_{current_id}.png"
            cv2.imwrite(tmp_marker_path, marker_img)

            # ---

            # # Draw the square around the marker on the PDF
            # square_x = horizontal_margin + col * total_marker_size
            # square_y = page_height - margin - (row + 1) * total_marker_size
            #
            # # Generate a very thin rectangle around the marker with thickness of 0.5 mm
            # c.rect(square_x, square_y, square_size, square_size, stroke=1)

            # ---

            # # Draw the square around the marker on the PDF
            # square_x = horizontal_margin + col * total_marker_size
            # square_y = page_height - margin - (row + 1) * total_marker_size

            # Draw the points at the corners of the square
            # point_radius = 1  # Radius of the points in mm
            # c.setFillColor(colors.Color(0.1, 0.1, 0.1))  # Set the fill color to grey
            # for dx, dy in [(0, 0), (0, square_size), (square_size, 0), (square_size, square_size)]:
            #     point_x = square_x + dx
            #     point_y = square_y + dy
            #     c.circle(point_x, point_y, point_radius, fill=1)
            # c.setFillColor(colors.black)

            # ---

            # Draw the rectangle on the PDF
            rectangle_x = horizontal_margin + col * rectangle_size
            rectangle_y = page_height - top_margin - (row + 1) * rectangle_size
            c.rect(rectangle_x, rectangle_y, rectangle_size, rectangle_size, stroke=1)

            # Draw the square around the marker on the PDF
            square_x = horizontal_margin + col * total_marker_size
            square_y = page_height - margin - (row + 1) * total_marker_size

            # Draw the rotated crosses at the corners of the square
            cross_size = 1 * mm  # Size of the cross in mm
            # c.setStrokeColor(colors.Color(0.8, 0.8, 0.8))  # Set the stroke color to grey
            for dx, dy in [(0, 0), (0, square_size), (square_size, 0), (square_size, square_size)]:
                point_x = rectangle_x + dx + (rectangle_size - square_size) / 2
                point_y = rectangle_y + dy + (rectangle_size - square_size) / 2
                c.line(point_x, point_y - cross_size / 2, point_x, point_y + cross_size / 2)  # Vertical line
                c.line(point_x - cross_size / 2, point_y, point_x + cross_size / 2, point_y)  # Horizontal line
            # ---

            # Draw the marker centered inside the square
            marker_x = rectangle_x + (rectangle_size - marker_size) / 2
            marker_y = rectangle_y + (rectangle_size - marker_size) / 2
            c.drawImage(tmp_marker_path, marker_x, marker_y, marker_size, marker_size)

            current_id += 1

        if current_id > end_id:
            break

    # Save the PDF
    c.save()

    # Cleanup temporary marker images
    for i in range(start_id, end_id + 1):
        tmp_marker_path = f"tags/marker_{i}.png"
        if os.path.exists(tmp_marker_path):
            os.remove(tmp_marker_path)

    print(f"ArUco grid PDF saved to {output_path}")


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

    # create_aruco_grid(
    #     args.output_path,
    #     args.marker_size_mm,
    #     args.square_size_mm,
    #     args.num_markers,
    #     args.start_id,
    #     args.end_id,
    #     args.aruco_dict_type
    # )

    start = 0
    end = 27
    step = 28
    count = 0

    while count < 21:  # Adjust this value based on your needs
        output_path = f"transfer-{start}-{end}-new.pdf"
        create_aruco_grid(
            output_path,
            args.marker_size_mm,
            args.square_size_mm,
            args.num_markers,
            start,
            end,
            args.aruco_dict_type
        )
        start += step
        end += step
        count += 1


if __name__ == "__main__":
    main()
