import json
from math import ceil, floor
from pathlib import Path
from typing import Tuple, Union

import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt

from ..board import Board
from ..utils import save_to_pdf

GRAYSCALE_COLORS = [(243, 243, 243), (200, 200, 200), (160, 160, 160), (122, 122, 122), (85, 85, 85), (51, 51, 51)]
NATURAL_COLORS = [(115, 82, 68), (194, 150, 130), (98, 122, 157), (87, 108, 67), (133, 128, 177), (103, 189, 170)]
MISCELLANEOUS_COLORS = [(214, 126, 44), (80, 91, 166), (193, 90, 99), (94, 60, 108), (157, 188, 64), (224, 163, 46)]
PRIMARY_COLORS = [(56, 61, 150), (70, 148, 73), (175, 54, 60), (231, 199, 31), (187, 86, 149), (8, 133, 161)]
ALL_COLORS = GRAYSCALE_COLORS + NATURAL_COLORS + MISCELLANEOUS_COLORS + PRIMARY_COLORS


class ColorBoard(Board):
    """
    A class for creating a color board for color calibration.

    Args:
        marker_size: The size of the markers in mm.
        marker_sepation: The separation between the markers in mm.
        gridsize: The number of squares in the board in the x and y directions.
    """

    def __init__(
        self,
        marker_size: float,
        marker_spacing: float,
        gridsize: Tuple[int, int],
    ):
        self.marker_size = marker_size
        self.marker_spacing = marker_spacing
        self.gridsize = gridsize
        width_mm = gridsize[0] * marker_size + (gridsize[0] - 1) * marker_spacing
        height_mm = gridsize[1] * marker_size + (gridsize[1] - 1) * marker_spacing
        self.size_mm = (width_mm, height_mm)
        aruco_dict = get_dictionary(gridsize[0] * gridsize[1])

        self.board = aruco.GridBoard(gridsize, marker_size, marker_spacing, aruco_dict)

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
        image = self._overwrite_colors(image)
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.show()

    def save_image(self, path: Union[str, Path]):
        pixel_size = mm_to_px(self.size_mm[0]), mm_to_px(self.size_mm[1])
        image = self.board.generateImage(pixel_size)
        image = self._overwrite_colors(image)

        # Check if result is bigger than A4
        if self.size_mm[0] > 297 or self.size_mm[1] > 297:
            paper_size = "A3"
            print("Board is larger than A4, saving to A3")
        else:
            paper_size = "A4"

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        save_to_pdf(image_bgr, path / "board.pdf", str(self), max_size_mm=max(self.size_mm), paper_size=paper_size)
        print(f"Board image saved to {path / 'board.pdf'}")

    def _overwrite_colors(self, image):
        marker_size_px = (self.marker_size * 300) / 25.4
        marker_spacing_px = self.marker_spacing * 300 / 25.4
        square_size_px = marker_size_px + marker_spacing_px
        n_markers = self.gridsize[0] * self.gridsize[1]
        markers_to_skip = [0, n_markers // 2, n_markers - 1]

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        counter = 0
        for row in range(self.gridsize[1]):
            for col in range(self.gridsize[0]):
                index = row * self.gridsize[0] + col
                if index in markers_to_skip:
                    continue

                top_left = (floor(col * square_size_px), floor(row * square_size_px))
                bottom_center = (top_left[0] + ceil(marker_size_px / 2), top_left[1] + ceil(marker_size_px) + 1)
                top_center = (top_left[0] + ceil(marker_size_px / 2), top_left[1])
                bottom_right = (top_left[0] + ceil(marker_size_px) + 1, top_left[1] + ceil(marker_size_px) + 1)

                color_left = ALL_COLORS[counter]
                image[top_left[1] : bottom_center[1], top_left[0] : bottom_center[0]] = color_left

                color_right = ALL_COLORS[counter + 1]
                image[top_left[1] : bottom_center[1], top_center[0] : bottom_right[0]] = color_right

                counter += 2

        return image

    def to_json(self, path: Union[str, Path]):
        with open(path, "w") as file:
            json.dump(
                {
                    "board_type": "aruco",
                    "marker_size": self.marker_size,
                    "marker_spacing": self.marker_spacing,
                    "gridsize": self.gridsize,
                },
                file,
            )

    def __str__(self):
        return f"ColorBoard(marker_size={self.marker_size}, marker_sepation={self.marker_spacing}, gridsize={self.gridsize})"


def mm_to_px(mm, ppi=300):
    return round(mm * ppi / 25.4)


def get_dictionary(n_markers: int):
    if n_markers < 50:
        dict_id = aruco.DICT_6X6_50
    elif n_markers < 100:
        dict_id = aruco.DICT_6X6_100
    elif n_markers < 250:
        dict_id = aruco.DICT_6X6_250
    elif n_markers < 1000:
        dict_id = aruco.DICT_6X6_1000
    else:
        raise ValueError(f"Invalid number of markers: {n_markers}")

    return aruco.getPredefinedDictionary(dict_id)
