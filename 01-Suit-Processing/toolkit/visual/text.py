from typing import Tuple

import cv2
import numpy as np


def draw_rounded_rectangle_with_text(
    img: np.ndarray,
    center: Tuple[int, int],
    height: int,
    color: Tuple[int, int, int],
    text: str,
    text_color: Tuple[int, int, int] = (0, 0, 0),
    radius: int = None,
    text_thickness: int = 1,
) -> np.ndarray:
    """
    Draw a rounded rectangle centered at a specific pixel and put a centered text inside it.

    Args:
        img : Input image as a NumPy array with shape (height, width, channels).
        center : Tuple (x, y) specifying the center of the rectangle.
        size : Tuple (width, height) of the rectangle.
        color : Tuple (B, G, R) specifying the color of the rectangle.
        thickness : Thickness of the rectangle edges. If -1, it fills the rectangle.
        text : The text to be placed in the center of the rectangle.
        radius : Radius of the rounded corner.
        text_color : Tuple (B, G, R) specifying the color of the text.
        text_thickness : Thickness of the text.

    Returns:
        The image with the rounded rectangle and centered text drawn on it.
    """
    if radius is None:
        radius = height // 10

    # Calculate the size of the text
    font_scale = 1.0
    font_face = cv2.FONT_HERSHEY_DUPLEX

    # Calculate the size of the text
    ((text_width, text_height), _) = cv2.getTextSize(text, font_face, font_scale, text_thickness)
    while text_height > height - 1.5 * radius:
        font_scale -= 0.1
        ((text_width, text_height), _) = cv2.getTextSize(text, font_face, font_scale, text_thickness)

    size = (text_width + 2.5 * radius, height)

    # Draw the rounded rectangle
    img = draw_rounded_rectangle(img, center, size, color, radius)

    # Calculate text position (centered in the rectangle)
    text_x = center[0] - text_width // 2
    text_y = center[1] + text_height // 2

    # Draw the text
    cv2.putText(img, text, (text_x, text_y), font_face, font_scale, text_color, text_thickness, cv2.LINE_AA)

    return img


def draw_rounded_rectangle(
    img: np.ndarray,
    center: Tuple[int, int],
    size: Tuple[int, int],
    color: Tuple[int, int, int],
    radius: int,
) -> np.ndarray:
    """
    Draw a rounded rectangle centered at a specific pixel.

    Args:
        img : Input image as a NumPy array.
        center : Tuple (x, y) specifying the center of the rectangle.
        size : Tuple (width, height) of the rectangle.
        color : Tuple (B, G, R) specifying the color of the rectangle.
        radius : Integer specifying the radius of the corner rounding.

    Returns:
        Modified image with the rounded rectangle drawn.
    """
    x, y = center
    width, height = size
    top_left = (int(x - width // 2), int(y - height // 2))
    bottom_right = (int(x + width // 2), int(y + height // 2))

    # Ensure that top_left and bottom_right are within image bounds
    top_left = (max(0, top_left[0]), max(0, top_left[1]))
    bottom_right = (min(img.shape[1] - 1, bottom_right[0]), min(img.shape[0] - 1, bottom_right[1]))

    # Define the rectangle corners to mask with circles for rounding
    rectangle = np.array(
        [
            [top_left[0] + radius, top_left[1]],
            [bottom_right[0] - radius, top_left[1]],
            [bottom_right[0], top_left[1] + radius],
            [bottom_right[0], bottom_right[1] - radius],
            [bottom_right[0] - radius, bottom_right[1]],
            [top_left[0] + radius, bottom_right[1]],
            [top_left[0], bottom_right[1] - radius],
            [top_left[0], top_left[1] + radius],
        ],
        dtype=np.int32,
    )

    # Draw rectangles to form the body
    cv2.fillPoly(img, [rectangle], color)

    # Draw corner circles
    cv2.circle(img, (top_left[0] + radius, top_left[1] + radius), radius, color, -1)
    cv2.circle(img, (bottom_right[0] - radius, top_left[1] + radius), radius, color, -1)
    cv2.circle(img, (bottom_right[0] - radius, bottom_right[1] - radius), radius, color, -1)
    cv2.circle(img, (top_left[0] + radius, bottom_right[1] - radius), radius, color, -1)

    return img
