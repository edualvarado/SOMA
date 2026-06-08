from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from reportlab.lib.pagesizes import A3, A4, legal, letter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

paper_sizes = {
    "A4": A4,
    "A3": A3,
    "Letter": letter,
    "Legal": legal,
}


def save_to_pdf(
    image: np.ndarray,
    filename: Path,
    description: str,
    paper_size: str = "A4",
    ppi: Optional[int] = None,
    max_size_mm: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Saves an image to a PDF file with a specified paper size, allowing the user to specify the image's size in millimeters or PPI.

    Parameters:
        image: The image to be saved.
        filename: The name of the output PDF file (without extension).
        description: A description to be included in the PDF.
        paper_size: The paper size for the PDF ('A4', 'A3', 'Letter', 'Legal'). Default is 'A4'.
        ppi: The pixels per inch setting for the image. If not specified, defaults to 72 PPI for PDF.
        max_size_mm: The desired size of the largest dimension of the image in millimeters.

    Either max_size_mm or ppi must be specified, but not both.
    """
    if max_size_mm is None and ppi is None:
        raise ValueError("Either max_size_mm or ppi must be specified")
    if max_size_mm and ppi:
        raise ValueError("max_size_mm and ppi cannot be specified simultaneously")

    page_width_pt, page_height_pt = paper_sizes[paper_size]

    if max_size_mm:
        max_size_pt = mm_to_px(max_size_mm)
        ratio = max(image.shape) / max_size_pt
        img_width_pt = image.shape[1] / ratio
        img_height_pt = image.shape[0] / ratio
    else:
        img_width_pt = px_to_pt(image.shape[1], ppi=ppi)
        img_height_pt = px_to_pt(image.shape[0], ppi=ppi)

    c = canvas.Canvas(str(filename), pagesize=paper_sizes[paper_size])

    # If the image does not fit try to rotate it by 90 degrees
    if img_width_pt > page_width_pt or img_height_pt > page_height_pt:
        page_width_pt, page_height_pt = page_height_pt, page_width_pt
        print("Image cannot fit in the page, trying rotation it by 90 degrees")

        # Translate to the middle, rotate and translate back
        c.translate(page_height_pt / 2, page_width_pt / 2)
        c.rotate(90)
        c.translate(-page_width_pt / 2, -page_height_pt / 2)

    # It the problem persists, raise an error
    if img_width_pt > page_width_pt or img_height_pt > page_height_pt:
        raise ValueError(f"Image exceeds {paper_size} dimensions")

    x_offset = (page_width_pt - img_width_pt) / 2
    y_offset = (page_height_pt - img_height_pt) / 2

    temp_filepath = Path("temp.png")
    cv2.imwrite(str(temp_filepath), image)

    c.drawImage(temp_filepath, x_offset, y_offset, width=img_width_pt, height=img_height_pt)

    fontsize = 8

    str_width_pt, str_height_pt = get_str_dimensions(description, font_size=fontsize)

    x = page_width_pt / 2 - str_width_pt / 2
    y = y_offset / 2 - str_height_pt / 2

    c.setFont("Helvetica", fontsize)
    c.drawString(x, y, description)
    c.save()
    temp_filepath.unlink()


def mm_to_px(mm, ppi=72):
    return round(mm * ppi / 25.4)


def px_to_pt(px, ppi):
    return px * 72 / ppi


def get_str_dimensions(string, font="Helvetica", font_size=12):
    # Returns the size of a string in mm
    width = stringWidth(string, font, font_size)
    height = font_size
    return width, height
