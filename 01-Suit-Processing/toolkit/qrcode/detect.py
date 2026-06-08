from datetime import datetime

import cv2
import pyzbar.pyzbar as pyzbar


def detect_date_from_qrcode(image, draw=False):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect QR codes in the image, knowing there will only be numbers
    qr_codes = pyzbar.decode(gray, symbols=[pyzbar.ZBarSymbol.QRCODE])

    # If at least one QR code is detected, return the data of the first one
    if qr_codes:
        qr_data = qr_codes[0].data.decode("utf-8")
        date = datetime.fromtimestamp(int(qr_data) / 1000)

        if draw:
            draw_qrcode_data(image, date.strftime("%Y-%m-%d %H:%M:%S_%f"), qr_codes[0].rect)

        return date

    # If no QR code is detected, return None
    return None


def draw_qrcode_data(image, text, rect, color=(73, 191, 252)):
    # Draw a rectangle around the detected QR code
    (x, y, w, h) = rect
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)

    # Write the QR data at the bottom of the image in a big font
    font_scale = 8
    thickness = 15
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = (image.shape[1] - text_width) // 2
    text_y = image.shape[0] - 20
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
