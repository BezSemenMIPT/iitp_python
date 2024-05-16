import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from .image_processing import find_max_value, find_point
from .radon_transform import discrete_radon


def draw_line(picture, s, t):
    """Draw line using angle and offset"""
    image = picture.copy()
    height, width = np.array(picture).shape
    lineThickness = 2
    if s + t > height:
        x_0 = find_point(height, s, t)
        zero_matrix = np.zeros((height, width))
        stacked_upper = np.vstack([zero_matrix, picture])[s : s + height, :]
        stacked_lower = np.vstack([picture, zero_matrix])[s : s + height, :]
        max_up = (find_max_value(discrete_radon(stacked_upper)),)
        max_low = (find_max_value(discrete_radon(stacked_lower)),)

        if max_up[0] > max_low[0]:
            cv.line(
                image, (x_0, 0), (width - 1, s + t - height), (64, 0, 0), lineThickness
            )
        else:
            cv.line(image, (x_0, height), (0, s), (64, 0, 0), lineThickness)

    else:
        plt.imshow(
            cv.line(image, (0, s), (width - 1, s + t), (64, 0, 0), lineThickness)
        )
    return image


def detecting_lines(img, threshold_ratio=0.9):
    """Detecting lines on the image"""
    obraz = discrete_radon(img)
    max_intensity, _ = find_max_value(obraz)

    threshold = max_intensity * threshold_ratio

    lines = np.argwhere(obraz >= threshold)

    image_with_lines = img.copy()
    for s, t in lines:
        image_with_lines = draw_line(image_with_lines, s, t)
    plt.imshow(image_with_lines)
    return max_intensity, obraz
