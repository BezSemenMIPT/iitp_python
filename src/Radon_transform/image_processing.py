import numpy as np
from PIL import Image


def read_image(FOLDER, name):
    """Read image"""
    input_img = Image.open(FOLDER + name)
    input_img = input_img.convert("L")
    input_img = np.array(input_img)
    return input_img


def find_max_value(picture):
    """Find max value of intensity"""
    result = 0
    a, b = picture.shape
    s, t = 0.0, 0.0
    for i in range(a):
        for j in range(b):
            if picture[i, j] > result:
                result = picture[i, j]
                s = i
                t = j
    return result, tuple((s, t))


def find_point(n, s, t):
    """Find necessary point"""
    x = round((n - s) * n / t)
    return x
