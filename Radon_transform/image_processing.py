import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image


def read_image(FOLDER, name):
    # final_img = cv.imread(FOLDER + name, cv.IMREAD_GRAYSCALE)
    input_img = Image.open(FOLDER + name)
    img = input_img.convert("L")
    # img = img.reduce(4)
    img = img.resize((100, 100))
    image_matrix = np.matrix(img)
    resized_img = np.ones(image_matrix.shape) * 255 - image_matrix

    # h, w = final_img.shape[:2]

    # new_h = 2 ** int(np.ceil(np.log2(h)))
    # new_w = 2 ** int(np.ceil(np.log2(w)))

    # resized_img = cv.resize(final_img, (new_w, new_h), interpolation=cv.INTER_LINEAR)

    # plt.imshow(resized_img)
    return resized_img


def find_max_value(picture):
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
    x = round((n - s) * n / t)
    return x
