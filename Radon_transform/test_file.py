import numpy as np
import cv2
from image_processing import read_image
from draw import draw_line, detecting_lines

FOLDER = 'test_img/'

def find_not_zero(array):
    # If element != 0, then change it to 64
    new_array = np.where(array != 0, 64, array)
    return new_array

def find_not_zero_for_rgb(grayscale_image):
    grayscale_image = list(grayscale_image.flatten())
    set_a = np.unique(grayscale_image)
    dict = {}
    for i in  set_a:
        dict[i] = grayscale_image.count(i)

    max_key = max(dict, key=lambda k: dict[k])

    new_array = np.where(grayscale_image != max_key, 64, grayscale_image)
    return new_array

def compare_with_inuit_image(image_1, image_2, threshold = 0.7):
    image_1 = find_not_zero_for_rgb(image_1)
    # Comparing two pictures
    correlation = cv2.matchTemplate(image_1, image_2, cv2.TM_CCOEFF_NORMED)[0][0]

    # Is correlation value more  then threshold
    return correlation >= threshold

def compare_with_inuit_image_rgb(image_1, image_2, threshold = 0.7):
    image_1 = (image_1)
    # Comparing two pictures
    correlation = cv2.matchTemplate(image_1, image_2, cv2.TM_CCOEFF_NORMED)[0][0]

    # Is correlation value more  then threshold
    return correlation >= threshold

class TestClass:

    def test_blured(self):
        image = read_image(FOLDER, 'blured.png')
        _, _, image_with_lines = detecting_lines(image, draw = 1)
        assert(compare_with_inuit_image(image_with_lines, image) == True)

    def test_hor_elongated(self):
        image = read_image(FOLDER, 'hor_elongated.png')
        _, _, image_with_lines = detecting_lines(image, draw = 1)
        assert(compare_with_inuit_image(image_with_lines, image) == True)

    def test_hor_line(self):
        image = read_image(FOLDER, 'hor_line.png')
        _, _, image_with_lines = detecting_lines(image, draw = 1)
        assert(compare_with_inuit_image(image_with_lines, image) == True)

    def test_hor_small(self):
        image = read_image(FOLDER, 'hor_small.png')
        _, _, image_with_lines = detecting_lines(image, draw = 1)
        assert(compare_with_inuit_image(image_with_lines, image) == True)

    def test_intermittent(self):
        image = read_image(FOLDER, 'intermittent.png')
        _, _, image_with_lines = detecting_lines(image, draw = 1)
        assert(compare_with_inuit_image(image_with_lines, image) == True)

    def test_intermittent2(self):
        image = read_image(FOLDER, 'intermittent2.png')
        _, _, image_with_lines = detecting_lines(image, draw = 1)
        assert(compare_with_inuit_image(image_with_lines, image) == True)

    def test_intermittent3(self):
        image = read_image(FOLDER, 'intermittent3.png')
        _, _, image_with_lines = detecting_lines(image, draw = 1)
        assert(compare_with_inuit_image(image_with_lines, image) == True)

    def test_rectangle1(self):
        image = read_image(FOLDER, 'rectangle1.png')
        _, _, image_with_lines = detecting_lines(image, draw = 1)
        assert(compare_with_inuit_image(image_with_lines, image) == True)

    def test_rgb1(self):
        image = read_image(FOLDER, 'rgb1.png')
        _, _, image_with_lines = detecting_lines(image, draw = 1)
        assert(compare_with_inuit_image_rgb(image_with_lines, image) == True)

    def test_rgb2(self):
        image = read_image(FOLDER, 'rgb2.png')
        _, _, image_with_lines = detecting_lines(image, draw = 1)
        assert(compare_with_inuit_image_rgb(image_with_lines, image) == True)

    def test_rgb3(self):
        image = read_image(FOLDER, 'rgb3.png')
        _, _, image_with_lines = detecting_lines(image, draw = 1)
        assert(compare_with_inuit_image_rgb(image_with_lines, image) == True)

    def test_rotated(self):
        image = read_image(FOLDER, 'rotated.png')
        _, _, image_with_lines = detecting_lines(image, draw = 1)
        assert(compare_with_inuit_image(image_with_lines, image) == True)

    def test_vert_elongated(self):
        image = read_image(FOLDER, 'vert_elongated.png')
        _, _, image_with_lines = detecting_lines(image, draw = 1)
        assert(compare_with_inuit_image(image_with_lines, image) == True)

    def test_vert_line(self):
        image = read_image(FOLDER, 'vert_line.png')
        _, _, image_with_lines = detecting_lines(image, draw = 1)
        assert(compare_with_inuit_image(image_with_lines, image) == True)

    def test_vert_small(self):
        image = read_image(FOLDER, 'vert_small.png')
        _, _, image_with_lines = detecting_lines(image, draw = 1)
        assert(compare_with_inuit_image(image_with_lines, image) == True)

    def test_wide_line(self):
        image = read_image(FOLDER, 'wide_line.png')
        _, _, image_with_lines = detecting_lines(image, draw = 1)
        assert(compare_with_inuit_image(image_with_lines, image) == True)