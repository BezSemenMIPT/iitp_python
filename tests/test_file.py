import adrt
from click.testing import CliRunner
import cv2
import numpy as np
import pytest
from skimage.draw import line

from Radon_transform import main
from Radon_transform.radon_transform import discrete_radon


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_main_succeeds():
    """Assert cli working"""
    runner = CliRunner()
    result = runner.invoke(main.main, "inter_lines.jpg")
    assert result.exit_code == 0


# Additional functions to check containing one picture inside anoter and create a pair
# of radon transform picture fro package adrt and using my own radon_transform function
# =========================================================================


def check_containing(img1, img2):
    if img1.shape[0] >= img2.shape[0] and img1.shape[1] >= img2.shape[1]:
        mTempl = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(mTempl)
        if max_val > 0.90:
            return True
    return False


def create_r_pair(image):
    res_adrt = adrt.adrt(image)
    res_adrt = adrt.utils.stitch_adrt(res_adrt)
    cv2.imwrite("img.jpg", res_adrt)
    res_adrt = cv2.imread("img.jpg", cv2.IMREAD_GRAYSCALE)

    res_i = discrete_radon(image)

    cv2.imwrite("img.jpg", res_i)
    res_i = cv2.imread("img.jpg", cv2.IMREAD_GRAYSCALE)

    return res_i, res_adrt


# Creating lines for testing detecting
# =========================================================


def create_blank_image(width, height):
    return np.zeros((height, width), dtype=np.uint8)


def test_radon_transform_intersecting_lines():
    img = create_blank_image(100, 100)
    rr1, cc1 = line(10, 10, 90, 90)
    rr2, cc2 = line(10, 90, 90, 10)
    img[rr1, cc1] = 255
    img[rr2, cc2] = 255
    cv2.imwrite("inter_lines.jpg", img)


def create_three_lines_from_one_point_image(width, height):
    img = create_blank_image(width, height)
    rr1, cc1 = line(10, 10, 90, 50)
    rr2, cc2 = line(10, 10, 10, 90)
    rr3, cc3 = line(10, 10, 50, 90)
    img[rr1, cc1] = 255
    img[rr2, cc2] = 255
    img[rr3, cc3] = 255
    cv2.imwrite("three lines.jpg", img)


def create_line_image(width, height, thickness):
    img = np.zeros((height, width), dtype=np.uint8)
    cv2.line(img, (10, 10), (90, 90), 255, thickness)
    cv2.imwrite("one line.jpg", img)


# =========================================================


class Test_ImageSizes:

    def test_square(self):
        image = np.diag(np.ones(200))
        res = discrete_radon(image)

        assert len(res) == len(image)

    def test_rectangle(self):
        image = np.zeros([100, 200, 3], dtype=np.uint8)
        for i in range(len(image)):
            image[i, i] = [255, 128, 0]

        res = discrete_radon(image)

        assert len(res) == len(image)

    def test_very_small(self):
        image = np.diag(np.ones(3))
        res = discrete_radon(image)

        assert len(res) == len(image)

    def test_2N(self):
        image = np.diag(np.ones(2**7))
        res = discrete_radon(image)
        assert len(res) == len(image)

    def test_2N_p1(self):
        image = np.diag(np.ones(2**6 + 1))
        res = discrete_radon(image)
        assert len(res) == len(image)

    def test_2N_m1(self):
        image = np.diag(np.ones(2**6 - 1))
        res = discrete_radon(image)
        assert len(res) == len(image)

    def test_k2N(self):
        image = np.diag(np.ones(2**5 * 5))
        res = discrete_radon(image)
        assert len(res) == len(image)


class Test_Format:

    def test_grayscale(self):
        image = np.zeros([100, 200, 3], dtype=np.uint8)
        for i in range(len(image)):
            image[i, i] = [255, 128, 0]
        cv2.imwrite("img.jpg", image)
        image = cv2.imread("img.jpg", cv2.IMREAD_GRAYSCALE)
        res = discrete_radon(image)
        assert len(res) == len(image)

    def test_rgbscale(self):
        image = np.zeros([100, 100, 3], dtype=np.uint8)
        for i in range(len(image)):
            image[i, i] = [255, 128, 0]
        cv2.imwrite("img.jpg", image)
        image = cv2.imread("img.jpg", cv2.IMREAD_COLOR)
        res = discrete_radon(image)
        assert len(res) == len(image)

    def test_png(self):

        image = np.diag(np.ones(128))

        res_adrt = adrt.adrt(image)
        res_adrt = adrt.utils.stitch_adrt(res_adrt)
        cv2.imwrite("img.png", res_adrt)
        res_adrt = cv2.imread("img.png", cv2.IMREAD_GRAYSCALE)
        res_i = discrete_radon(image)

        cv2.imwrite("img.png", res_i)
        res_i = cv2.imread("img.png", cv2.IMREAD_GRAYSCALE)

        assert check_containing(res_adrt, res_i) == True

    def test_jpg(self):
        image = np.diag(np.ones(128))
        res_adrt = adrt.adrt(image)
        res_adrt = adrt.utils.stitch_adrt(res_adrt)
        cv2.imwrite("img.jpg", res_adrt)
        res_adrt = cv2.imread("img.jpg", cv2.IMREAD_GRAYSCALE)
        res_i = discrete_radon(image)
        res_i = cv2.imread("img.jpg", cv2.IMREAD_GRAYSCALE)
        assert check_containing(res_adrt, res_i) == True


class Test_Lines:

    @pytest.mark.parametrize(
        "width, coord",
        [
            (1, [(10, 0), (100, 50)]),
            (3, [(20, 50), (80, 40)]),
            (7, [(20, 30), (120, 90)]),
            (8, [(10, 0), (100, 50)]),
        ],
    )
    def test_multi_width(self, width, coord):
        image = np.zeros((128, 128))
        cv2.line(
            image, pt1=coord[0], pt2=coord[1], thickness=width, color=(255, 255, 255)
        )

        res_i, res_adrt = create_r_pair(image)

        assert check_containing(res_adrt, res_i) == True

    def test_clear_img(self):

        image = np.zeros((128, 128))

        res_i, res_adrt = create_r_pair(image)

        assert check_containing(res_adrt, res_i) == True

    @pytest.mark.parametrize(
        "width, coord",
        [
            (1, [(10, 0), (100, 50), (20, 50), (80, 40), (20, 30), (120, 90)]),
            (1, [(10, 20), (80, 50), (20, 70), (80, 40), (20, 33), (110, 75)]),
            (1, [(30, 40), (100, 50), (50, 50), (70, 40), (20, 30), (120, 90)]),
            (1, [(10, 80), (100, 50), (10, 50), (90, 10), (20, 20), (65, 80)]),
        ],
    )
    def test_multi_lines(self, width, coord):

        image = np.zeros((128, 128))
        cv2.line(
            image, pt1=coord[0], pt2=coord[1], color=(255, 255, 255), thickness=width
        )
        cv2.line(
            image, pt1=coord[2], pt2=coord[3], color=(255, 255, 255), thickness=width
        )
        cv2.line(
            image, pt1=coord[4], pt2=coord[5], color=(255, 255, 255), thickness=width
        )

        res_i, res_adrt = create_r_pair(image)

        assert check_containing(res_adrt, res_i) == True

    @pytest.mark.parametrize(
        "width, coord",
        [
            (1, [(10, 0), (100, 0)]),
            (1, [(0, 20), (0, 130)]),
            (1, [(20, 30), (120, 90)]),
            (1, [(140, 0), (10, 140)]),
            (1, [(50, 80), (70, 140)]),
            (1, [(110, 80), (50, 90)]),
            (1, [(0, 80), (0, 150)]),
            (1, [(110, 0), (50, 0)]),
        ],
    )
    def test_multi_angles(self, width, coord):

        image = np.zeros((256, 256))
        cv2.line(
            image, pt1=coord[0], pt2=coord[1], thickness=width, color=(255, 255, 255)
        )

        res_i, res_adrt = create_r_pair(image)

        assert check_containing(res_adrt, res_i) == True
