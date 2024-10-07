import os
import numpy as np
import cv2

from .basic_conv import conv2d
from ..common import compare_images

if __name__ == "__main__":
    # read
    img = cv2.imread("data/sample_1.png")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bilateral_filter = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    img_out = conv2d(img_gray, bilateral_filter)

    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/bilateral.png", img_out)

    # compare with original
    compare_images(img_gray, img_out, "Original", "Convolution")
