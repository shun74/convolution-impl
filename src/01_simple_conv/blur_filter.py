import sys
sys.path.append("src")

import os
from time import time

import numpy as np
import cv2

from basic_conv import conv2d
from common import compare_images

if __name__ == "__main__":
    # read
    img = cv2.imread("data/sample_1.png")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur_filter = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
    start = time()
    img_out = conv2d(img_gray, blur_filter)
    # gaussian_filter = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    # img_out = conv2d(img_gray, gaussian_filter)
    end = time()
    print(f"Elapsed time: {end - start :.4f} sec")

    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/blur.png", img_out)

    # compare with original
    compare_images(img_gray, img_out, "Original", "Convolution")
