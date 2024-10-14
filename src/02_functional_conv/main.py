import sys
sys.path.append("src")

import os
from time import time

import numpy as np
import cv2

from functional_conv import conv2d
from common import compare_images

if __name__ == "__main__":
    # read
    img = cv2.imread("data/sample_1.png")
    bilateral_filter = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]]) / 2

    # apply convolution
    start = time()
    img_out = conv2d(img, bilateral_filter, (1, 1))
    end = time()
    print(f"Elapsed time: {end - start :.4f} sec")

    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/rgb.png", img_out)

    # compare with original
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_out_rgb = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
    compare_images(img_rgb, img_out_rgb, "Original", "Convolution", cmap=None)

    # stride
    start = time()
    img_out = conv2d(img, bilateral_filter, (2, 2))
    end = time()
    print(f"Elapsed time: {end - start :.4f} sec")

    cv2.imwrite("outputs/rgb_stride.png", img_out)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_out_rgb = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
    compare_images(img_rgb, img_out_rgb, "Original", "Convolution", cmap=None)
