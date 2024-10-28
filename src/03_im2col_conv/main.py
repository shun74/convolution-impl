import sys
sys.path.append("src")

import os
from time import time

import numpy as np
import cv2

import im2col_fast
import im2col_naive
from common import compare_images

if __name__ == "__main__":
    img = cv2.imread("data/sample_1.png")
    bilateral_filter = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]]) / 2

    # naive
    start = time()
    img_out = im2col_naive.conv2d(img, bilateral_filter)
    end = time()
    print(f"Naive im2col elapsed time: {end - start :.4f} sec")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_out_rgb = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
    compare_images(img_rgb, img_out_rgb, "Original", "Convolution", cmap=None)

    # fast
    start = time()
    img_out = im2col_fast.conv2d(img, bilateral_filter)
    end = time()
    print(f"Fast im2col elapsed time: {end - start :.4f} sec")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_out_rgb = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
    compare_images(img_rgb, img_out_rgb, "Original", "Convolution", cmap=None)
