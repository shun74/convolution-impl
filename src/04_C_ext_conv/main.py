import sys
sys.path.append("src")

from time import time

import cv2

from common import compare_images

import ctypes
from pathlib import Path
import numpy as np

LIBRARY_PATH = Path(__file__).parent / "conv2d.so"

def conv2d(src : np.ndarray, kernel : np.ndarray) -> np.ndarray:
    lib = ctypes.CDLL(LIBRARY_PATH)
    
    h, w, ch = src.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    
    pad_src = np.zeros((h + pad_h * 2, w + pad_w * 2, ch), dtype=np.uint8)
    pad_src[pad_h : h + pad_h, pad_w : w + pad_w, :] = src
    pad_src = np.ascontiguousarray(pad_src, dtype=np.uint8)
    kernel = np.ascontiguousarray(kernel, dtype=np.float32)
    dst = np.zeros((h, w, ch), dtype=np.uint8)
    
    lib.conv2d(
        pad_src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        kernel.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(h), ctypes.c_int(w), ctypes.c_int(ch),
        ctypes.c_int(kh), ctypes.c_int(kw)
    )
    
    return dst

if __name__ == "__main__":
    img = cv2.imread("data/sample_1.png")
    bilateral_filter = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]]) / 2

    start = time()
    img_out = conv2d(img, bilateral_filter)
    end = time()
    print(f"Elapsed time: {end - start :.4f} sec")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_out_rgb = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
    compare_images(img_rgb, img_out_rgb, "Original", "Convolution", cmap=None)
