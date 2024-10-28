import numpy as np
from typing import Tuple

def conv2d(src : np.ndarray, kernel : np.ndarray, stride : Tuple = (1, 1)) -> np.ndarray:
    # check inputs
    assert src.ndim == 2 or src.ndim == 3
    assert kernel.ndim == 2
    assert len(stride) == 2

    if src.ndim == 2:
        src = np.expand_dims(src, axis=0)

    # prepare for processing
    h, w, ch = src.shape
    kh, kw = kernel.shape
    sh, sw = stride
    pad_w = kw // 2
    pad_h = kh // 2
    pad_src = np.zeros((h + pad_h*2, w + pad_w*2, ch), dtype=np.float32)
    pad_src[pad_h:h + pad_h, pad_w:w + pad_w, :] = src
    dst_w = int(w/ sw)
    dst_h = int(h/ sh)
    dst = np.zeros((dst_h, dst_w, ch), dtype=np.float32)

    # apply convolution
    for c in range(0, ch):
        for y in range(0, dst_h):
            for x in range(0, dst_w):
                for ky in range(0, kh):
                    for kx in range(0, kw):
                        dst[y, x, c] += kernel[ky, kx] * pad_src[y * sh + ky, x * sw + kx, c]

    # clip and convert dtype
    dst = np.clip(dst, 0, 255)
    dst = dst.astype(np.uint8)

    return dst
