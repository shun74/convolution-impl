import numpy as np

def conv2d(src : np.ndarray, kernel : np.ndarray) -> np.ndarray:
    # check inputs
    assert kernel.ndim == 2
    assert src.ndim == 2

    # prepare for processing
    h, w = src.shape
    kh, kw = kernel.shape
    gap_w = kw // 2
    gap_h = kh // 2
    dst = np.ndarray((h-gap_h*2, w-gap_w*2), dtype=np.float32)

    # apply convolution
    for y in range(0, h-gap_h*2):
        for x in range(0, w-gap_w*2):
            for ky in range(0, kh):
                for kx in range(0, kw):
                    dst[y, x] += kernel[ky, kx] * src[y + ky, x + kx]

    dst = np.clip(dst, 0, 255)
    dst = dst.astype(np.uint8)
    return dst
