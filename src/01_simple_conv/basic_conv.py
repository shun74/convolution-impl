import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def simple_conv2d(kernel : np.ndarray, src : np.ndarray) -> np.ndarray:
    # check inputs
    assert kernel.ndim == 2
    assert src.ndim == 2
    assert kernel.shape[0] == kernel.shape[1]

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

if __name__ == "__main__":
    # read
    img = cv2.imread("data/sample_1.png")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur_kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
    img_out = simple_conv2d(blur_kernel, img_gray)

    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/output.png", img_out)

    # compare with original
    plt.subplot(1, 2, 1)
    plt.imshow(img_gray, cmap="gray")
    plt.title("Original")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(img_out, cmap="gray")
    plt.title("Convolution")
    plt.axis("off")
    plt.show()
