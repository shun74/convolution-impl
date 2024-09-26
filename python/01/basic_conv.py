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
    dst = np.ndarray((h, w), dtype=np.float32)

    # apply convolution
    for y in range(0, h):
        for x in range(0, w):
            for ky in range(0, kh):
                for kx in range(0, kw):
                    dy = ky - kh // 2
                    dx = kx - kw // 2
                    if y + dy < 0 or y + dy >= h or x + dx < 0 or x + dx >= w:
                        continue
                    dst[y, x] += kernel[dy, dx] * src[y + dy, x + dx]
    return dst

if __name__ == "__main__":
    # read
    img = cv2.imread("data/sample_1.png")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur_kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
    img_out = simple_conv2d(blur_kernel, img_gray)

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
