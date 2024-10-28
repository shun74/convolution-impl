import numpy as np

def im2col_fast(img, kernel_h, kernel_w):
    # convert img (c, h, w) to col (c, kernel_h * kernel_w, (h - 2*(kernel_h//2)) * (w - 2*(kernel_w//2)))
    # consider padding as done
    ch, h, w = img.shape
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    
    col_img = np.zeros((ch, kernel_h * kernel_w, (h - 2 * pad_h) * (w - 2 * pad_w)), dtype=np.float32)

    for c in range(ch):
        for ky in range(kernel_h):
            for kx in range(kernel_w):
                kernel_pixels = img[c, ky:ky + h - 2*pad_h, kx:kx + w - 2*pad_w]
                col_img[c, kx + ky * (kernel_w), :] = kernel_pixels.reshape((h - 2*pad_h) * (w - 2*pad_w))
    
    return col_img

def conv2d(src : np.ndarray, kernel : np.ndarray):
    assert src.ndim == 3 or src.ndim == 2
    assert kernel.ndim == 2
    
    h, w, ch = src.shape
    kh, kw = kernel.shape

    trans_src = src.transpose(2, 0, 1)

    pad_h = kh // 2
    pad_w = kw // 2

    pad_src = np.zeros((ch, h + pad_h * 2, w + pad_w * 2), dtype=np.float32)
    pad_src[:, pad_h : h + pad_h, pad_w : w + pad_w] = trans_src

    flatten_kernel = kernel.reshape(kh * kw)
    col_src = im2col_fast(pad_src, kh, kw)
    col_dst = flatten_kernel @ col_src
    trans_dst = col_dst.reshape(ch, h, w)
    dst = trans_dst.transpose(1, 2, 0)

    dst = np.clip(dst, 0, 255)
    dst = dst.astype(np.uint8)

    return dst