#include <stdint.h>

void conv2d(uint8_t* pad_src, uint8_t* dst, float* kernel,
              int h, int w, int ch, int kh, int kw) {

    int pad_w = kw / 2;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int c = 0; c < ch; c++) {
                float sum = 0;
                for (int ky = 0; ky < kh; ky++) {
                    for (int kx = 0; kx < kw; kx++) {
                        sum += pad_src[((y + ky)* (w + pad_w * 2) + x + kx) * ch + c] * kernel[ky * kw + kx];
                    }
                }
                sum = sum < 0 ? 0 : sum;
                sum = sum > 255 ? 255 : sum;
                dst[(y * w + x) * ch + c] = (uint8_t)sum;
            }
        }
    }
}
