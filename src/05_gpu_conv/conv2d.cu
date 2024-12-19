#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(const uint8_t *pad_src, uint8_t *dst, const float *kernel,
                              int h, int w, int ch, int kh, int kw, int pad_width)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= w * h * ch)
        return;

    int x = tid % w;
    int y = (tid / w) % h;
    int c = tid / (w * h);

    int dst_idx = (y * w + x) * ch + c;

    float sum = 0.0f;
    for (int ky = 0; ky < kh; ky++)
    {
        for (int kx = 0; kx < kw; kx++)
        {
            int py = y + ky;
            int px = x + kx;
            int pad_idx = (py * pad_width + px) * ch + c;
            sum += pad_src[pad_idx] * kernel[ky * kw + kx];
        }
    }
    dst[dst_idx] = (uint8_t)fminf(255.0f, fmaxf(0.0f, sum));
}

extern "C" void conv2d(uint8_t *h_pad_src, uint8_t *h_dst, float *h_kernel,
                       int h, int w, int ch, int kh, int kw)
{
    int pad_width = w + (kw / 2) * 2;
    size_t src_size = (h + kh - 1) * pad_width * ch * sizeof(uint8_t);
    size_t dst_size = h * w * ch * sizeof(uint8_t);
    size_t kernel_size = kh * kw * sizeof(float);

    uint8_t *d_pad_src, *d_dst;
    float *d_kernel;

    cudaMalloc(&d_pad_src, src_size);
    cudaMalloc(&d_dst, dst_size);
    cudaMalloc(&d_kernel, kernel_size);
    cudaMemcpy(d_pad_src, h_pad_src, src_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    cudaEventRecord(start);

    int block = 128;
    int grid = (w * h *ch + block - 1) / block;
    conv2d_kernel<<<grid, block>>>(d_pad_src, d_dst, d_kernel,
                                  h, w, ch, kh, kw, pad_width);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("  execution time: %.3f ms\n", milliseconds);

    cudaMemcpy(h_dst, d_dst, dst_size, cudaMemcpyDeviceToHost);

    cudaFree(d_pad_src);
    cudaFree(d_dst);
    cudaFree(d_kernel);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

extern "C" void warmup()
{
    cudaFree(0);
}
