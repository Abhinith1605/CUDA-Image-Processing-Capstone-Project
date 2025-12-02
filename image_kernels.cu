#include <cuda_runtime.h>
#include <stdint.h>
#include "colormap.hpp"

extern "C" {

__global__ void rgb_to_gray_kernel(const unsigned char* in, float* gray, int W, int H, int in_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * in_pitch + 3 * x;
    unsigned char b = in[idx + 0];
    unsigned char g = in[idx + 1];
    unsigned char r = in[idx + 2];
    gray[y * W + x] = 0.114f * b + 0.587f * g + 0.299f * r;
}

__global__ void normalize_kernel(float* gray, float minv, float maxv, unsigned char* out_norm, int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * W + x;
    float v = gray[idx];
    float nv = 0.0f;
    if (maxv > minv) nv = (v - minv) / (maxv - minv);
    if (nv < 0.0f) nv = 0.0f;
    if (nv > 1.0f) nv = 1.0f;
    out_norm[idx] = (unsigned char)(nv * 255.0f + 0.5f);
}

__constant__ unsigned char d_colormap[256*3];

__global__ void apply_colormap_kernel(const unsigned char* in_norm, unsigned char* out_rgb, int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * W + x;
    unsigned char v = in_norm[idx];
    int cmap_idx = (int)v * 3;
    out_rgb[3*idx + 0] = d_colormap[cmap_idx + 2];
    out_rgb[3*idx + 1] = d_colormap[cmap_idx + 1];
    out_rgb[3*idx + 2] = d_colormap[cmap_idx + 0];
}

void launch_copy_colormap_to_device()
{
    cudaMemcpyToSymbol(d_colormap, colormap_256, sizeof(colormap_256));
}

void launch_rgb_to_gray(const unsigned char* d_in, float* d_gray, int W, int H, int in_pitch, dim3 grid, dim3 block)
{
    rgb_to_gray_kernel<<<grid, block>>>(d_in, d_gray, W, H, in_pitch);
}

void launch_normalize(const float* d_gray, float minv, float maxv, unsigned char* d_out_norm, int W, int H, dim3 grid, dim3 block)
{
    normalize_kernel<<<grid, block>>>((float*)d_gray, minv, maxv, d_out_norm, W, H);
}

void launch_apply_colormap(const unsigned char* d_in_norm, unsigned char* d_out_rgb, int W, int H, dim3 grid, dim3 block)
{
    apply_colormap_kernel<<<grid, block>>>(d_in_norm, d_out_rgb, W, H);
}

} // extern "C"
