#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

extern "C" {
void launch_copy_colormap_to_device();
void launch_rgb_to_gray(const unsigned char* d_in, float* d_gray, int W, int H, int in_pitch, dim3 grid, dim3 block);
void launch_normalize(const float* d_gray, float minv, float maxv, unsigned char* d_out_norm, int W, int H, dim3 grid, dim3 block);
void launch_apply_colormap(const unsigned char* d_in_norm, unsigned char* d_out_rgb, int W, int H, dim3 grid, dim3 block);
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cout << "Usage: GPU_Image_Filter <input_image> <output_image>\n";
        return 1;
    }

    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cout << "Failed to open input image: " << argv[1] << std::endl;
        return 1;
    }
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    int W = img.cols;
    int H = img.rows;
    size_t in_pitch = img.step; 

    unsigned char *d_in = nullptr;
    float *d_gray = nullptr;
    unsigned char *d_norm = nullptr;
    unsigned char *d_out = nullptr;

    size_t in_bytes = in_pitch * H;
    size_t gray_bytes = W * H * sizeof(float);
    size_t norm_bytes = W * H * sizeof(unsigned char);
    size_t out_bytes = W * H * 3 * sizeof(unsigned char);

    cudaMalloc(&d_in, in_bytes);
    cudaMalloc(&d_gray, gray_bytes);
    cudaMalloc(&d_norm, norm_bytes);
    cudaMalloc(&d_out, out_bytes);

    cudaMemcpy(d_in, img.data, in_bytes, cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((W + block.x -1)/block.x, (H + block.y -1)/block.y);

    launch_copy_colormap_to_device();
    launch_rgb_to_gray(d_in, d_gray, W, H, (int)in_pitch, grid, block);
    cudaDeviceSynchronize();

    std::vector<float> h_gray(W*H);
    cudaMemcpy(h_gray.data(), d_gray, gray_bytes, cudaMemcpyDeviceToHost);

    float minv = 1e30f, maxv = -1e30f;
    for (int i=0;i<W*H;i++){
        float v = h_gray[i];
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
    }
    if (minv==maxv) { maxv = minv + 1.0f; }

    launch_normalize(d_gray, minv, maxv, d_norm, W, H, grid, block);
    cudaDeviceSynchronize();

    launch_apply_colormap(d_norm, d_out, W, H, grid, block);
    cudaDeviceSynchronize();

    std::vector<unsigned char> h_out(W*H*3);
    cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost);

    cv::Mat out_img(H, W, CV_8UC3, h_out.data());
    cv::cvtColor(out_img, out_img, cv::COLOR_RGB2BGR);
    if (!cv::imwrite(argv[2], out_img)) {
        std::cout << "Failed to write output: " << argv[2] << std::endl;
    } else {
        std::cout << "Processing complete. Output written to: " << argv[2] << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_gray);
    cudaFree(d_norm);
    cudaFree(d_out);
    return 0;
}
