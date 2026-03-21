#include <stdio.h>

#include "lk.cuh"

/**
 * @param d_frame_res  Output buffer of gradient magnitudes (float, device memory).
 *                     Must be pre-allocated to width * height * sizeof(float).
 * @param d_frame      Input grayscale image (unsigned char, device memory).
 *                     Must be pre-allocated to width * height * sizeof(unsigned char).
 * @param width        Width of the frame in pixels.
 * @param height       Height of the frame in pixels.
 */
__global__ void
sobelFilter(float *d_frame_res, unsigned char *d_frame, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int dx, dy;

    if (x > 0 && y > 0 && x < width - 1 && y < height - 1)
    {
        dx = (-1 * d_frame[(y - 1) * width + (x - 1)]) + (-2 * d_frame[y * width + (x - 1)]) +
             (-1 * d_frame[(y + 1) * width + (x - 1)]) + (d_frame[(y - 1) * width + (x + 1)]) +
             (2 * d_frame[y * width + (x + 1)]) + (d_frame[(y + 1) * width + (x + 1)]);
        dy = (d_frame[(y - 1) * width + (x - 1)]) + (2 * d_frame[(y - 1) * width + x]) +
             (d_frame[(y - 1) * width + (x + 1)]) + (-1 * d_frame[(y + 1) * width + (x - 1)]) +
             (-2 * d_frame[(y + 1) * width + x]) + (-1 * d_frame[(y + 1) * width + (x + 1)]);
        d_frame_res[y * width + x] = sqrtf((dx * dx) + (dy * dy));
    }
}


/**
 * @param frame   Input grayscale frame (CV_8UC1, host memory). Must be
 *                single-channel; behavior is undefined for multi-channel input.
 * @param result  Output Mat that receives processed LK frame from GPU.
 *                Created or resized by this function via result.create().
 */
void
lucasKanadeSingleFrameGPU(const cv::Mat &frame, cv::Mat &result)
{
    int width = frame.cols;
    int height = frame.rows;
    int size = width * height * sizeof(unsigned char);

    unsigned char *d_frame = NULL;
    float *d_frame_res = NULL;
    cudaMalloc(&d_frame, width * height * sizeof(unsigned char));
    cudaMalloc(&d_frame_res, width * height * sizeof(float));
    cudaMemcpy(d_frame, frame.data, size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16, 1);
    dim3 gridDim((int)ceil((float)width / blockDim.x), (int)ceil((float)height / blockDim.y), 1);

    // TODO: Kernel for Harris/Shi-Tomasi/FAST Detector

    // Fast: https://en.wikipedia.org/wiki/Features_from_accelerated_segment_test
    // Harris: https://en.wikipedia.org/wiki/Harris_corner_detector
    // Shi-Tomasi:
    // https://en.wikipedia.org/wiki/Corner_detection#The_Harris_&_Stephens_/_Shi%E2%80%93Tomasi_corner_detection_algorithms

    /*
    Sparse LK Algorithm:
    - Obtain Ix, Iy using Sobel
    - Obtain It using simple frame subtraction
    - Obtain features (in this case, corners w/ Harris/FAST)
    - For each feature:
      - Sum Ix^2, Iy^2, IxIy, IxIt, IyIt over a local window (5x5 preferably)
    - Do the Ax = b inverted multiply, which is x = A^-1 b
    - Obtain u, v from x, and then result that
     */

    sobelFilter<<<gridDim, blockDim>>>(d_frame_res, d_frame, width, height);

    // cudaDeviceSynchronize();
    result.create(height, width, CV_32F);
    cudaMemcpy(result.data, d_frame_res, width * height * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(d_frame);
    cudaFree(d_frame_res);
}