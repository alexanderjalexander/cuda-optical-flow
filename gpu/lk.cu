#include <stdio.h>

#include "lk.cuh"

#define SOBEL_BLOCK_WIDTH 16

/**
 *
 */
__global__ void
sobelFilter(unsigned char *d_frame, int width, int height)
{
}

/**
 *
 */
void
processFrameOnGPU(const cv::Mat &frame)
{
    int width = frame.cols;
    int height = frame.rows;
    int size = width * height * sizeof(unsigned char);

    unsigned char *d_frame = NULL;
    cudaMalloc(&d_frame, width * height * sizeof(unsigned char));
    cudaMemcpy(d_frame, frame.data, size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16, 1);
    dim3 gridDim((int)ceil(width / blockDim.x), (int)ceil(height / blockDim.y), 1);

    // TODO: Kernel for Sobel Operator
    // TODO: Kernel for Harris/Shi-Tomasi/FAST Detector

    // Fast: https://en.wikipedia.org/wiki/Features_from_accelerated_segment_test

    // Harris: https://en.wikipedia.org/wiki/Harris_corner_detector

    // Shi-Tomasi:
    // https://en.wikipedia.org/wiki/Corner_detection#The_Harris_&_Stephens_/_Shi%E2%80%93Tomasi_corner_detection_algorithms

    sobelFilter<<<gridDim, blockDim>>>(d_frame, width, height);

    cudaDeviceSynchronize();
    cudaMemcpy(frame.data, d_frame, size, cudaMemcpyDeviceToHost);
    cudaFree(d_frame);
}