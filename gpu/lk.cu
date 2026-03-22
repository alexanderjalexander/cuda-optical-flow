#include <stdio.h>

#include "lk.cuh"

__global__ void
sobelFilter(int *ix, int *iy, unsigned char *d_frame, int width, int height)
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

        ix[y * width + x] = dx;
        iy[y * width + x] = dy;
    }
}

__global__ void
temporalDifference(int *it, unsigned char *prevFrame, unsigned char *frame, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height)
    {
        it[y * width + x] = frame[y * width + x] - prevFrame[y * width + x];
    }
}

void
lucasKanade(const cv::Mat &prevFrame, const cv::Mat &frame, cv::Mat &result)
{
    int width = frame.cols;
    int height = frame.rows;
    int size = width * height * sizeof(unsigned char);

    unsigned char *deviceFrame = NULL;
    unsigned char *devicePrevFrame = NULL;
    int *ix = NULL;
    int *iy = NULL;
    int *it = NULL;

    cudaMalloc(&deviceFrame, width * height * sizeof(unsigned char));
    cudaMalloc(&devicePrevFrame, width * height * sizeof(unsigned char));
    cudaMalloc(&ix, width * height * sizeof(int));
    cudaMalloc(&iy, width * height * sizeof(int));
    cudaMalloc(&it, width * height * sizeof(int));

    cudaMemcpy(deviceFrame, frame.data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devicePrevFrame, prevFrame.data, size, cudaMemcpyHostToDevice);

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

    sobelFilter<<<gridDim, blockDim>>>(ix, iy, deviceFrame, width, height);
    temporalDifference<<<gridDim, blockDim>>>(it, devicePrevFrame, deviceFrame, width, height);
    cudaDeviceSynchronize();

    result.create(height, width, CV_32S);
    cudaMemcpy(result.data, ix, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(deviceFrame);
    cudaFree(deviceFrame);
    cudaFree(ix);
    cudaFree(iy);
    cudaFree(it);
}