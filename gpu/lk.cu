#include <stdio.h>

#include "lk.cuh"

#define BLOCK_SIZE 16

__global__ void
sobelFilter(int *ix, int *iy, unsigned char *frame, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int dx, dy;

    if (x > 0 && y > 0 && x < width - 1 && y < height - 1)
    {
        dx = (-1 * frame[(y - 1) * width + (x - 1)]) + (-2 * frame[y * width + (x - 1)]) +
             (-1 * frame[(y + 1) * width + (x - 1)]) + (frame[(y - 1) * width + (x + 1)]) +
             (2 * frame[y * width + (x + 1)]) + (frame[(y + 1) * width + (x + 1)]);
        dy = (frame[(y - 1) * width + (x - 1)]) + (2 * frame[(y - 1) * width + x]) +
             (frame[(y - 1) * width + (x + 1)]) + (-1 * frame[(y + 1) * width + (x - 1)]) +
             (-2 * frame[(y + 1) * width + x]) + (-1 * frame[(y + 1) * width + (x + 1)]);

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

__global__ void
harrisResponse(float *response, int *ix, int *iy, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < 2 || y < 2 || x >= width - 2 || y >= height - 2)
    {
        return;
    }

    float sumIxx = 0;
    float sumIyy = 0;
    float sumIxy = 0;

    // Creating the sum matrices for each pixel
    for (int dy = -2; dy <= 2; dy++)
    {
        for (int dx = -2; dx <= 2; dx++)
        {
            float gx = (float)ix[(y + dy) * width + (x + dx)];
            float gy = (float)iy[(y + dy) * width + (x + dx)];
            sumIxx += gx * gx;
            sumIyy += gy * gy;
            sumIxy += gx * gy;
        }
    }

    // Getting the corner response needed for this.
    float det = (sumIxx * sumIyy) - (sumIxy * sumIxy);
    float trace = (sumIxx + sumIyy);
    response[y * width + x] = det - (0.06 * trace * trace);
}

__global__ void
harrisThresholder(int *features, int *featureCount, float *response, int maxFeatures, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1)
    {
        return;
    }

    float r = response[y * width + x];

    // NMS && Thresholding, 3x3 window. just like in my 558 hw1
    if (r > response[(y - 1) * width + (x - 1)] && r > response[(y - 1) * width + (x)] &&
        r > response[(y - 1) * width + (x + 1)] && r > response[(y)*width + (x - 1)] &&
        r > response[(y)*width + (x + 1)] && r > response[(y + 1) * width + (x - 1)] &&
        r > response[(y + 1) * width + (x)] && r > response[(y + 1) * width + (x + 1)])
    {
        int featureSlot = atomicAdd(featureCount, 1);
        if (featureSlot < maxFeatures)
        {
            features[featureSlot * 2] = x;
            features[featureSlot * 2 + 1] = y;
        }
    }
}

void
lucasKanade(const cv::Mat &prevFrame, const cv::Mat &frame, cv::Mat &result, int maxFeatures)
{
    // TODO: Consider mallocing the whole video or frame chunks to optimize?
    // TODO: any way we can make min/max computation live on GPU?
    // TODO: sobel and temporal diff are independent, how can we make them run together?
    // TODO: reduce global memory accesses?

    int width = frame.cols;
    int height = frame.rows;
    int size = width * height * sizeof(unsigned char);

    unsigned char *deviceFrame = NULL;
    unsigned char *devicePrevFrame = NULL;
    int *ix = NULL;
    int *iy = NULL;
    int *it = NULL;

    int *deviceFrameFeatures = NULL;
    int *deviceFrameFeatureCount = NULL;
    float *deviceResponse = NULL;

    cudaMalloc(&deviceFrame, width * height * sizeof(unsigned char));
    cudaMalloc(&devicePrevFrame, width * height * sizeof(unsigned char));
    cudaMalloc(&ix, width * height * sizeof(int));
    cudaMalloc(&iy, width * height * sizeof(int));
    cudaMalloc(&it, width * height * sizeof(int));
    cudaMalloc(&deviceFrameFeatures, 2 * maxFeatures * sizeof(int));
    cudaMalloc(&deviceFrameFeatureCount, sizeof(int));
    cudaMalloc(&deviceResponse, width * height * sizeof(float));

    cudaMemcpy(deviceFrame, frame.data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devicePrevFrame, prevFrame.data, size, cudaMemcpyHostToDevice);
    cudaMemset(ix, 0, width * height * sizeof(int));
    cudaMemset(iy, 0, width * height * sizeof(int));
    cudaMemset(it, 0, width * height * sizeof(int));
    cudaMemset(deviceFrameFeatures, 0, 2 * maxFeatures * sizeof(int));
    cudaMemset(deviceFrameFeatureCount, 0, sizeof(int));
    cudaMemset(deviceResponse, 0, width * height * sizeof(float));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridDim((int)ceil((float)width / blockDim.x), (int)ceil((float)height / blockDim.y), 1);

    // TODO: Kernel for Shi-Tomasi Detector

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
    harrisResponse<<<gridDim, blockDim>>>(deviceResponse, ix, iy, width, height);

    cudaDeviceSynchronize();

    harrisThresholder<<<gridDim, blockDim>>>(deviceFrameFeatures, deviceFrameFeatureCount, deviceResponse, maxFeatures,
                                             width, height);

    cudaDeviceSynchronize();
    result.create(height, width, CV_32F);
    cudaMemcpy(result.data, deviceResponse, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(deviceFrame);
    cudaFree(devicePrevFrame);
    cudaFree(ix);
    cudaFree(iy);
    cudaFree(it);
    cudaFree(deviceFrameFeatures);
    cudaFree(deviceFrameFeatureCount);
    cudaFree(deviceResponse);
}