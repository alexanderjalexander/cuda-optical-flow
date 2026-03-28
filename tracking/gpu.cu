#include <stdio.h>

#include "../processing/drawing.hpp"

#include "lucasKanade.hpp"

#define BLOCK_SIZE 16
#define EPSILON 0.04

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
    response[y * width + x] = det - (EPSILON * trace * trace);
}

__global__ void
harrisThresholder(float3 *features, int *featureCount, float *response, int maxFeatures, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1)
    {
        return;
    }

    float r = response[y * width + x];

    // NMS, 3x3 window. just like in my 558 hw1
    if (r > response[(y - 1) * width + (x - 1)] && r > response[(y - 1) * width + (x)] &&
        r > response[(y - 1) * width + (x + 1)] && r > response[(y)*width + (x - 1)] &&
        r > response[(y)*width + (x + 1)] && r > response[(y + 1) * width + (x - 1)] &&
        r > response[(y + 1) * width + (x)] && r > response[(y + 1) * width + (x + 1)])
    {
        int featureSlot = atomicAdd(featureCount, 1);
        if (featureSlot < maxFeatures)
        {
            features[featureSlot].x = x;
            features[featureSlot].y = y;
            features[featureSlot].z = 1; // valid or invalid check
        }
    }
}

__global__ void
lucasKanadeSolver(float2 *flowVectors, int *ix, int *iy, int *it, float3 *features, int *featureCount, int width, int height)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *featureCount || features[i].z != 1)
        return;

    int centerX = features[i].x;
    int centerY = features[i].y;

    float sumIxx = 0;
    float sumIyy = 0;
    float sumIxy = 0;
    float sumIxt = 0;
    float sumIyt = 0;

    // 15x15 window, just like in the CPU version
    int window_half = 7;
    for (int y = -window_half; y <= window_half; y++)
    {
        for (int x = -window_half; x <= window_half; x++)
        {
            if ((centerX + x) < 0 || (centerX + x) >= width || (centerY + y) < 0 || (centerY + y) >= height)
                continue;
            int currentCoord = ((centerY + y) * width) + (centerX + x);
            sumIxx += ix[currentCoord] * ix[currentCoord];
            sumIyy += iy[currentCoord] * iy[currentCoord];
            sumIxy += ix[currentCoord] * iy[currentCoord];
            sumIxt += ix[currentCoord] * it[currentCoord];
            sumIyt += iy[currentCoord] * it[currentCoord];
        }
    }

    // Calculation at the bottom of the "Concept" section:
    // https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method#Concept
    // expanded out.
    float det = sumIxx * sumIyy - (sumIxy * sumIxy);
    if (fabs(det) < 1e-6)
    {
        features[i].z = 0;
        return; // it's already memset to 0, so we're fine.
    }

    flowVectors[i].x = ((sumIyy * -sumIxt) + (-sumIxy * -sumIyt)) / det;
    flowVectors[i].y = ((-sumIxy * -sumIxt) + (sumIxx * -sumIyt)) / det;
}

__global__ void
updateTrackingPoints(float3 *features, int *featureCount, float2 *flowVectors, int width, int height)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *featureCount)
        return;

    float updatedX = roundf(features[i].x + flowVectors[i].x);
    float updatedY = roundf(features[i].y + flowVectors[i].y);

    if (updatedX < 0 || updatedX >= width || updatedY < 0 || updatedY >= height)
        features[i].z = 0;

    features[i].x = updatedX;
    features[i].y = updatedY;
}

void
sparseLucasKanadeGPU(VideoInfo &video)
{
    if (video.frames.empty())
    {
        return;
    }

    int width = video.frames[0].cols;
    int height = video.frames[0].rows;
    int size = width * height * sizeof(unsigned char);

    // For coloring the output
    Mat mask = Mat::zeros(video.frames[0].size(), CV_8UC3);

    unsigned char *deviceFrame = NULL;
    unsigned char *devicePrevFrame = NULL;

    int *deviceIx = NULL;
    int *deviceIy = NULL;
    int *deviceIt = NULL;

    float3 *deviceFrameFeatures = NULL;
    float2 *deviceFlowVectors = NULL;
    int *deviceFrameFeatureCount = NULL;
    float *deviceResponse = NULL;

    cudaMalloc(&deviceFrame, width * height * sizeof(unsigned char));
    cudaMalloc(&devicePrevFrame, width * height * sizeof(unsigned char));

    cudaMalloc(&deviceIx, width * height * sizeof(int));
    cudaMalloc(&deviceIy, width * height * sizeof(int));
    cudaMalloc(&deviceIt, width * height * sizeof(int));

    cudaMalloc(&deviceFrameFeatures, MAX_FEATURES * sizeof(float3));
    cudaMalloc(&deviceFlowVectors, MAX_FEATURES * sizeof(float2));
    cudaMalloc(&deviceFrameFeatureCount, sizeof(int));
    cudaMalloc(&deviceResponse, width * height * sizeof(float));

    cudaMemcpy(deviceFrame, video.frames[0].data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devicePrevFrame, video.frames[1].data, size, cudaMemcpyHostToDevice);

    cudaMemset(deviceIx, 0, width * height * sizeof(int));
    cudaMemset(deviceIy, 0, width * height * sizeof(int));
    cudaMemset(deviceIt, 0, width * height * sizeof(int));

    cudaMemset(deviceFrameFeatures, 0, MAX_FEATURES * sizeof(float3));
    cudaMemset(deviceFlowVectors, 0, MAX_FEATURES * sizeof(float2));
    cudaMemset(deviceFrameFeatureCount, 0, sizeof(int));
    cudaMemset(deviceResponse, 0, width * height * sizeof(float));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridDim((int)ceil((float)width / blockDim.x), (int)ceil((float)height / blockDim.y), 1);

    sobelFilter<<<gridDim, blockDim>>>(deviceIx, deviceIy, deviceFrame, width, height);
    harrisResponse<<<gridDim, blockDim>>>(deviceResponse, deviceIx, deviceIy, width, height);
    harrisThresholder<<<gridDim, blockDim>>>(deviceFrameFeatures, deviceFrameFeatureCount, deviceResponse, MAX_FEATURES,
                                             width, height);
    cudaDeviceSynchronize();

    int featureCount = 0;
    cudaMemcpy(&featureCount, deviceFrameFeatureCount, sizeof(int), cudaMemcpyDeviceToHost);

    vector<Scalar> pt_colors = getRandomColors(featureCount);

    for (int i = 1; i < video.frames.size(); i++) {
        std::cout << i << endl;
        // unsigned char* tempPtr = deviceFrame;
        cudaMemcpy(deviceFrame, video.frames[i].data, size, cudaMemcpyHostToDevice);
        cudaMemcpy(devicePrevFrame, video.frames[i-1].data, size, cudaMemcpyHostToDevice);

        sobelFilter<<<gridDim, blockDim>>>(deviceIx, deviceIy, deviceFrame, width, height);
        temporalDifference<<<gridDim, blockDim>>>(deviceIt, devicePrevFrame, deviceFrame, width, height);
        cudaDeviceSynchronize();

        dim3 featureBlockDim(BLOCK_SIZE*BLOCK_SIZE, 1, 1);
        dim3 featureGridDim((int)ceil((float)MAX_FEATURES / featureBlockDim.x), 1, 1);

        lucasKanadeSolver<<<featureGridDim, featureBlockDim>>>(deviceFlowVectors, deviceIx, deviceIy, deviceIt, deviceFrameFeatures, deviceFrameFeatureCount, width, height);
        updateTrackingPoints<<<featureGridDim, featureBlockDim>>>(deviceFrameFeatures, deviceFrameFeatureCount, deviceFlowVectors, width, height);

        std::vector<float3> frameFeatures(MAX_FEATURES);
        cudaMemcpy(frameFeatures.data(), deviceFrameFeatures, featureCount * sizeof(float3), cudaMemcpyDeviceToHost);

        // TODO: this is still broken...
        Mat output;
        cvtColor(video.frames[i], output, COLOR_GRAY2BGR);
        drawOpticalFlowGPU(output, mask, reinterpret_cast<cv::Vec3f*>(frameFeatures.data()), featureCount, pt_colors, DRAW_CONTINUOUS_LINES);
        video.outputFrames.push_back(output);

        // TODO: If Features get low, then recalculate them.
    }

    cudaFree(deviceFrame);
    cudaFree(devicePrevFrame);

    cudaFree(deviceIx);
    cudaFree(deviceIy);
    cudaFree(deviceIt);

    cudaFree(deviceFrameFeatures);
    cudaFree(deviceFlowVectors);
    cudaFree(deviceFrameFeatureCount);
    cudaFree(deviceResponse);
}
