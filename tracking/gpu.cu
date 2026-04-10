#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include "../processing/drawing.hpp"

#include "lucasKanade.hpp"

#include <stdio.h>

#define BLOCK_SIZE 16

/**
 * Device-only code to collaboratively load a block of data into shared memory, including a halo for convolution
 * operations. Loads zero if the thread corresponds to a data or halo element outside the original frame. Calls
 * syncthreads at the end to ensure all threads have finished loading their data before any subsequent operations.
 *
 * @param T The type of the data to be loaded.
 * @param shared  Device shared memory to load the data into, expected to be a 2D array.
 * @param global Device global memory array containing input data.
 * @param halo_radius The radius of the halo to load around the internal block (e.g. 1 for a 3x3 mask, 2 for a 5x5 mask,
 * etc.).
 * @param width The input frame's width.
 * @param height The input frame's height.
 */
template <typename T>
__device__ void
loadSharedMemoryWithHalo(T **shared, T *global, int halo_radius, int width, int height)
{
    // identify the coordinates of the output pixel to work on
    int tx = threadIdx.x, ty = threadIdx.y;
    int tx_adj = tx + halo_radius;
    int ty_adj = ty + halo_radius;
    int x = tx + blockIdx.x * blockDim.x;
    int y = ty + blockIdx.y * blockDim.y;

    // internal elements - loaded directly by corresponding threads, offset in
    // shared memory by halo size
    shared[ty_adj][tx_adj] = (y < height && x < width) ? global[y * width + x] : 0;

    // top halo edge - loaded by bottom edge of threads
    int halo_top_row = (blockIdx.y - 1) * BLOCK_SIZE + ty;
    if (ty >= BLOCK_SIZE - halo_radius)
    {
        shared[ty - (BLOCK_SIZE - halo_radius)][tx_adj] = (halo_top_row < 0) ? 0 : global[halo_top_row * width + x];
    }

    // bottom halo edge - loaded by top edge of threads
    int halo_bottom_row = (blockIdx.y + 1) * BLOCK_SIZE + ty;
    if (ty < halo_radius)
    {
        shared[ty + BLOCK_SIZE + halo_radius][tx_adj] =
            (halo_bottom_row >= height) ? 0 : global[halo_bottom_row * width + x];
    }

    // left halo edge - loaded by right edge of threads
    int halo_left_col = (blockIdx.x - 1) * BLOCK_SIZE + tx;
    if (tx >= BLOCK_SIZE - halo_radius)
    {
        shared[ty_adj][tx - (BLOCK_SIZE - halo_radius)] = (halo_left_col < 0) ? 0 : global[y * width + halo_left_col];
    }

    // right halo edge - loaded by left edge of threads
    int halo_right_col = (blockIdx.x + 1) * BLOCK_SIZE + tx;
    if (tx < halo_radius)
    {
        shared[ty_adj][tx + BLOCK_SIZE + halo_radius] =
            (halo_right_col >= width) ? 0 : global[y * width + halo_right_col];
    }

    // halo corners - each loaded by the thread farthest from it (e.g. top right
    // of halo by bottom left thread) logic is a combination of the conditions
    // above
    if (ty >= BLOCK_SIZE - halo_radius)
    {
        // top edge of halo

        if (tx >= BLOCK_SIZE - halo_radius)
        {
            // top left halo corner
            shared[ty - (BLOCK_SIZE - halo_radius)][tx - (BLOCK_SIZE - halo_radius)] =
                (halo_top_row < 0 || halo_left_col < 0) ? 0 : global[halo_top_row * width + halo_left_col];
        }
        else if (tx < halo_radius)
        {
            // top right halo corner
            shared[ty - (BLOCK_SIZE - halo_radius)][tx + BLOCK_SIZE + halo_radius] =
                (halo_top_row < 0 || halo_right_col >= width) ? 0 : global[halo_top_row * width + halo_right_col];
        }
    }
    else if (ty < halo_radius)
    {
        // bottom edge of halo

        if (tx >= BLOCK_SIZE - halo_radius)
        {
            // bottom left halo corner
            shared[ty + BLOCK_SIZE + halo_radius][tx - (BLOCK_SIZE - halo_radius)] =
                (halo_bottom_row >= height || halo_left_col < 0) ? 0 : global[halo_bottom_row * width + halo_left_col];
        }
        else if (tx < halo_radius)
        {
            // bottom right halo corner
            shared[ty + BLOCK_SIZE + halo_radius][tx + BLOCK_SIZE + halo_radius] =
                (halo_bottom_row >= height || halo_right_col >= width)
                    ? 0
                    : global[halo_bottom_row * width + halo_right_col];
        }
    }

    __syncthreads();
}

/**
 * Device-only code to get the bilinear interpolation of a point in an image.
 *
 * @param img The image to interpolate against.
 * @param x The desired x coordinate.
 * @param y The desired y coordinate.
 * @param width The image `img`'s width.
 * @param height The image `img`'s height.
 *
 * @returns A floating point value representing the interpolated intensity.
 */
__device__ float
bilinearInterpolate(unsigned char *img, float x, float y, int width, int height)
{
    int x1 = (int)floor(x);
    int x2 = (int)ceil(x);
    int y1 = (int)floor(y);
    int y2 = (int)ceil(y);

    x1 = max(0, min(x1, width - 1));
    x2 = max(0, min(x2, width - 1));
    y1 = max(0, min(y1, height - 1));
    y2 = max(0, min(y2, height - 1));

    float q11 = (float)img[x1 + (y1 * width)];
    float q12 = (float)img[x1 + (y2 * width)];
    float q21 = (float)img[x2 + (y1 * width)];
    float q22 = (float)img[x2 + (y2 * width)];

    float dx = x - float(x1);
    float dy = y - float(y1);

    return (1.0 - dx) * (1.0 - dy) * q11 + (1.0 - dx) * (dy)*q12 + (dx) * (1.0 - dy) * q21 + (dx) * (dy)*q22;
}

/**
 * Kernel code to obtain the horizontal and vertical Sobel derivatives of an image.
 *
 * Results are normalized by dividing the final result by 8.
 *
 * @param ix Device memory to store Ix, the horizontal derivative.
 * @param iy Device memory to store Iy, the vertical derivative.
 * @param frame The 8-bit, grayscale image to calculate the sobel derivative on.
 * @param width The image's width.
 * @param height The image's height.
 */
__global__ void
sobelFilter(float *ix, float *iy, unsigned char *frame, int width, int height)
{
    // define a block of shared memory large enough to hold the internal values and the halo
    __shared__ unsigned char frameShared[BLOCK_SIZE + SOBEL_MASK_SIZE - 1][BLOCK_SIZE + SOBEL_MASK_SIZE - 1];
    int halo_radius = SOBEL_MASK_SIZE / 2;

    // identify the coordinates of the output pixel to work on
    int tx = threadIdx.x, ty = threadIdx.y;
    int tx_adj = tx + halo_radius;
    int ty_adj = ty + halo_radius;
    int x = tx + blockIdx.x * blockDim.x;
    int y = ty + blockIdx.y * blockDim.y;
    int global_index = y * width + x;

    // load the input (including halo) into shared memory
    loadSharedMemoryWithHalo<unsigned char>(frameShared, frame, halo_radius, width, height);

    // =====================================
    // perform the actual calculation
    // =====================================

    // don't write to output for nonexistent pixels
    if (y >= height || x >= width)
    {
        return;
    }

    // don't compute for the outermost layer of pixels
    if (y == 0 || y == height - 1 || x == 0 || x == width - 1)
    {
        ix[global_index] = 0;
        iy[global_index] = 0;
        return;
    }

    // do the actual computation for relevant threads
    float dx = (-1 * frameShared[ty_adj - 1][tx_adj - 1]) + (-2 * frameShared[ty_adj][tx_adj - 1]) +
               (-1 * frameShared[ty_adj + 1][tx_adj - 1]) + (frameShared[ty_adj - 1][tx_adj + 1]) +
               (2 * frameShared[ty_adj][tx_adj + 1]) + (frameShared[ty_adj + 1][tx_adj + 1]);

    float dy = (-1 * frameShared[ty_adj - 1][tx_adj - 1]) + (-2 * frameShared[ty_adj - 1][tx_adj]) +
               (-1 * frameShared[ty_adj - 1][tx_adj + 1]) + (1 * frameShared[ty_adj + 1][tx_adj - 1]) +
               (2 * frameShared[ty_adj + 1][tx_adj]) + (1 * frameShared[ty_adj + 1][tx_adj + 1]);

    ix[global_index] = dx / 8;
    iy[global_index] = dy / 8;
}

/**
 * Kernel code to obtain the temporal difference between two images.
 *
 * @param it Device memory to store It, the temporal derivative.
 * @param prevFrame The 8-bit, grayscale image to calculate the sobel derivative on.
 * @param frame The 8-bit, grayscale image to calculate the sobel derivative on.
 * @param width The image's width.
 * @param height The image's height.
 */
__global__ void
temporalDifference(float *it, unsigned char *prevFrame, unsigned char *frame, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height)
    {
        int global_index = y * width + x;
        it[global_index] = frame[global_index] - prevFrame[global_index];
    }
}

/**
 * Kernel code to obtain an image's Harris Response given the sobel derivatives.
 *
 * @param response Device memory to store the initial Harris response.
 * @param ix Device memory containing Ix, the horizontal derivative.
 * @param iy Device memory containing Iy, the vertical derivative.
 * @param width The image's width.
 * @param height The image's height.
 */
__global__ void
harrisResponse(float *response, float *ix, float *iy, int width, int height)
{
    __shared__ float ixShared[BLOCK_SIZE + HARRIS_MASK_SIZE - 1][BLOCK_SIZE + HARRIS_MASK_SIZE - 1];
    __shared__ float iyShared[BLOCK_SIZE + HARRIS_MASK_SIZE - 1][BLOCK_SIZE + HARRIS_MASK_SIZE - 1];
    int halo_radius = HARRIS_MASK_SIZE / 2;

    int tx = threadIdx.x, ty = threadIdx.y;
    int tx_adj = tx + halo_radius;
    int ty_adj = ty + halo_radius;
    int x = tx + blockIdx.x * blockDim.x;
    int y = ty + blockIdx.y * blockDim.y;

    // collaboratively load the horizontal and vertical derivatives into shared memory, including a halo
    loadSharedMemoryWithHalo<float>(ixShared, ix, halo_radius, width, height);
    loadSharedMemoryWithHalo<float>(iyShared, iy, halo_radius, width, height);


    // don't compute for the outermost layers of pixels
    if (x < halo_radius || y < halo_radius || x >= width - halo_radius || y >= height - halo_radius)
    {
        return;
    }

    float sumIxx = 0;
    float sumIyy = 0;
    float sumIxy = 0;

    // Creating the sum matrices for each pixel
    // TODO: Gaussian Weights??
    // goodFeaturesToTrack does so, maybe this will reduce false positives :shrug:
    for (int dy = -halo_radius; dy <= halo_radius; dy++)
    {
        for (int dx = -halo_radius; dx <= halo_radius; dx++)
        {
            float gx = ixShared[ty_adj + dy][tx_adj + dx];
            float gy = iyShared[ty_adj + dy][tx_adj + dx];
            sumIxx += gx * gx;
            sumIyy += gy * gy;
            sumIxy += gx * gy;
        }
    }

    // Getting the corner response needed for this.
    float det = (sumIxx * sumIyy) - (sumIxy * sumIxy);
    float trace = (sumIxx + sumIyy);
    response[y * width + x] = det - (HARRIS_EPSILON * trace * trace);
}

/**
 * Kernel code to threshold an image's Harris Response, in order to obtain
 * the final feature set.
 *
 * @param features The device memory storing all the features we wish to track.
 * @param featureCount How many features we're attempting to store.
 * @param response Device memory storing the initial Harris response.
 * @param threshold A float representing (R_max * QUALITY_LEVEL), where R_Max
 * is the maximum Response in the entire image.
 * @param maxFeatures The maximum features we're limiting this program to be able to run.
 * @param width The image's width.
 * @param height The image's height.
 */
__global__ void
harrisThresholder(float3 *features, int *featureCount, float *response, float threshold, int maxFeatures, int width,
                  int height)
{
    // define a block of shared memory large enough to hold the internal values and the halo
    __shared__ float responseShared[BLOCK_SIZE + (HARRIS_DISTANCE * 2)][BLOCK_SIZE + (HARRIS_DISTANCE * 2)];
    int halo_radius = HARRIS_DISTANCE;

    int tx = threadIdx.x, ty = threadIdx.y;
    int tx_adj = tx + halo_radius;
    int ty_adj = ty + halo_radius;
    int x = tx + blockIdx.x * blockDim.x;
    int y = ty + blockIdx.y * blockDim.y;

    loadSharedMemoryWithHalo<float>(responseShared, response, halo_radius, width, height);

    // don't calculate for the outermost layers of pixels
    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1)
    {
        return;
    }


    // Initial threshold check
    float r = responseShared[ty_adj][tx_adj];
    if (r < threshold)
    {
        return;
    }

    // Non-Max Suppression
    for (int yShift = -HARRIS_DISTANCE; yShift <= HARRIS_DISTANCE; yShift++)
    {
        for (int xShift = -HARRIS_DISTANCE; xShift <= HARRIS_DISTANCE; xShift++)
        {
            if (yShift == 0 && xShift == 0)
            {
                continue;
            }
            if (y + yShift < 0 || x + xShift < 0 || y + yShift >= height || x + xShift >= width)
            {
                continue;
            }
            if (r <= responseShared[ty_adj + yShift][tx_adj + xShift])
            {
                return;
            }
        }
    }

    int featureSlot = atomicAdd(featureCount, 1);
    if (featureSlot < maxFeatures)
    {
        features[featureSlot] = make_float3(x, y, 1);
        // features[featureSlot].x = x;
        // features[featureSlot].y = y;
        // features[featureSlot].z = 1;
    }
}

/**
 * Kernel code to perform iterative Lucas Kanade on the features given between two
 * images.
 *
 * @param flowVectors The flow directions we're attempting to store and run back.
 * @param ix Device memory storing Ix, the horizontal derivative.
 * @param iy Device memory storing Iy, the vertical derivative.
 * @param frame Global memory storing the original frame.
 * @param prevFrame Global memory storing the previous frame.
 * @param features The device memory storing all the features we wish to track.
 * @param featureCount How many features we have in actuality.
 * @param width The image's width.
 * @param height The image's height.
 */
__global__ void
iterLucasKanadeSolver(float2 *flowVectors, float *ix, float *iy, unsigned char *frame, unsigned char *prevFrame,
                      float3 *features, int *featureCount, int width, int height)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *featureCount || features[i].z != 1)
    {
        return;
    }

    flowVectors[i] = {0.0f, 0.0f};

    int centerX = (int)features[i].x;
    int centerY = (int)features[i].y;

    float u = 0.0f;
    float v = 0.0f;

    // 15x15 window, just like in the CPU version
    int windowHalf = 7;

    for (int iteration = 0; iteration < LK_ITERATIONS; iteration++)
    {
        float sumIxx = 0.0f;
        float sumIyy = 0.0f;
        float sumIxy = 0.0f;
        float sumIxt = 0.0f;
        float sumIyt = 0.0f;

        for (int y = -windowHalf; y <= windowHalf; y++)
        {
            for (int x = -windowHalf; x <= windowHalf; x++)
            {
                int iterCenterX = centerX + x;
                int iterCenterY = centerY + y;

                if (iterCenterX < 0 || iterCenterX >= width || iterCenterY < 0 || iterCenterY >= height)
                {
                    continue;
                }

                float warpedX = (float)iterCenterX + u;
                float warpedY = (float)iterCenterY + v;

                if (warpedX < 0.0f || warpedX >= (float)width || warpedY < 0.0f || warpedY >= (float)height)
                {
                    continue;
                }

                int currentCoord = iterCenterY * width + iterCenterX;
                float gx = ix[currentCoord];
                float gy = iy[currentCoord];
                float it = bilinearInterpolate(frame, warpedX, warpedY, width, height) - (float)prevFrame[currentCoord];

                sumIxx += gx * gx;
                sumIyy += gy * gy;
                sumIxy += gx * gy;
                sumIxt += gx * it;
                sumIyt += gy * it;
            }
        }

        float det = sumIxx * sumIyy - (sumIxy * sumIxy);
        if (fabs(det) < 1e-6f)
        {
            features[i].z = 0;
            return;
        }

        float du = ((sumIyy * -sumIxt) + (-sumIxy * -sumIyt)) / det;
        float dv = ((-sumIxy * -sumIxt) + (sumIxx * -sumIyt)) / det;

        u += du;
        v += dv;

        if (du * du + dv * dv < LK_EPSILON)
        {
            break;
        }
    }

    flowVectors[i].x = u;
    flowVectors[i].y = v;
}

/**
 * Kernel code to perform primitive Lucas Kanade on the features given between two images.
 *
 * @param flowVectors The flow directions we're attempting to store and run back.
 * @param ix Device memory storing Ix, the horizontal derivative.
 * @param iy Device memory storing Iy, the vertical derivative.
 * @param it Device memory storing It, the temporal derivative.
 * @param features The device memory storing all the features we wish to track.
 * @param featureCount How many features we have in actuality.
 * @param width The image's width.
 * @param height The image's height.
 */
__global__ void
lucasKanadeSolver(float2 *flowVectors, float *ix, float *iy, float *it, float3 *features, int *featureCount, int width,
                  int height)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *featureCount || features[i].z != 1)
    {
        return;
    }

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
            {
                continue;
            }
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
        return;
    }

    flowVectors[i].x = ((sumIyy * -sumIxt) + (-sumIxy * -sumIyt)) / det;
    flowVectors[i].y = ((-sumIxy * -sumIxt) + (sumIxx * -sumIyt)) / det;
}

/**
 * Kernel code to update the tracking points after an full Lucas Kanade run between
 * two images.
 *
 * @param features The device memory storing all the features we wish to track.
 * @param featureCount How many features we have in actuality.
 * @param flowVectors The flow directions we're attempting to store and run back.
 * @param width The image's width.
 * @param height The image's height.
 */
__global__ void
updateTrackingPoints(float3 *features, int *featureCount, float2 *flowVectors, int width, int height)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *featureCount || features[i].z != 1)
    {
        return;
    }

    float updatedX = (features[i].x + flowVectors[i].x);
    float updatedY = (features[i].y + flowVectors[i].y);

    if (updatedX < 0 || updatedX >= width || updatedY < 0 || updatedY >= height)
    {
        features[i].z = 0;
        return;
    }

    features[i].x = updatedX;
    features[i].y = updatedY;
}

/**
 * Host code that initiates a full flow of Sparse Lucas Kanade on the GPU.
 *
 * At first, this will calculate possibly good features, very similar to OpenCV's
 * goodFeaturesToTrack (more specifically, using a Harris Response with the same
 * parameters as the CPU variant in this codebase). Then, it will perform Lucas
 * Kanade flow calculations per each frame pair, draw the frames results, and
 * continue onwards. It will do this until there are no more frames left to
 * calculate for Lucas Kanade optical flow.
 *
 * @param video the VideoInfo struct storing the initial frame's videos.
 */
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
    cv::Mat mask = cv::Mat::zeros(video.frames[0].size(), CV_8UC3);

    // === Pointer Declaration ===

    unsigned char *deviceFrame = NULL;
    unsigned char *devicePrevFrame = NULL;

    float *deviceIx = NULL;
    float *deviceIy = NULL;
    float *deviceIt = NULL;

    float3 *deviceFrameFeatures = NULL;
    float2 *deviceFlowVectors = NULL;
    int *deviceFrameFeatureCount = NULL;
    float *deviceResponse = NULL;

    // === Pointer Memory Allocation ===

    // TODO: Error check???
    cudaMalloc(&deviceFrame, width * height * sizeof(unsigned char));
    cudaMalloc(&devicePrevFrame, width * height * sizeof(unsigned char));

    cudaMalloc(&deviceIx, width * height * sizeof(float));
    cudaMalloc(&deviceIy, width * height * sizeof(float));
    cudaMalloc(&deviceIt, width * height * sizeof(float));

    cudaMalloc(&deviceFrameFeatures, MAX_FEATURES * sizeof(float3));
    cudaMalloc(&deviceFlowVectors, MAX_FEATURES * sizeof(float2));
    cudaMalloc(&deviceFrameFeatureCount, sizeof(int));
    cudaMalloc(&deviceResponse, width * height * sizeof(float));

    // === Pointer Memory Copying & Zeroing ===

    cudaMemcpy(deviceFrame, video.frames[0].data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devicePrevFrame, video.frames[1].data, size, cudaMemcpyHostToDevice);

    cudaMemset(deviceIx, 0, width * height * sizeof(float));
    cudaMemset(deviceIy, 0, width * height * sizeof(float));
    cudaMemset(deviceIt, 0, width * height * sizeof(float));

    cudaMemset(deviceFrameFeatures, 0, MAX_FEATURES * sizeof(float3));
    cudaMemset(deviceFlowVectors, 0, MAX_FEATURES * sizeof(float2));
    cudaMemset(deviceFrameFeatureCount, 0, sizeof(int));
    cudaMemset(deviceResponse, 0, width * height * sizeof(float));

    // === Kernel Blocks & Grids Initialization ===

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridDim((int)ceil((float)width / blockDim.x), (int)ceil((float)height / blockDim.y), 1);

    // === First Frame Procedure ===
    // 1. Grab Sobel & Harris Response
    // 2. Use Thrust Extrema to calculate threshold
    // 3. Threshold all features
    // 3. Perform Lucas Kanade Solve on every frame past that

    // == Initial Sobel Filter & Harris Response ==

    sobelFilter<<<gridDim, blockDim>>>(deviceIx, deviceIy, deviceFrame, width, height);
    harrisResponse<<<gridDim, blockDim>>>(deviceResponse, deviceIx, deviceIy, width, height);
    cudaDeviceSynchronize();

    // == Threshold Calculations w/ Thrust ==

    thrust::device_ptr<float> responsePtr(deviceResponse);
    float responseMax = *thrust::max_element(responsePtr, responsePtr + (width * height));
    float responseThreshold = responseMax * QUALITY_LEVEL;

    harrisThresholder<<<gridDim, blockDim>>>(deviceFrameFeatures, deviceFrameFeatureCount, deviceResponse,
                                             responseThreshold, MAX_FEATURES, width, height);
    cudaDeviceSynchronize();

    // == Getting Feature Counts ==

    int featureCount = 0;
    cudaMemcpy(&featureCount, deviceFrameFeatureCount, sizeof(int), cudaMemcpyDeviceToHost);
    /**
     * Below line is needed to prevent CUDA from accessing bad memory, otherwise we'll have no results.
     * Segfaults seem to be... silent? When we exceeded this, every single status was turning into 1.
     */
    featureCount = min(featureCount, MAX_FEATURES);
    float3 *prevFrameFeatures = (float3 *)calloc(featureCount, sizeof(float3));
    float3 *frameFeatures = (float3 *)calloc(featureCount, sizeof(float3));
    cudaMemcpy(prevFrameFeatures, deviceFrameFeatures, featureCount * sizeof(float3), cudaMemcpyDeviceToHost);
    std::vector<cv::Scalar> pt_colors = getRandomColors(featureCount);

    dim3 featureBlockDim(BLOCK_SIZE * BLOCK_SIZE, 1, 1);
    dim3 featureGridDim((int)ceil((float)featureCount / featureBlockDim.x), 1, 1);

    // == Repeated Frame LK Procedure ==

    for (int i = 1; i < video.frames.size(); i++)
    {
        // Switch Frames
        cudaMemcpy(devicePrevFrame, deviceFrame, size, cudaMemcpyDeviceToDevice);
        cudaMemcpy(deviceFrame, video.frames[i].data, size, cudaMemcpyHostToDevice);

        // Do Sobel & Temporal Difference
        sobelFilter<<<gridDim, blockDim>>>(deviceIx, deviceIy, devicePrevFrame, width, height);
        cudaDeviceSynchronize();

        // Obtain Lucas Kanade Solve on 1 dimensional grid/block array
        iterLucasKanadeSolver<<<featureGridDim, featureBlockDim>>>(deviceFlowVectors, deviceIx, deviceIy, deviceFrame,
                                                                   devicePrevFrame, deviceFrameFeatures,
                                                                   deviceFrameFeatureCount, width, height);
        updateTrackingPoints<<<featureGridDim, featureBlockDim>>>(deviceFrameFeatures, deviceFrameFeatureCount,
                                                                  deviceFlowVectors, width, height);
        cudaDeviceSynchronize();

        cudaMemcpy(frameFeatures, deviceFrameFeatures, featureCount * sizeof(float3), cudaMemcpyDeviceToHost);

        cv::Mat output;
        cvtColor(video.frames[i], output, cv::COLOR_GRAY2BGR);
        drawSparseOpticalFlowGPU(output, mask, reinterpret_cast<cv::Vec3f *>(prevFrameFeatures),
                           reinterpret_cast<cv::Vec3f *>(frameFeatures), featureCount, pt_colors,
                           DRAW_CONTINUOUS_LINES);

        std::memcpy(prevFrameFeatures, frameFeatures, featureCount * sizeof(float3));
        video.outputFrames.push_back(output);

        // TODO: Gaussian Weighted Average
        // 2:00 - https://www.youtube.com/watch?v=79Ty2Kkivvc
        // TODO: Coarse-To-Fine
        // 2:55 - https://www.youtube.com/watch?v=79Ty2Kkivvc
        // TODO: If Features get low, then recalculate them.

        // TODO: consider dense might be faster and possibly more parallelizable???
        // - Every feature in DLK is just... every pixel... and it's the same.
        // - Then we just change how we display the output to be like a heatmap or smth

        /**
         * Possible Optimizations To Consider
         * - Shared Memory
         * - Texture Memory
         * - 1 Kernel, calculating certain steps on the fly with device specific functions
         */
    }

    // === Memory Freeing Procedure ===

    free(prevFrameFeatures);
    free(frameFeatures);

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
