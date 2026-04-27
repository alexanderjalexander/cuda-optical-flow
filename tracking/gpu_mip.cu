#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include "../processing/drawing.hpp"

#include "gpu_utilities.cuh"
#include "lucasKanade.hpp"

#include <stdio.h>

/**
 * Kernel code to obtain an image's Harris Response given the sobel derivatives.
 *
 * @param response Device memory to store the initial Harris response.
 * @param frame Device memory containing the image to obtain the Harris response on.
 * @param width The image's width.
 * @param height The image's height.
 */
__global__ void
harrisResponseMip(float *response, cudaTextureObject_t frameTex, int width, int height)
{
    // constant definitions for my own sanity
    const int TOTAL_HALO = HARRIS_MASK_RAD + SOBEL_MASK_RAD;

    // standard variable definitions, like always
    int tx = threadIdx.x, ty = threadIdx.y;
    int x = tx + blockIdx.x * blockDim.x;
    int y = ty + blockIdx.y * blockDim.y;

    // define multiple shared memory blocks large enough to hold the internal values and the halos
    __shared__ unsigned char frameShared[BLOCK_SIZE + (2 * TOTAL_HALO)][BLOCK_SIZE + (2 * TOTAL_HALO)];
    __shared__ float ixShared[BLOCK_SIZE + (2 * HARRIS_MASK_RAD)][BLOCK_SIZE + (2 * HARRIS_MASK_RAD)];
    __shared__ float iyShared[BLOCK_SIZE + (2 * HARRIS_MASK_RAD)][BLOCK_SIZE + (2 * HARRIS_MASK_RAD)];

    // collaboratively load the horizontal and vertical derivatives into shared memory, including halos
    for (int i = ty; i < BLOCK_SIZE + (2 * TOTAL_HALO); i += BLOCK_SIZE)
    {
        for (int j = tx; j < BLOCK_SIZE + (2 * TOTAL_HALO); j += BLOCK_SIZE)
        {
            float gx = ((blockIdx.x * blockDim.x - TOTAL_HALO + j) + 0.5)/width;
            float gy = ((blockIdx.y * blockDim.y - TOTAL_HALO + i) + 0.5)/height;
            frameShared[i][j] = (unsigned char)(tex2DLod<float>(frameTex, gx, gy, 0.0f) * 255.0f);
        }
    }
    __syncthreads();

    // on the fly derivative calculations
    int derivWindowSize = BLOCK_SIZE + (2 * HARRIS_MASK_RAD);
    for (int i = ty; i < derivWindowSize; i += BLOCK_SIZE)
    {
        for (int j = tx; j < derivWindowSize; j += BLOCK_SIZE)
        {
            // t_idx = 0,0. We'll need the sobel derivative w.r.t. 1,1.
            // It'll then go to 0,16. then 16,0. then 16,16
            int fsY = i + SOBEL_MASK_RAD;
            int fsX = j + SOBEL_MASK_RAD;

            float dx = (-1.0f * frameShared[fsY - 1][fsX - 1]) + (-2.0f * frameShared[fsY][fsX - 1]) +
                       (-1.0f * frameShared[fsY + 1][fsX - 1]) + (1.0f * frameShared[fsY - 1][fsX + 1]) +
                       (2.0f * frameShared[fsY][fsX + 1]) + (1.0f * frameShared[fsY + 1][fsX + 1]);

            float dy = (-1.0f * frameShared[fsY - 1][fsX - 1]) + (-2.0f * frameShared[fsY - 1][fsX]) +
                       (-1.0f * frameShared[fsY - 1][fsX + 1]) + (1.0f * frameShared[fsY + 1][fsX - 1]) +
                       (2.0f * frameShared[fsY + 1][fsX]) + (1.0f * frameShared[fsY + 1][fsX + 1]);

            ixShared[i][j] = dx / 8.0f;
            iyShared[i][j] = dy / 8.0f;
        }
    }
    __syncthreads();

    // don't compute for the outermost layers of pixels
    if (x <= TOTAL_HALO || y <= TOTAL_HALO || x >= width - TOTAL_HALO || y >= height - TOTAL_HALO)
    {
        return;
    }

    float sumIxx = 0, sumIyy = 0, sumIxy = 0;

    // Creating the sum matrices for each pixel
    int tx_adj = tx + HARRIS_MASK_RAD;
    int ty_adj = ty + HARRIS_MASK_RAD;
    for (int dy = -HARRIS_MASK_RAD; dy <= HARRIS_MASK_RAD; dy++)
    {
        for (int dx = -HARRIS_MASK_RAD; dx <= HARRIS_MASK_RAD; dx++)
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
harrisThresholderMip(float3 *features, int *featureCount, float *response, float threshold, int maxFeatures, int width,
                     int height)
{
    // define a block of shared memory large enough to hold the internal values and the halo
    int halo_radius = HARRIS_DISTANCE;
    __shared__ float responseShared[BLOCK_SIZE + (HARRIS_DISTANCE * 2)][BLOCK_SIZE + (HARRIS_DISTANCE * 2)];

    // load the input (including halo) into shared memory
    load2dSharedMemoryWithHalo<float>(responseShared[0], response, halo_radius, width, height);
    __syncthreads();

    int tx = threadIdx.x, ty = threadIdx.y;
    int tx_adj = tx + halo_radius;
    int ty_adj = ty + halo_radius;
    int x = tx + blockIdx.x * blockDim.x;
    int y = ty + blockIdx.y * blockDim.y;

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

    // Check if there's space for another feature. Overflow is incredibly unlikely because the integer limit is much
    // larger than the total number of pixels in any frame
    if (*featureCount < maxFeatures)
    {
        int featureSlot = atomicAdd(featureCount, 1);
        if (featureSlot < maxFeatures)
        {
            features[featureSlot] = make_float3(x, y, 1);
        }
    }
}

/**
 * Kernel code to perform iterative Lucas Kanade on the features given between two
 * images.
 *
 * Assumes that the block size used to initialize the kernel is the actual window
 * size we go off of. Ideally, the block dimensions should be LK_WINDOW_HALF *
 * LK_WINDOW_HALF.
 *
 * @param frame Global memory storing the original frame.
 * @param prevFrame Global memory storing the previous frame.
 * @param features The device memory storing all the features we wish to track.
 * @param featureCount How many features we have in actuality.
 * @param width The image's width.
 * @param height The image's height.
 */
__global__ void
iterLucasKanadeSolverMip(cudaTextureObject_t frameTex, cudaTextureObject_t prevFrameTex, float3 *features,
                         int *featureCount, int width, int height)
{
    // usual per-thread registers/variables
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int flatIdx = ty * blockDim.x + tx; // flat index w.r.t. shared memory

    int featureNum = blockIdx.x; // grid is 1d array of blocks, each block is 2d array of threads
    if (featureNum >= *featureCount)
    {
        return;
    }

    int reductionStride = 1 << (31 - __clz(LK_WINDOW_WIDTH * LK_WINDOW_WIDTH - 1));

    // identify the feature to work on, early terminate if necessary
    float3 feature = features[featureNum];
    if (feature.z != 1)
    {
        return;
    }
    int fx = (int)feature.x;
    int fy = (int)feature.y;

    // define multiple shared memory blocks large enough to hold the internal values and the halos
    __shared__ float ixShared[LK_WINDOW_WIDTH][LK_WINDOW_WIDTH];
    __shared__ float iyShared[LK_WINDOW_WIDTH][LK_WINDOW_WIDTH];
    __shared__ unsigned char prevFrameShared[LK_WINDOW_WIDTH + (2 * SOBEL_MASK_RAD)]
                                            [LK_WINDOW_WIDTH + (2 * SOBEL_MASK_RAD)];

    __shared__ float ixxShared[LK_WINDOW_NUM];
    __shared__ float iyyShared[LK_WINDOW_NUM];
    __shared__ float ixyShared[LK_WINDOW_NUM];
    __shared__ float ixtShared[LK_WINDOW_NUM];
    __shared__ float iytShared[LK_WINDOW_NUM];

    __shared__ float u, v, du, dv, det;
    __shared__ bool invalidDet;

    // Coordinates on GLOBAL image, relative to feature and thread index.
    // Assuming block size of 15x15, 7,7 should be the very middle.
    int baseX = fx - LK_WINDOW_WIDTH_HALF - SOBEL_MASK_RAD;
    int baseY = fy - LK_WINDOW_WIDTH_HALF - SOBEL_MASK_RAD;

    // TODO: Implement the coarse-to-fine loop starting from here.

    // get the halos into shared memory, including halos
    for (int i = ty; i < LK_WINDOW_WIDTH + (2 * SOBEL_MASK_RAD); i += LK_WINDOW_WIDTH)
    {
        for (int j = tx; j < LK_WINDOW_WIDTH + (2 * SOBEL_MASK_RAD); j += LK_WINDOW_WIDTH)
        {
            prevFrameShared[i][j] =
                (unsigned char)(tex2DLod<float>(prevFrameTex, (baseX + j + 0.5f)/width, (baseY + i + 0.5f)/height, 0.0f) * 255.0f);
        }
    }
    __syncthreads();

    // Load with bounds checking - clamp to zero if out of bounds
    int fsY = ty + SOBEL_MASK_RAD;
    int fsX = tx + SOBEL_MASK_RAD;

    float dx = (-1.0f * prevFrameShared[fsY - 1][fsX - 1]) + (-2.0f * prevFrameShared[fsY][fsX - 1]) +
               (-1.0f * prevFrameShared[fsY + 1][fsX - 1]) + (1.0f * prevFrameShared[fsY - 1][fsX + 1]) +
               (2.0f * prevFrameShared[fsY][fsX + 1]) + (1.0f * prevFrameShared[fsY + 1][fsX + 1]);

    float dy = (-1.0f * prevFrameShared[fsY - 1][fsX - 1]) + (-2.0f * prevFrameShared[fsY - 1][fsX]) +
               (-1.0f * prevFrameShared[fsY - 1][fsX + 1]) + (1.0f * prevFrameShared[fsY + 1][fsX - 1]) +
               (2.0f * prevFrameShared[fsY + 1][fsX]) + (1.0f * prevFrameShared[fsY + 1][fsX + 1]);

    ixShared[ty][tx] = dx / 8.0f;
    iyShared[ty][tx] = dy / 8.0f;

    __syncthreads();

    // Let thread 0 do this
    if (flatIdx == 0)
    {
        u = v = du = dv = 0.0f;
    }
    __syncthreads();

    // Per-thread calculation of ixx, iyy, and ixy
    ixxShared[flatIdx] = ixShared[ty][tx] * ixShared[ty][tx];
    iyyShared[flatIdx] = iyShared[ty][tx] * iyShared[ty][tx];
    ixyShared[flatIdx] = ixShared[ty][tx] * iyShared[ty][tx];
    __syncthreads();

    // Giant Reduction Sum for the three big sums, Ixx, Iyy, and Ixy
    for (unsigned int stride = reductionStride; stride >= 1; stride >>= 1)
    {
        if (flatIdx < stride && (flatIdx + stride) < (LK_WINDOW_NUM))
        {
            ixxShared[flatIdx] += ixxShared[flatIdx + stride];
            iyyShared[flatIdx] += iyyShared[flatIdx + stride];
            ixyShared[flatIdx] += ixyShared[flatIdx + stride];
        }
        __syncthreads();
    }

    // Determinant check before calculation proceeds
    if (flatIdx == 0)
    {
        det = ixxShared[0] * iyyShared[0] - (ixyShared[0] * ixyShared[0]);
        invalidDet = (fabs(det) < 1e-6f);
    }

    __syncthreads();

    if (invalidDet)
    {
        if (flatIdx == 0)
        {
            features[featureNum].z = 0;
        }
        return;
    }

    // Giant Iteration Loop
    for (int iteration = 0; iteration < LK_ITERATIONS; iteration++)
    {
        float warpedX = (fx + u + (tx - LK_WINDOW_WIDTH_HALF) + 0.5f)/width;
        float warpedY = (fy + v + (ty - LK_WINDOW_WIDTH_HALF) + 0.5f)/height;

        float it = (tex2DLod<float>(frameTex, warpedX, warpedY, 0.0f) * 255.0f) -
                   (float)prevFrameShared[fsY][fsX];

        ixtShared[flatIdx] = ixShared[ty][tx] * it;
        iytShared[flatIdx] = iyShared[ty][tx] * it;

        __syncthreads();

        // Giant Reduction Sum for the remaining big sums, Ixt and Iyt
        for (unsigned int stride = reductionStride; stride >= 1; stride >>= 1)
        {
            if (flatIdx < stride && (flatIdx + stride) < LK_WINDOW_NUM)
            {
                ixtShared[flatIdx] += ixtShared[flatIdx + stride];
                iytShared[flatIdx] += iytShared[flatIdx + stride];
            }
            __syncthreads();
        }

        if (flatIdx == 0)
        {
            du = ((iyyShared[0] * -ixtShared[0]) + (-ixyShared[0] * -iytShared[0])) / det;
            dv = ((-ixyShared[0] * -ixtShared[0]) + (ixxShared[0] * -iytShared[0])) / det;

            u += du;
            v += dv;
        }
        __syncthreads();

        // Convergence check
        if (du * du + dv * dv < LK_EPSILON)
        {
            break;
        }
        __syncthreads();
    }

    if (flatIdx == 0)
    {
        float3 feature = features[featureNum];
        float updatedX = (feature.x + u);
        float updatedY = (feature.y + v);
        float status = feature.z;

        if (updatedX < 0 || updatedX >= width || updatedY < 0 || updatedY >= height)
        {
            status = 0;
        }
        features[featureNum] = make_float3(updatedX, updatedY, status);
    }
}

/**
 * Device code to fill up a mipmap level given a previous level.
 *
 * @param prevLevelTex Texture Object of the previous level
 * @param levelSurf Surface Object of the current level.
 * @param destWidth Destination MipMap level's width
 * @param destHeight Destination MipMap level's height
 */
__global__ void
fillMipMap(cudaTextureObject_t prevLevelTex, cudaSurfaceObject_t levelSurf, int destWidth, int destHeight)
{
    // Grab where the pixel is gonna go :P
    int destX = blockIdx.x * blockDim.x + threadIdx.x;
    int destY = blockIdx.y * blockDim.y + threadIdx.y;
    float srcX = 1.0 / float(destWidth);
    float srcY = 1.0 / float(destHeight);

    // Grab the original pixel height n stuff
    if (destX < destWidth && destY < destHeight)
    {
        float q11 = 255.0f * tex2D<float>(prevLevelTex, (destX + 0) * srcX, (destY + 0) * srcY);
        float q12 = 255.0f * tex2D<float>(prevLevelTex, (destX + 0) * srcX, (destY + 1) * srcY);
        float q21 = 255.0f * tex2D<float>(prevLevelTex, (destX + 1) * srcX, (destY + 0) * srcY);
        float q22 = 255.0f * tex2D<float>(prevLevelTex, (destX + 1) * srcX, (destY + 1) * srcY);

        u_char val = (u_char)((q11 + q12 + q21 + q22) * 0.25f);

        surf2Dwrite<u_char>(val, levelSurf, destX * sizeof(u_char), destY * sizeof(u_char), cudaBoundaryModeZero);
    }
}

/**
 * Host code to generate mipmaps for all levels of a texture.
 *
 * @param mipmap The mipmapped array to generate mipmaps off of.
 * @param extent The cudaExtent storing the width and height of the original input
 * @param levels How many levels in total to generate (not 0-indexed).
 */
void
generateMipMaps(cudaMipmappedArray_t *mipmap, cudaExtent *extent, unsigned int levels)
{
    // Modeled after:
    // https://github.com/NVIDIA/cuda-samples/blob/master/Samples/3_CUDA_Features/bindlessTexture/bindlessTexture_kernel.cu
    for (int level = 1; level < levels; level++)
    {
        cudaArray_t levelArray, prevLevelArray;
        cudaGetMipmappedArrayLevel(&levelArray, *mipmap, level);
        cudaGetMipmappedArrayLevel(&prevLevelArray, *mipmap, level - 1);

        cudaResourceDesc levelResDesc = {}, prevLevelResDesc = {};
        levelResDesc.resType = cudaResourceTypeArray;
        levelResDesc.res.array.array = levelArray;
        prevLevelResDesc.resType = cudaResourceTypeArray;
        prevLevelResDesc.res.array.array = prevLevelArray;

        cudaSurfaceObject_t levelSurf;
        cudaCreateSurfaceObject(&levelSurf, &levelResDesc);

        cudaTextureDesc prevLevelTexDesc = {};
        prevLevelTexDesc.addressMode[0] = cudaAddressModeClamp;
        prevLevelTexDesc.addressMode[1] = cudaAddressModeClamp;
        prevLevelTexDesc.filterMode = cudaFilterModeLinear;
        prevLevelTexDesc.readMode = cudaReadModeNormalizedFloat;
        prevLevelTexDesc.normalizedCoords = 1;

        cudaTextureObject_t prevLevelTex;
        cudaCreateTextureObject(&prevLevelTex, &prevLevelResDesc, &prevLevelTexDesc, nullptr);

        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
        int gridWidth = (int)(ceil((float)(extent->width >> level) / BLOCK_SIZE));
        int gridHeight = (int)(ceil((float)(extent->height >> level) / BLOCK_SIZE));
        dim3 gridDim(gridWidth, gridHeight, 1);

        cudaGetLastError(); // drain
        fillMipMap<<<gridDim, blockDim>>>(prevLevelTex, levelSurf, (extent->width) >> level, (extent->height) >> level);
        cudaDeviceSynchronize();

        cudaDestroySurfaceObject(levelSurf);
        cudaDestroyTextureObject(prevLevelTex);
    }
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
sparseLucasKanadeGPUMip(VideoInfo &video)
{
    if (video.frames.empty())
    {
        return;
    }

    int width = video.frames[0].cols;
    int height = video.frames[0].rows;

    // For coloring the output
    cv::Mat mask = cv::Mat::zeros(video.frames[0].size(), CV_8UC3);

    // === Pointer Declaration ===

    cudaMipmappedArray_t deviceFrameMipMap, devicePrevFrameMipMap;
    cudaTextureObject_t deviceFrameTex = {}, devicePrevFrameTex = {};
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
    cudaExtent extent = make_cudaExtent(width, height, 0);

    float3 *deviceFrameFeatures = NULL;
    int *deviceFrameFeatureCount = NULL;
    float *deviceResponse = NULL;

    // === Memory Allocation ===

    cudaMallocMipmappedArray(&deviceFrameMipMap, &channelDesc, extent, MAX_PYR_LEVELS);
    cudaMallocMipmappedArray(&devicePrevFrameMipMap, &channelDesc, extent, MAX_PYR_LEVELS);

    cudaMalloc(&deviceFrameFeatures, MAX_FEATURES * sizeof(float3));
    cudaMalloc(&deviceFrameFeatureCount, sizeof(int));
    cudaMalloc(&deviceResponse, width * height * sizeof(float));

    // === Memory Copying & Zeroing ===

    cudaMemset(deviceFrameFeatures, 0, MAX_FEATURES * sizeof(float3));
    cudaMemset(deviceFrameFeatureCount, 0, sizeof(int));
    cudaMemset(deviceResponse, 0, width * height * sizeof(float));

    // === MipMap Copying & Level Calculations ===

    cudaArray_t level0;
    cudaGetMipmappedArrayLevel(&level0, deviceFrameMipMap, 0);
    CUDA_CHECK(cudaMemcpy2DToArray(level0, 0, 0, video.frames[0].data, width * sizeof(u_char), width * sizeof(u_char), height,
                        cudaMemcpyHostToDevice));
    cudaGetMipmappedArrayLevel(&level0, devicePrevFrameMipMap, 0);
    CUDA_CHECK(cudaMemcpy2DToArray(level0, 0, 0, video.frames[0].data, width * sizeof(u_char), width * sizeof(u_char), height,
                        cudaMemcpyHostToDevice));

    generateMipMaps(&devicePrevFrameMipMap, &extent, MAX_PYR_LEVELS);
    generateMipMaps(&deviceFrameMipMap, &extent, MAX_PYR_LEVELS);

    // === Texture Memory Declaration & Initialization ===

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeMipmappedArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;
    texDesc.mipmapFilterMode = cudaFilterModePoint;
    texDesc.minMipmapLevelClamp = 0.0f;
    texDesc.maxMipmapLevelClamp = (float)(MAX_PYR_LEVELS - 1);

    resDesc.res.mipmap.mipmap = deviceFrameMipMap;
    CUDA_CHECK(cudaCreateTextureObject(&deviceFrameTex, &resDesc, &texDesc, nullptr));

    resDesc.res.mipmap.mipmap = devicePrevFrameMipMap;
    CUDA_CHECK(cudaCreateTextureObject(&devicePrevFrameTex, &resDesc, &texDesc, nullptr));

    // === Kernel Blocks & Grids Initialization ===

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridDim((int)ceil((float)width / blockDim.x), (int)ceil((float)height / blockDim.y), 1);

    // === First Frame Procedure ===
    // 1. Grab Harris Response
    // 2. Use Thrust Extrema to calculate threshold
    // 3. Threshold all features
    // 3. Perform Lucas Kanade Solve on every frame past that

    // == Initial Harris Response ==

    harrisResponseMip<<<gridDim, blockDim>>>(deviceResponse, deviceFrameTex, width, height);
    cudaDeviceSynchronize();

    // == Threshold Calculations w/ Thrust ==

    thrust::device_ptr<float> responsePtr(deviceResponse);
    float responseMax = *thrust::max_element(responsePtr, responsePtr + (width * height));
    float responseThreshold = responseMax * QUALITY_LEVEL;

    harrisThresholderMip<<<gridDim, blockDim>>>(deviceFrameFeatures, deviceFrameFeatureCount, deviceResponse,
                                                responseThreshold, MAX_FEATURES, width, height);
    cudaDeviceSynchronize();

    // == Getting Feature Counts ==

    int featureCount = 0;
    cudaMemcpy(&featureCount, deviceFrameFeatureCount, sizeof(int), cudaMemcpyDeviceToHost);
    /**
     * Below line is needed to prevent CUDA from accessing bad memory, otherwise we'll have no results.
     */
    featureCount = min(featureCount, MAX_FEATURES);
    float3 *prevFrameFeatures = (float3 *)calloc(featureCount, sizeof(float3));
    float3 *frameFeatures = (float3 *)calloc(featureCount, sizeof(float3));
    cudaMemcpy(prevFrameFeatures, deviceFrameFeatures, featureCount * sizeof(float3), cudaMemcpyDeviceToHost);
    std::vector<cv::Scalar> pt_colors = getRandomColors(featureCount);

    // Each block is responsible for each feature.
    dim3 featureBlockDim(LK_WINDOW_WIDTH, LK_WINDOW_WIDTH, 1);
    dim3 featureGridDim(featureCount, 1, 1);

    // == Repeated Frame LK Procedure ==

    for (int i = 1; i < video.frames.size(); i++)
    {
        // Switch Frames using pointer swapping
        std::swap(deviceFrameMipMap, devicePrevFrameMipMap);
        std::swap(deviceFrameTex, devicePrevFrameTex);

        // Remake the texture objects again.
        cudaArray_t level0;
        cudaGetMipmappedArrayLevel(&level0, deviceFrameMipMap, 0);
        cudaMemcpy2DToArray(level0, 0, 0, video.frames[i].data, width * sizeof(u_char), width * sizeof(u_char), height,
                            cudaMemcpyHostToDevice);

        generateMipMaps(&deviceFrameMipMap, &extent, MAX_PYR_LEVELS);

        cudaDestroyTextureObject(deviceFrameTex);
        resDesc.res.mipmap.mipmap = deviceFrameMipMap;
        cudaCreateTextureObject(&deviceFrameTex, &resDesc, &texDesc, nullptr);

        // Obtain Lucas Kanade Solve on 1 dimensional grid/block array
        iterLucasKanadeSolverMip<<<featureGridDim, featureBlockDim>>>(
            deviceFrameTex, devicePrevFrameTex, deviceFrameFeatures, deviceFrameFeatureCount, width, height);
        cudaDeviceSynchronize();

        cudaMemcpy(frameFeatures, deviceFrameFeatures, featureCount * sizeof(float3), cudaMemcpyDeviceToHost);

        cv::Mat output;
        cvtColor(video.frames[i], output, cv::COLOR_GRAY2BGR);
        drawSparseOpticalFlowGPU(output, mask, reinterpret_cast<cv::Vec3f *>(prevFrameFeatures),
                                 reinterpret_cast<cv::Vec3f *>(frameFeatures), featureCount, pt_colors,
                                 DRAW_CONTINUOUS_LINES);

        std::memcpy(prevFrameFeatures, frameFeatures, featureCount * sizeof(float3));
        video.outputFrames.push_back(output);
    }

    // === Memory Freeing Procedure ===

    free(prevFrameFeatures);
    free(frameFeatures);

    cudaDestroyTextureObject(deviceFrameTex);
    cudaDestroyTextureObject(devicePrevFrameTex);

    cudaFreeMipmappedArray(deviceFrameMipMap);
    cudaFreeMipmappedArray(devicePrevFrameMipMap);

    cudaFree(deviceFrameFeatures);
    cudaFree(deviceFrameFeatureCount);
    cudaFree(deviceResponse);
}
