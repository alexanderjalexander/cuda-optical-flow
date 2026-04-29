#include "gpu_utilities.cuh"
#include "lucasKanade.hpp"

void
initSobelGaussianKernel(float sigma)
{
    constexpr int SIZE = 2 * SOBEL_MASK_RAD + 1;
    float kernel[SIZE][SIZE];
    float sum = 0.0f;

    for (int y = -SOBEL_MASK_RAD; y <= SOBEL_MASK_RAD; y++)
    {
        for (int x = -SOBEL_MASK_RAD; x <= SOBEL_MASK_RAD; x++)
        {
            float val = expf(-(x * x + y * y) / (2.0f * sigma * sigma));
            kernel[y + SOBEL_MASK_RAD][x + SOBEL_MASK_RAD] = val;
            sum += val;
        }
    }

    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            kernel[i][j] /= sum;
        }
    }

    cudaMemcpyToSymbol(sobelGaussianKernel, kernel, SIZE * SIZE * sizeof(float));
}

void
initHarrisGaussianKernel(float sigma)
{
    constexpr int SIZE = 2 * HARRIS_MASK_RAD + 1;
    float kernel[SIZE][SIZE];
    float sum = 0.0f;

    for (int y = -HARRIS_MASK_RAD; y <= HARRIS_MASK_RAD; y++)
    {
        for (int x = -HARRIS_MASK_RAD; x <= HARRIS_MASK_RAD; x++)
        {
            float val = expf(-(x * x + y * y) / (2.0f * sigma * sigma));
            kernel[y + HARRIS_MASK_RAD][x + HARRIS_MASK_RAD] = val;
            sum += val;
        }
    }

    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            kernel[i][j] /= sum;
        }
    }

    cudaMemcpyToSymbol(harrisGaussianKernel, kernel, SIZE * SIZE * sizeof(float));
}
