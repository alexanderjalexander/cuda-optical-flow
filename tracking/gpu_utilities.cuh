#ifndef GPU_UTILITIES_CUH
#define GPU_UTILITIES_CUH

#include "gpu_utilities.cuh"
#include "lucasKanade.hpp"

#include <stdio.h>

/**

Terminology for the functions that load shared memory:

* Global data is the "input" to be loaded.
* Shared memory is the "output" where the input data is saved.
* Blocks are hardware-backed groups of threads on the GPU.
* Tiles are conceptual groups of threads that load data from global to shared memory. They are the same as Blocks when
using 2D blocks and grids, but not with 1D blocks and grids.

*/

/**
 * Core internal logic for loading a square tile of data into 2D shared memory including a halo for convolution
 * operations. Works regardless of block or grid shape based on provided indices within the corresponding tile.
 *
 * @param shared  Device shared memory to load the data into, expected to point to the first element of a 2D array.
 * @param global Device global memory array containing input data.
 * @param halo_radius The radius of the halo to load around the internal tile (e.g. 1 for a 3x3 mask, 2 for a 5x5 mask,
 * etc.).
 * @param widthGlobal The width (in elements) of the global memory data.
 * @param heightGlobal The height (in elements) of the global memory data.
 * @param tileSize The size (both width and height) of the shared memory tile to be loaded, not including the halo.
 * @param tx The index of the current thread within the tile in the X axis.
 * @param ty The index of the current thread within the tile in the Y axis.
 * @param xGlobal The index of the current thread within the global memory in the X axis.
 * @param yGlobal The index of the current thread within the global memory in the Y axis.
 */
template <typename T>
__device__ void
load2dSharedMemoryCore(T *shared, T *global, int halo_radius, int widthGlobal, int heightGlobal, int tileSize, int tx,
                       int ty, int xGlobal, int yGlobal)
{
    // identify the coordinates of the output data to work on, accounting for the differences between tile size and
    // shared memory size
    int xShared = tx + halo_radius;
    int yShared = ty + halo_radius;
    int widthShared = tileSize + (halo_radius * 2);

    // internal elements - loaded directly by corresponding threads, offset in
    // shared memory by halo size
    shared[yShared * widthShared + xShared] =
        (yGlobal < heightGlobal && xGlobal < widthGlobal) ? global[yGlobal * widthGlobal + xGlobal] : 0;

    // top halo edge - loaded by bottom edge of threads
    int halo_top_row = yGlobal - tileSize;
    if (ty >= tileSize - halo_radius)
    {
        shared[(ty - (tileSize - halo_radius)) * widthShared + xShared] =
            (halo_top_row < 0) ? 0 : global[halo_top_row * widthGlobal + xGlobal];
    }

    // bottom halo edge - loaded by top edge of threads
    int halo_bottom_row = yGlobal + tileSize;
    if (ty < halo_radius)
    {
        shared[(ty + tileSize + halo_radius) * widthShared + xShared] =
            (halo_bottom_row >= heightGlobal) ? 0 : global[halo_bottom_row * widthGlobal + xGlobal];
    }

    // left halo edge - loaded by right edge of threads
    int halo_left_col = xGlobal - tileSize;
    if (tx >= tileSize - halo_radius)
    {
        shared[yShared * widthShared + (tx - (tileSize - halo_radius))] =
            (halo_left_col < 0) ? 0 : global[yGlobal * widthGlobal + halo_left_col];
    }

    // right halo edge - loaded by left edge of threads
    int halo_right_col = xGlobal + tileSize;
    if (tx < halo_radius)
    {
        shared[yShared * widthShared + (tx + tileSize + halo_radius)] =
            (halo_right_col >= widthGlobal) ? 0 : global[yGlobal * widthGlobal + halo_right_col];
    }

    // halo corners - each loaded by the thread farthest from it (e.g. top right
    // of halo by bottom left thread) logic is a combination of the conditions
    // above
    if (ty >= tileSize - halo_radius)
    {
        // top edge of halo

        if (tx >= tileSize - halo_radius)
        {
            // top left halo corner
            shared[(ty - (tileSize - halo_radius)) * widthShared + (tx - (tileSize - halo_radius))] =
                (halo_top_row < 0 || halo_left_col < 0) ? 0 : global[halo_top_row * widthGlobal + halo_left_col];
        }
        else if (tx < halo_radius)
        {
            // top right halo corner
            shared[(ty - (tileSize - halo_radius)) * widthShared + (tx + tileSize + halo_radius)] =
                (halo_top_row < 0 || halo_right_col >= widthGlobal)
                    ? 0
                    : global[halo_top_row * widthGlobal + halo_right_col];
        }
    }
    else if (ty < halo_radius)
    {
        // bottom edge of halo

        if (tx >= tileSize - halo_radius)
        {
            // bottom left halo corner
            shared[(ty + tileSize + halo_radius) * widthShared + (tx - (tileSize - halo_radius))] =
                (halo_bottom_row >= heightGlobal || halo_left_col < 0)
                    ? 0
                    : global[halo_bottom_row * widthGlobal + halo_left_col];
        }
        else if (tx < halo_radius)
        {
            // bottom right halo corner
            shared[(ty + tileSize + halo_radius) * widthShared + (tx + tileSize + halo_radius)] =
                (halo_bottom_row >= heightGlobal || halo_right_col >= widthGlobal)
                    ? 0
                    : global[halo_bottom_row * widthGlobal + halo_right_col];
        }
    }

    // Note: letting the caller synchronize means multiple shared memory loads can be done back to back without
    // unnecessary synchronizations between them
    // __syncthreads();
}

/**
 * Device-only code to collaboratively load a tile of data into 2D shared memory, including a halo for convolution
 * operations. Loads zero if the thread corresponds to a data or halo element outside the original frame. Does NOT
 * call syncthreads, so the caller is responsible for doing this to ensure all threads have finished loading their
 * data before any subsequent operations.
 *
 * @param shared  Device shared memory to load the data into, expected to point to the first element of a 2D array.
 * @param global Device global memory array containing input data.
 * @param halo_radius The radius of the halo to load around the internal tile (e.g. 1 for a 3x3 mask, 2 for a 5x5 mask,
 * etc.).
 * @param widthGlobal The width (in elements) of the global memory data.
 * @param heightGlobal The height (in elements) of the global memory data.
 */
template <typename T>
__device__ void
load2dSharedMemoryWithHalo(T *shared, T *global, int halo_radius, int widthGlobal, int heightGlobal)
{
    // with 2D blocks, tile == block
    int tx = threadIdx.x, ty = threadIdx.y;
    int xGlobal = tx + blockIdx.x * blockDim.x;
    int yGlobal = ty + blockIdx.y * blockDim.y;

    load2dSharedMemoryCore<T>(shared, global, halo_radius, widthGlobal, heightGlobal, blockDim.x, tx, ty, xGlobal,
                              yGlobal);
}

/**
 * Device-only code to collaboratively load a tile of data into 2D shared memory, including a halo for convolution
 * operations. Should only be used when the calling thread is part of a 1D block and grid. Loads zero if the thread
 * corresponds to a data or halo element outside the original frame. Does NOT call syncthreads, so the caller is
 * responsible for doing this to ensure all threads have finished loading their data before any subsequent operations.
 *
 * @param shared  Device shared memory to load the data into, expected to point to the first element of a 2D array.
 * @param global Device global memory array containing input data.
 * @param tileSize The size (both width and height) of the shared memory tile to be loaded, not including the halo.
 * @param halo_radius The radius of the halo to load around the internal tile (e.g. 1 for a 3x3 mask, 2 for a 5x5 mask,
 * etc.).
 * @param widthGlobal The width (in elements) of the global memory data.
 * @param heightGlobal The height (in elements) of the global memory data.
 */
template <typename T>
__device__ void
load2dSharedMemoryWithHalo1dBlock(T *shared, T *global, int tileSize, int halo_radius, int widthGlobal,
                                  int heightGlobal)
{
    // with 1D blocks, manually convert 1D coordinates into 2D
    int localIdx = threadIdx.x, bx = blockIdx.x;

    // calculate the tile location within the global data
    int tilesPerRow = (int)ceil((float)widthGlobal / tileSize);
    int tileX = bx % tilesPerRow;
    int tileY = bx / tilesPerRow;

    // calculate the thread location within the tile
    int tx = localIdx % tileSize;
    int ty = localIdx / tileSize;

    // calculate the global index
    int xGlobal = tileX * tileSize + tx;
    int yGlobal = tileY * tileSize + ty;

    load2dSharedMemoryCore<T>(shared, global, halo_radius, widthGlobal, heightGlobal, tileSize, tx, ty, xGlobal,
                              yGlobal);
}

// todo broken 1D to 1D version

// load1dSharedMemoryWithHalo1dBlock(T *shared, T *global, int halo_radius, int width, int height)
// {
//     // convert the 1D thread and block indices into 2D coordinates of the output data to work on
//     int tx = threadIdx.x, bx = blockIdx.x;
//     int xGlobal = tx + bx * blockDim.x;
//     int xShared = (xGlobal % width) + halo_radius;
//     int yShared = (xGlobal / width) + halo_radius;

//     // internal elements - loaded directly by corresponding threads, offset in
//     // shared memory by halo size
//     shared[yShared][xShared] = (xGlobal < width) ? global[xGlobal] : 0;

//     // TODO MG CURRENT - make compatible with 1D!
//     // TODO MG - this would only work for loading a 1D shared memory, and needs to be translated with the new 1d
//     param

//     // left side halo - loaded by right side of threads
//     int halo_left_col = (blockIdx.x - 1) * BLOCK_SIZE + tx;
//     if (tx >= BLOCK_SIZE - halo_radius)
//     {
//         shared[yShared][tx - (BLOCK_SIZE - halo_radius)] = (halo_left_col < 0) ? 0 : global[yGlobal * width +
//         halo_left_col];
//     }

//     // right side halo - loaded by left side of threads
//     int halo_right_col = (blockIdx.x + 1) * BLOCK_SIZE + tx;
//     if (tx < halo_radius)
//     {
//         shared[yShared][tx + BLOCK_SIZE + halo_radius] =
//             (halo_right_col >= width) ? 0 : global[yGlobal * width + halo_right_col];
//     }

//     __syncthreads();
// }
#endif