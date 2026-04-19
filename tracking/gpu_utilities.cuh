#ifndef GPU_UTILITIES_CUH
#define GPU_UTILITIES_CUH

template <typename T>
__device__ void load2dSharedMemoryWithHalo(T **shared, T *global, int halo_radius, int width, int height);

template <typename T>
__device__ void load2dSharedMemoryWithHalo1dBlock(T **shared, T *global, int halo_radius, int width, int height);

#endif