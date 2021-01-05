// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_CUDA_MEMORY_HPP
#define MOTUTAPU_CUDA_MEMORY_HPP
#ifdef WITH_CUDA
#include <cuda_runtime.h>

__host__ bool CudaSetDevice(int deviceId);

__host__ __device__ bool CudaMalloc(void** ptr, size_t bytes);

__host__ __device__ bool CudaFree(void** ptr);

__host__ bool MemcpyHostToGpu(void* gpuPtr, void* hostPtr, size_t bytes);

__host__ bool MemcpyGpuToHost(void* hostPtr, void* gpuPtr, size_t bytes);



#endif
#endif
