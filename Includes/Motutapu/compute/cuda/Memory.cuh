// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_COMPUTE_CUDA_MEMORY_CUH
#define MOTUTAPU_COMPUTE_CUDA_MEMORY_CUH

#include <cuda_fp16.h>
#include <Motutapu/compute/cuda/CudaParams.hpp>

namespace Motutapu::Compute::Cuda
{
__host__ bool CudaSetDevice(int deviceId);

template <typename T>
__global__ void CopyOnGpu(T* dest, const T* const src, unsigned int size)
{
    const auto index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < size)
    {
        dest[index] = src[index];
    }
}

__host__ __device__ bool CudaMallocFloat(float** ptr, unsigned int size);

__host__ __device__ bool CudaMallocHalf(half** ptr, unsigned int size);


__host__ __device__ bool CudaFree(void** ptr);

__host__ bool MemcpyHostToGpuFloat(float* gpuPtr, float* hostPtr, unsigned int size);

__host__ bool MemcpyHostToGpuHalf(half* gpuPtr, half* hostPtr,
                                   unsigned int size);

__host__ bool MemcpyGpuToHostFloat(float* hostPtr, float* gpuPtr,
                                   unsigned int size);

__host__ bool MemcpyGpuToHostHalf(half* hostPtr, half* gpuPtr,
                                   unsigned int size);

__host__ void MemcpyGpuToGpuFloat(float* dest, const float* src, unsigned int size);

__host__ void MemcpyGpuToGpuHalf(half* dest, const half* src,
                                  unsigned int size);

}  // namespace Motutapu::Compute::Cuda
#endif
