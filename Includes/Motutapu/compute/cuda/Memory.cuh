// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_COMPUTE_CUDA_MEMORY_CUH
#define MOTUTAPU_COMPUTE_CUDA_MEMORY_CUH

#include <Motutapu/compute/cuda/CudaParams.cuh>

namespace Motutapu::Compute::Cuda
{
__host__ bool CudaSetDevice(int deviceId);

__host__ __device__ bool CudaMalloc(void** ptr, unsigned int size);

__host__ __device__ bool CudaFree(void* ptr);

__host__ __device__ bool CudaFree(void* ptr);

__host__ bool MemcpyHostToGpu(void* gpuPtr, void* hostPtr, unsigned int size);

__host__ bool MemcpyGpuToHost(void* hostPtr, void* gpuPtr, unsigned int size);

__host__ void MemcpyGpuToGpu(float* dest, const float* src, unsigned int size);

__host__ void MemcpyGpuToGpuBroadcast(float* dest, const float* src,
                                      unsigned int size,
                                      unsigned int srcStride);
}  // namespace Motutapu::Compute::Cuda
#endif
