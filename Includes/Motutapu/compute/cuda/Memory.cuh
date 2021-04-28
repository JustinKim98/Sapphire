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

__host__ bool CopyHostToGpu(void* gpuPtr, void* hostPtr, unsigned int size);

__host__ bool CopyGpuToHost(void* hostPtr, void* gpuPtr, unsigned int size);

__host__ bool CopyGpuToGpu(void* dst, const void* src, unsigned int byteSize);

__host__ bool CopyGpuToGpuAsync(float* dst, const float* src,
                                unsigned int byteSize, cudaStream_t stream);

__host__ bool CopyGpuToGpuBroadcast(float* dst, const float* src,
                                    unsigned int byteSize, unsigned int srcStrideByteSize);
}  // namespace Motutapu::Compute::Cuda
#endif
