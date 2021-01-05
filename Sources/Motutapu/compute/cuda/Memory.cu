// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/cuda/Memory.hpp>
#include <cuda_runtime.h>

__host__ bool CudaSetDevice(int deviceId)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceId < deviceCount)
    {
        const cudaError_t error = cudaSetDevice(deviceId);
        return error == cudaSuccess;
    }
    return false;
}

__host__ __device__ bool CudaMalloc(void** ptr, size_t bytes)
{
    const cudaError_t error = cudaMalloc(ptr, bytes);
    return error == cudaSuccess;
}

__host__ __device__ bool CudaFree(void** ptr)
{
    const cudaError_t error = cudaFree(ptr);
    return error == cudaSuccess;
}

__host__ bool MemcpyHostToGpu(void* gpuPtr, void* hostPtr, size_t bytes)
{
    const cudaError_t error =
        cudaMemcpy(gpuPtr, hostPtr, bytes, cudaMemcpyHostToDevice);

    return error == cudaSuccess;
}

__host__ bool MemcpyGpuToHost(void* hostPtr, void* gpuPtr, size_t bytes)
{
    const cudaError_t error =
        cudaMemcpy(hostPtr, gpuPtr, bytes, cudaMemcpyDeviceToHost);

    return error == cudaSuccess;
}
