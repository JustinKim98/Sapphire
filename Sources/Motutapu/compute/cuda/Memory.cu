// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/cuda/Memory.cuh>

namespace Motutapu::Compute::Cuda
{
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

__host__ __device__ bool CudaMallocFloat(float** ptr, unsigned int size)
{
    const cudaError_t error =
        cudaMalloc(reinterpret_cast<void**>(ptr), size * sizeof(float));
    return error == cudaSuccess;
}

__host__ __device__ bool CudaMallocHalf(half** ptr, unsigned int size)
{
    const cudaError_t error =
        cudaMalloc(reinterpret_cast<void**>(ptr), size * sizeof(float));
    return error == cudaSuccess;
}

template <typename T>
__host__ __device__ bool CudaFree(void** ptr)
{
    const cudaError_t error = cudaFree(ptr);
    return error == cudaSuccess;
}

template <typename T>
__host__ bool MemcpyHostToGpu(T* gpuPtr, T* hostPtr, unsigned int size)
{
    const cudaError_t error = cudaMemcpy(
        reinterpret_cast<void*>(gpuPtr), reinterpret_cast<void*>(hostPtr),
        size * sizeof(T), cudaMemcpyHostToDevice);

    return error == cudaSuccess;
}

__host__ bool MemcpyHostToGpuFloat(float* gpuPtr, float* hostPtr, unsigned int size)
{
    const cudaError_t error = cudaMemcpy(
        reinterpret_cast<void*>(gpuPtr), reinterpret_cast<void*>(hostPtr),
        size * sizeof(float), cudaMemcpyHostToDevice);

    return error == cudaSuccess;
}

__host__ bool MemcpyHostToGpuHalf(half* gpuPtr, half* hostPtr,
                                   unsigned int size)
{
    const cudaError_t error = cudaMemcpy(
        reinterpret_cast<void*>(gpuPtr), reinterpret_cast<void*>(hostPtr),
        size * sizeof(half), cudaMemcpyHostToDevice);

    return error == cudaSuccess;
}

__host__ bool MemcpyGpuToHostFloat(float* hostPtr, float* gpuPtr,
                                   unsigned int size)
{
    const cudaError_t error = cudaMemcpy(
        reinterpret_cast<void*>(hostPtr), reinterpret_cast<void*>(gpuPtr),
        size * sizeof(float), cudaMemcpyDeviceToHost);

    return error == cudaSuccess;
}

__host__ bool MemcpyGpuToHostHalf(half* hostPtr, half* gpuPtr,
                                   unsigned int size)
{
    const cudaError_t error = cudaMemcpy(
        reinterpret_cast<void*>(hostPtr), reinterpret_cast<void*>(gpuPtr),
        size * sizeof(half), cudaMemcpyDeviceToHost);

    return error == cudaSuccess;
}

__host__ void MemcpyGpuToGpuFloat(float* dest, const float* src, unsigned int size)
{
    unsigned int elementsCopied = 0;

    if (size > MAX_THREAD_DIM_X)
    {
        cudaStream_t stream0;
        cudaStreamCreate(&stream0);
        const auto requiredBlocks = size / MAX_THREAD_DIM_X;
        CopyOnGpu<float><<<requiredBlocks, MAX_THREAD_DIM_X>>>(
            dest, src, requiredBlocks * MAX_THREAD_DIM_X);

        elementsCopied += requiredBlocks * MAX_THREAD_DIM_X;
    }

    CopyOnGpu<float><<<1, size>>>(dest + elementsCopied, src + elementsCopied,
                                  size - elementsCopied);
}

__host__ void MemcpyGpuToGpuHalf(half* dest, const half* src,
                                 unsigned int size)
{
    unsigned int elementsCopied = 0;

    if (size > MAX_THREAD_DIM_X)
    {
        cudaStream_t stream0;
        cudaStreamCreate(&stream0);
        const auto requiredBlocks = size / MAX_THREAD_DIM_X;
        CopyOnGpu<half><<<requiredBlocks, MAX_THREAD_DIM_X>>>(
            dest, src, requiredBlocks * MAX_THREAD_DIM_X);

        elementsCopied += requiredBlocks * MAX_THREAD_DIM_X;
    }

    CopyOnGpu<half><<<1, size>>>(dest + elementsCopied, src + elementsCopied,
                                  size - elementsCopied);
}
}
