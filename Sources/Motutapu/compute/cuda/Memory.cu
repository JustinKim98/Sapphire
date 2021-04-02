// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/cuda/Memory.cuh>

namespace Motutapu::Compute::Cuda
{
__global__ void CopyOnGpuKernelBroadcast(float* dst, const float* const src,
                                         unsigned int srcStride,
                                         unsigned int size)
{
    const auto sizePerBlock = size / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        dst[blockOffset + blockDim.x * i + threadIdx.x] =
            src[(blockOffset + blockDim.x * i + threadIdx.x) % srcStride];
    }
}

__global__ void CopyOnGpuKernel(float* dst, const float* const src,
                                unsigned int size)
{
    const auto sizePerBlock = size / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        dst[blockOffset + blockDim.x * i + threadIdx.x] =
            src[blockOffset + blockDim.x * i + threadIdx.x];
    }
}

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

__host__ __device__ bool CudaMalloc(void** ptr, unsigned int size)
{
    const cudaError_t error = cudaMalloc((void**)ptr, size);
    return error == cudaSuccess;
}

__host__ __device__ bool CudaFree(void* ptr)
{
    const cudaError_t error = cudaFree((void*)(ptr));
    return error == cudaSuccess;
}

__host__ bool CopyHostToGpu(void* gpuPtr, void* hostPtr, unsigned int size)
{
    const cudaError_t error = cudaMemcpy((void*)(gpuPtr), (void*)(hostPtr),
                                         size, cudaMemcpyHostToDevice);

    return error == cudaSuccess;
}

__host__ bool CopyGpuToHost(void* hostPtr, void* gpuPtr, unsigned int size)
{
    const cudaError_t error = cudaMemcpy((void*)(hostPtr), (void*)(gpuPtr),
                                         size, cudaMemcpyDeviceToHost);

    return error == cudaSuccess;
}

__host__ void CopyGpuToGpu(void* dst, const void* src, unsigned int size)
{
    const auto numLoops = 16;
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

    const auto blockDim = size / (threadDim * numLoops);
    const auto firstLaunchSize = blockDim * threadDim * numLoops;

    if (firstLaunchSize > 0)
        CopyOnGpuKernel<<<blockDim, threadDim>>>(dst, src, firstLaunchSize);
    if (size > firstLaunchSize)
        CopyOnGpuKernel<<<1, size - firstLaunchSize>>>(dst + firstLaunchSize,
                                                       src + firstLaunchSize,
                                                       size - firstLaunchSize);
}

__host__ void CopyGpuToGpuAsync(float* dst, const float* src, unsigned int size,
                                cudaStream_t stream)
{
    const auto numLoops = 16;
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

    const auto blockDim = size / (threadDim * numLoops);
    const auto firstLaunchSize = blockDim * threadDim * numLoops;

    if (firstLaunchSize > 0)
        CopyOnGpuKernel<<<blockDim, threadDim, 0, stream>>>(dst, src,
                                                            firstLaunchSize);
    cudaStreamSynchronize(stream);

    if (size > firstLaunchSize)
        CopyOnGpuKernel<<<1, size - firstLaunchSize, 0, stream>>>(
            dst + firstLaunchSize, src + firstLaunchSize,
            size - firstLaunchSize);
}

__host__ void CopyGpuToGpuBroadcast(float* dst, const float* src,
                                    unsigned int size, unsigned int srcStride)
{
    const auto numLoops = 8;
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

    const auto blockDim = size / (threadDim * numLoops);
    const auto firstLaunchSize = blockDim * threadDim * numLoops;

    if (firstLaunchSize > 0)
        CopyOnGpuKernelBroadcast<<<blockDim, threadDim>>>(dst, src, srcStride,
                                                          firstLaunchSize);
    if (size > firstLaunchSize)
        CopyOnGpuKernelBroadcast<<<1, size - firstLaunchSize>>>(
            dst + firstLaunchSize, src, srcStride, size - firstLaunchSize);
}

}  // namespace Motutapu::Compute::Cuda
