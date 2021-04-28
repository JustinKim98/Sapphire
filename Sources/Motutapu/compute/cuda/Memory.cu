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

__host__ bool CopyGpuToGpu(void* dst, const void* src, unsigned int byteSize)
{
    const cudaError_t error =
        cudaMemcpy(dst, src, byteSize, cudaMemcpyDeviceToDevice);
    return error == cudaSuccess;
}

__host__ bool CopyGpuToGpuAsync(float* dst, const float* src,
                                unsigned int byteSize, cudaStream_t stream)
{
    const cudaError_t error =
        cudaMemcpyAsync(dst, src, byteSize, cudaMemcpyDeviceToDevice, stream);
    return error == cudaSuccess;
}

__host__ bool CopyGpuToGpuBroadcast(float* dst, const float* src,
                                    unsigned int byteSize,
                                    unsigned int srcStrideByteSize)
{
    for (unsigned int idx = 0; idx < byteSize; idx += srcStrideByteSize)
    {
        const cudaError_t error = cudaMemcpy(dst + idx, src, srcStrideByteSize,
                                             cudaMemcpyDeviceToDevice);
        if (error != cudaSuccess)
            return false;
    }

    return true;
}

}  // namespace Motutapu::Compute::Cuda
