// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/cudaUtil/Memory.hpp>
#include <stdexcept>

namespace Sapphire::Compute::Cuda
{
void CudaSetDevice(int deviceId)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceId < deviceCount)
    {
        const cudaError_t error = cudaSetDevice(deviceId);
        if (error != cudaSuccess)
            throw std::runtime_error("CudaSetDevice failed with " +
                                     std::to_string(error));
    }
    else
        throw std::runtime_error(
            "CudaSetDevice - deviceId exceeds number of available devices");
}

void CudaMalloc(void** ptr, unsigned int byteSize)
{
    const cudaError_t error = cudaMalloc((void**)ptr, byteSize);
    if (error != cudaSuccess)
        throw std::runtime_error("CudaMalloc failed with " +
                                 std::to_string(error));
}

void CudaFree(void* ptr)
{
    const cudaError_t error = cudaFree((void*)(ptr));
    if (error != cudaSuccess)
        throw std::runtime_error("CudaFree failed with " +
                                 std::to_string(error));
}

void CopyHostToDevice(void* devicePtr, void* hostPtr, unsigned int byteSize)
{
    const cudaError_t error = cudaMemcpy((void*)(devicePtr), (void*)(hostPtr),
                                         byteSize, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
        throw std::runtime_error("CopyDeviceToDevice failed with " +
                                 std::to_string(error));
}

void CopyDeviceToHost(void* hostPtr, void* devicePtr, unsigned int byteSize)
{
    const cudaError_t error = cudaMemcpy((void*)(hostPtr), (void*)(devicePtr),
                                         byteSize, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
        throw std::runtime_error("CopyDeviceToHost failed with " +
                                 std::to_string(error));
}

void CopyDeviceToDevice(void* dst, const void* src, unsigned int byteSize)
{
    const cudaError_t error =
        cudaMemcpy(dst, src, byteSize, cudaMemcpyDeviceToDevice);
    if (error != cudaSuccess)
        throw std::runtime_error("CopyDeviceToDevice failed with " +
                                 std::to_string(error));
}

void CopyDeviceToDeviceAsync(void* dst, const void* src, unsigned int byteSize,
                             cudaStream_t stream)
{
    const cudaError_t error =
        cudaMemcpyAsync(dst, src, byteSize, cudaMemcpyDeviceToDevice, stream);
    if (error != cudaSuccess)
        throw std::runtime_error("CopyDeviceToDeviceAsync failed with " +
                                 std::to_string(error));
}

void CopyDeviceToDeviceBroadcast(void* dst, const void* src,
                                 unsigned int byteSize,
                                 unsigned int srcStrideByteSize)
{
    for (unsigned int idx = 0; idx < byteSize; idx += srcStrideByteSize)
    {
        const cudaError_t error =
            cudaMemcpy(static_cast<uint8_t*>(dst) + idx, src, srcStrideByteSize,
                       cudaMemcpyDeviceToDevice);
        if (error != cudaSuccess)
            throw std::runtime_error(
                "CopyDeviceToDeviceBroadcast failed in idx" +
                std::to_string(idx) + " with " + std::to_string(error));
    }
}

}  // namespace Sapphire::Compute::Cuda
