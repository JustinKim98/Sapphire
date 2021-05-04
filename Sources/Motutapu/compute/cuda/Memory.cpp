// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/cuda/Memory.hpp>

namespace Motutapu::Compute::Cuda
{
//__global__ void CopyOnDeviceKernelBroadcast(float* dst, const float* const src,
//                                            unsigned int srcStride,
//                                            unsigned int size)
//{
//    const auto sizePerBlock = size / gridDim.x;
//    const auto numLoops = sizePerBlock / blockDim.x;
//    const auto blockOffset = sizePerBlock * blockIdx.x;
//
//    for (unsigned int i = 0; i < numLoops; i++)
//    {
//        dst[blockOffset + blockDim.x * i + threadIdx.x] =
//            src[(blockOffset + blockDim.x * i + threadIdx.x) % srcStride];
//    }
//}
//
//__global__ void CopyOnDeviceKernel(float* dst, const float* const src,
//                                   unsigned int size)
//{
//    const auto sizePerBlock = size / gridDim.x;
//    const auto numLoops = sizePerBlock / blockDim.x;
//    const auto blockOffset = sizePerBlock * blockIdx.x;
//
//    for (unsigned int i = 0; i < numLoops; i++)
//    {
//        dst[blockOffset + blockDim.x * i + threadIdx.x] =
//            src[blockOffset + blockDim.x * i + threadIdx.x];
//    }
//}

bool CudaSetDevice(int deviceId)
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

bool CudaMalloc(void** ptr, unsigned int byteSize)
{
    const cudaError_t error = cudaMalloc((void**)ptr, byteSize);
    return error == cudaSuccess;
}

bool CudaFree(void* ptr)
{
    const cudaError_t error = cudaFree((void*)(ptr));
    return error == cudaSuccess;
}

bool CopyHostToDevice(void* devicePtr, void* hostPtr,
                               unsigned int byteSize)
{
    const cudaError_t error = cudaMemcpy((void*)(devicePtr), (void*)(hostPtr),
                                         byteSize, cudaMemcpyHostToDevice);

    return error == cudaSuccess;
}

bool CopyDeviceToHost(void* hostPtr, void* devicePtr,
                               unsigned int byteSize)
{
    const cudaError_t error = cudaMemcpy((void*)(hostPtr), (void*)(devicePtr),
                                         byteSize, cudaMemcpyDeviceToHost);

    return error == cudaSuccess;
}

bool CopyDeviceToDevice(void* dst, const void* src,
                                 unsigned int byteSize)
{
    const cudaError_t error =
        cudaMemcpy(dst, src, byteSize, cudaMemcpyDeviceToDevice);
    return error == cudaSuccess;
}

bool CopyDeviceToDeviceAsync(float* dst, const float* src,
                                      unsigned int byteSize,
                                      cudaStream_t stream)
{
    const cudaError_t error =
        cudaMemcpyAsync(dst, src, byteSize, cudaMemcpyDeviceToDevice, stream);
    return error == cudaSuccess;
}

bool CopyDeviceToDeviceBroadcast(void* dst, const void* src,
                                          unsigned int byteSize,
                                          unsigned int srcStrideByteSize)
{
    for (unsigned int idx = 0; idx < byteSize; idx += srcStrideByteSize)
    {
        const cudaError_t error =
            cudaMemcpy(static_cast<uint8_t*>(dst) + idx, src, srcStrideByteSize,
                       cudaMemcpyDeviceToDevice);
        if (error != cudaSuccess)
            return false;
    }

    return true;
}

}  // namespace Motutapu::Compute::Cuda
