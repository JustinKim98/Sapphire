// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Sapphire_COMPUTE_CUDA_MEMORY_CUH
#define Sapphire_COMPUTE_CUDA_MEMORY_CUH

#ifdef WITH_CUDA
#include <Sapphire/compute/cudaUtil/CudaParams.cuh>

namespace Sapphire::Compute::Cuda
{
void CudaSetDevice(int deviceId);

 void CudaMalloc(void** ptr, unsigned int byteSize);

 void CudaFree(void* ptr);

void CopyHostToDevice(void* devicePtr, void* hostPtr, unsigned int byteSize);

void CopyDeviceToHost(void* hostPtr, void* devicePtr, unsigned int byteSize);

void CopyDeviceToDevice(void* dst, const void* src, unsigned int byteSize);

void CopyDeviceToDeviceAsync(void* dst, const void* src,
                                unsigned int byteSize, cudaStream_t stream);

void CopyDeviceToDeviceBroadcast(void* dst, const void* src,
                                    unsigned int byteSize, unsigned int srcStrideByteSize);
}  // namespace Sapphire::Compute::Cuda
#endif
#endif
