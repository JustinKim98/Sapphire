// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Sapphire_COMPUTE_CUDA_MEMORY_CUH
#define Sapphire_COMPUTE_CUDA_MEMORY_CUH

#include <Sapphire/compute/cudaUtil/CudaParams.cuh>

namespace Sapphire::Compute::Cuda
{
 bool CudaSetDevice(int deviceId);

bool CudaMalloc(void** ptr, unsigned int byteSize);

bool CudaFree(void* ptr);

bool CopyHostToDevice(void* devicePtr, void* hostPtr, unsigned int byteSize);

bool CopyDeviceToHost(void* hostPtr, void* devicePtr, unsigned int byteSize);

bool CopyDeviceToDevice(void* dst, const void* src, unsigned int byteSize);

bool CopyDeviceToDeviceAsync(void* dst, const void* src,
                                unsigned int byteSize, cudaStream_t stream);

bool CopyDeviceToDeviceBroadcast(void* dst, const void* src,
                                    unsigned int byteSize, unsigned int srcStrideByteSize);
}  // namespace Sapphire::Compute::Cuda
#endif
