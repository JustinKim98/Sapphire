// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_COMPUTE_CUDA_MEMORY_CUH
#define MOTUTAPU_COMPUTE_CUDA_MEMORY_CUH

#include <Motutapu/compute/cuda/CudaParams.cuh>

namespace Motutapu::Compute::Cuda
{
 bool CudaSetDevice(int deviceId);

bool CudaMalloc(void** ptr, unsigned int byteSize);

bool CudaFree(void* ptr);

bool CopyHostToDevice(void* devicePtr, void* hostPtr, unsigned int byteSize);

bool CopyDeviceToHost(void* hostPtr, void* devicePtr, unsigned int byteSize);

bool CopyDeviceToDevice(void* dst, const void* src, unsigned int byteSize);

bool CopyDeviceToDeviceAsync(float* dst, const float* src,
                                unsigned int byteSize, cudaStream_t stream);

bool CopyDeviceToDeviceBroadcast(void* dst, const void* src,
                                    unsigned int byteSize, unsigned int srcStrideByteSize);
}  // namespace Motutapu::Compute::Cuda
#endif
