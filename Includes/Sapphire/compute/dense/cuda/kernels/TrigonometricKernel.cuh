// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_COMPUTE_DENSE_CUDA_TRIGONOMETRIC_KERNEL_CUH
#define SAPPHIRE_COMPUTE_DENSE_CUDA_TRIGONOMETRIC_KERNEL_CUH

#include <Sapphire/compute/cudaUtil/CudaParams.cuh>

namespace Sapphire::Compute::Dense::Cuda
{
__global__ void CosKernel(float* output, const float* input,
                          unsigned int totalSize);

__global__ void SinKernel(float* output, const float* input,
                          unsigned int totalSize);

__global__ void TanKernel(float* y, const float* x, unsigned int totalSize);

__global__ void CoshKernel(float* y, const float* x, unsigned int totalSize);

__global__ void SinhKernel(float* y, const float* x, unsigned int totalSize);

__global__ void TanhKernel(float* y, const float* x, unsigned int totalSize);

__global__ void ArcCosKernel(float* output, const float* input,
                             unsigned int totalSize);

__global__ void ArcSinKernel(float* output, const float* input,
                             unsigned int totalSize);

__global__ void ArcTanKernel(float* y, const float* x, unsigned int totalSize);

__global__ void ArcCoshKernel(float* y, const float* x, unsigned int totalSize);

__global__ void ArcSinhKernel(float* y, const float* x, unsigned int totalSize);

__global__ void ArcTanhKernel(float* y, const float* x, unsigned int totalSize);
}

#endif
