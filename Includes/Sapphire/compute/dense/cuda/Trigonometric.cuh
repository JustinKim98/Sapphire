// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_COMPUTE_DENSE_CUDA_TRIGONOMETRIC_CUH
#define SAPPHIRE_COMPUTE_DENSE_CUDA_TRIGONOMETRIC_CUH

#include <Sapphire/compute/cudaUtil/CudaParams.cuh>

namespace Sapphire::Compute::Dense::Cuda
{
__host__ void Cos(float* y, const float* x, unsigned int totalSize);

__host__ void Sin(float* y, const float* x, unsigned int totalSize);

__host__ void Tan(float* y, const float* x, unsigned int totalSize);

__host__ void Cosh(float* y, const float* x, unsigned int totalSize);

__host__ void Sinh(float* y, const float* x, unsigned int totalSize);

__host__ void Tanh(float* y, const float* x, unsigned int totalSize);

__host__ void ArcCos(float* y, const float* x, unsigned int totalSize);

__host__ void ArcSin(float* y, const float* x, unsigned int totalSize);

__host__ void ArcTan(float* y, const float* x, unsigned int totalSize);

__host__ void ArcCosh(float* y, const float* x, unsigned int totalSize);

__host__ void ArcSinh(float* y, const float* x, unsigned int totalSize);

__host__ void ArcTanh(float* y, const float* x, unsigned int totalSize);

__host__ void CosBackward(float* dx, const float* dy, const float* x,
                          unsigned int totalSize);

__host__ void SinBackward(float* dx, const float* dy, const float* x,
                          unsigned int totalSize);

__host__ void TanBackward(float* dx, const float* dy, const float* x,
                          unsigned int totalSize);

__host__ void CoshBackward(float* dx, const float* dy, const float* x,
                           unsigned int totalSize);

__host__ void SinhBackward(float* dx, const float* dy, const float* x,
                           unsigned int totalSize);

__host__ void TanhBackward(float* dx, const float* dy, const float* x,
                           unsigned int totalSize);

__host__ void ArcCosBackward(float* dx, const float* dy, const float* x,
                             unsigned int totalSize);

__host__ void ArcSinBackward(float* dx, const float* dy, const float* x,
                             unsigned int totalSize);

__host__ void ArcTanBackward(float* dx, const float* dy, const float* x,
                             unsigned int totalSize);

__host__ void ArcCoshBackward(float* dx, const float* dy, const float* x,
                              unsigned int totalSize);

__host__ void ArcSinhBackward(float* dx, const float* dy, const float* x,
                              unsigned int totalSize);

__host__ void ArcTanhBackward(float* dx, const float* dy, const float* x,
                              unsigned int totalSize);
}
#endif
