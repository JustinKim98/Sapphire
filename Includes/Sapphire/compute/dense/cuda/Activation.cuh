// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_COMPUTE_DENSE_CUDA_ACTAVATION_CUH
#define SAPPHIRE_COMPUTE_DENSE_CUDA_ACTAVATION_CUH

#include <Sapphire/compute/cudaUtil/CudaParams.cuh>

namespace Sapphire::Compute::Dense::Cuda
{
//! Forward Kernels
__host__ void ReLU(float* y, const float* x, unsigned int totalSize);

__host__ void LeakyReLU(float* y, const float* x, float a,
                        unsigned int totalSize);

__host__ void SoftMax(float* y, const float* x, unsigned int totalSize,
                      unsigned int unitSize);

//! Backward Kernels
__host__ void ReLUBackward(float* dx, const float* dy, const float* x,
                           unsigned int totalSize);

__host__ void LeakyReLUBackward(float* dx, const float* dy, const float* x,
                                float a, unsigned int totalSize);

__host__ void SoftmaxBackward(float* dx, const float* dy, const float* y,
                              unsigned int totalSize, unsigned int unitSize);
}

#endif
