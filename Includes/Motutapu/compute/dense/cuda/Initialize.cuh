// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any unsigned
// intellectual property of any third parties.

#ifndef MOTUTAPU_COMPUTE_CUDA_DENSE_INITIALIZE_CUH
#define MOTUTAPU_COMPUTE_CUDA_DENSE_INITIALIZE_CUH

#include <Motutapu/compute/cudaUtil/CudaParams.cuh>
#include <Motutapu/compute/dense/cuda/InitializeKernel.cuh>

namespace Motutapu::Compute::Cuda::Dense
{
__host__ void Normal(float* data, float mean, float sd, unsigned int size,
                     int seed);

__host__ void Uniform(float* data, float min, float max, unsigned int size,
                      int seed);

__host__ void Scalar(float* data, float value, unsigned int size);
}  // namespace Motutapu::Compute::Cuda::Dense

#endif
