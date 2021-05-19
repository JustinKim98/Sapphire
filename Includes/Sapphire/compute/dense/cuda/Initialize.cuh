// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any unsigned
// intellectual property of any third parties.

#ifndef Sapphire_COMPUTE_CUDA_DENSE_INITIALIZE_CUH
#define Sapphire_COMPUTE_CUDA_DENSE_INITIALIZE_CUH

#include <Sapphire/compute/cudaUtil/CudaParams.cuh>
#include <Sapphire/compute/dense/cuda/InitializeKernel.cuh>

namespace Sapphire::Compute::Dense::Cuda
{
__host__ void Normal(float* data, float mean, float sd, unsigned int size,
                     int seed);

__host__ void Uniform(float* data, float min, float max, unsigned int size,
                      int seed);

__host__ void Scalar(float* data, float value, unsigned int size);
}  // namespace Sapphire::Compute::Cuda::Dense

#endif
