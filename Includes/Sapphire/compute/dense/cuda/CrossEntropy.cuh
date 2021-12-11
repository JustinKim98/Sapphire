// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_COMPUTE_DENSE_CUDA_CROSS_ENTROPY_CUH
#define SAPPHIRE_COMPUTE_DENSE_CUDA_CROSS_ENTROPY_CUH

#include <Sapphire/compute/cudaUtil/CudaParams.cuh>

namespace Sapphire::Compute::Dense::Cuda
{
__host__ void CrossEntropy(float* y, const float* x, const float* label,
                           int batchSize, int unitSize);

__host__ void CrossEntropyBackward(float* dx, const float* x,
                                   const float* label,
                                   int batchSize, int unitSize);
}

#endif
