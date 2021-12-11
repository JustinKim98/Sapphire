// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/dense/cuda/CrossEntropy.cuh>
#include <Sapphire/compute/dense/cuda/kernels/CrossEntropyKernel.cuh>

namespace Sapphire::Compute::Dense::Cuda
{
__host__ void CrossEntropy(float* y, const float* x, const float* label,
                           int batchSize, int unitSize)
{
    const int blockDim =
        MAX_THREAD_DIM_X / 16 > batchSize ? batchSize : MAX_THREAD_DIM_X / 16;
    const int gridDim = (batchSize % blockDim == 0) ? batchSize / blockDim
                                                 : batchSize / blockDim + 1;
    CrossEntropyKernel<<<gridDim, blockDim>>>(y, x, label, batchSize, unitSize);
}

__host__ void CrossEntropyBackward(float* dx,const float* x,  const float* label, int batchSize,
                                   int unitSize)
{
    const int blockDim =
        MAX_THREAD_DIM_X / 16 > batchSize ? batchSize : MAX_THREAD_DIM_X / 16; 
    const int gridDim = (batchSize % blockDim == 0) ? batchSize / blockDim
                                                 : batchSize / blockDim + 1;
    CrossEntropyBackwardKernel<<<gridDim, blockDim>>>(dx, x, label, batchSize,
                                                      unitSize);
}
}  // namespace Sapphire::Compute::Dense::Cuda
