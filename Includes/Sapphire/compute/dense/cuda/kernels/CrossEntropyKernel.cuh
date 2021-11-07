// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/cudaUtil/CudaParams.cuh>

namespace Sapphire::Compute::Dense::Cuda
{
__global__ void CrossEntropyKernel(float* y, const float* x, const float* label,
                                   int batchSize, int unitSize);

__global__ void CrossEntropyBackwardKernel(float* dx, const float* label,
                                           int batchSize, int unitSize);
}  // namespace Sapphire::Compute::Dense::Cuda
