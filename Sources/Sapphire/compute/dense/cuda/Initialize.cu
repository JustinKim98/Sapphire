// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/dense/cuda/Initialize.cuh>
#include <Sapphire/compute/dense/cuda/kernels/InitializeKernel.cuh>

namespace Sapphire::Compute::Dense::Cuda
{
__host__ void Normal(float* data, float mean, float sd, unsigned int size,
                     int seed)
{
    const auto numLoops = 8;
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

    const auto blockDim = size / (threadDim * numLoops);
    const auto firstLaunchSize = blockDim * threadDim * numLoops;

    if (firstLaunchSize > 0)
        NormalKernel<<<blockDim, threadDim>>>(data, mean, sd, firstLaunchSize,
                                              seed);
    if (size > firstLaunchSize)
        NormalKernel<<<1, size - firstLaunchSize>>>(
            data + firstLaunchSize, mean, sd, size - firstLaunchSize, seed);
}

__host__ void Uniform(float* data, float min, float max, unsigned int size,
                      int seed)
{
    const auto numLoops = 8;
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

    const auto blockDim = size / (threadDim * numLoops);
    const auto firstLaunchSize = blockDim * threadDim * numLoops;

    if (firstLaunchSize > 0)
        UniformKernel<<<blockDim, threadDim>>>(data, min, max, firstLaunchSize,
                                               seed);
    if (size > firstLaunchSize)
        UniformKernel<<<1, size - firstLaunchSize>>>(
            data + firstLaunchSize, min, max, size - firstLaunchSize, seed);
}

__host__ void Scalar(float* data, float value, unsigned int size)
{
    const auto numLoops = 8;
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

    const auto blockDim = size / (threadDim * numLoops);
    const auto firstLaunchSize = blockDim * threadDim * numLoops;

    if (firstLaunchSize > 0)
        ScalarKernel<<<blockDim, threadDim>>>(data, value, firstLaunchSize);
    if (size > firstLaunchSize)
        ScalarKernel<<<1, size - firstLaunchSize>>>(
            data + firstLaunchSize, value, size - firstLaunchSize);
}
}  // namespace Sapphire::Compute::Cuda::Dense