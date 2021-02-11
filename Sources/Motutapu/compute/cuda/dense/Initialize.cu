// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/cuda/dense/Initialize.cuh>

namespace Motutapu::Compute::Cuda::Dense
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

    cudaDeviceSynchronize();
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

    cudaDeviceSynchronize();
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

    cudaDeviceSynchronize();
}
}  // namespace Motutapu::Compute::Cuda::Dense