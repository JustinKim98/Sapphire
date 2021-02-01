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
    const auto numThreads = (size < MAX_THREAD_DIM_X) ? size : MAX_THREAD_DIM_X;

    NormalKernel<<<1, numThreads>>>(data, mean, sd, size, seed);
}

__host__ void Uniform(float* data, float min, float max, unsigned int size,
                      int seed)
{
    const auto numThreads = (size < MAX_THREAD_DIM_X) ? size : MAX_THREAD_DIM_X;

    UniformKernel<<<1, numThreads>>>(data, min, max, size, seed);
}

__host__ void Scalar(float* data, float value, unsigned int size, int seed)
{
    const auto numThreads = (size < MAX_THREAD_DIM_X) ? size : MAX_THREAD_DIM_X;

    ScalarKernel<<<1, numThreads>>>(data, value, size);
}
}  // namespace Motutapu::Compute::Cuda::Dense