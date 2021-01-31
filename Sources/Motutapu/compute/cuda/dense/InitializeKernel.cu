// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/cuda/CudaParams.cuh>
#include <Motutapu/compute/cuda/dense/InitializeKernel.cuh>

namespace Motutapu::Compute::Cuda::Dense
{
__global__ void NormalKernel(float* data, float mean, float sd,
                             unsigned int size, int seed)
{
    const unsigned int id = threadIdx.x;

    const auto numLoopPerThread =
        blockDim.x == 0 ? size / blockDim.x : size / blockDim.x + 1;

    curandState localState;
    curand_init(1234, id, 0, &localState);

    for (unsigned int i = id * numLoopPerThread; i < size; i++)
    {
        data[i] = (curand_normal(&localState) - mean) / sd;
    }
}

__global__ void UniformKernel(float* data, float min, float max,
                              unsigned int size, int seed)
{
    const unsigned int id = threadIdx.x;

    const auto numLoopPerThread =
        blockDim.x == 0 ? size / blockDim.x : size / blockDim.x + 1;

    curandState localState;
    curand_init(1234, id, 0, &localState);

    for (unsigned int i = id * numLoopPerThread; i < size; i++)
    {
        data[i] = (curand_uniform(&localState) * (max - min) + min);
    }
}

__global__ void ScalarKernel(float* data, float value, unsigned int size)
{
    const unsigned int id = threadIdx.x;

    const auto numLoopPerThread =
        blockDim.x == 0 ? size / blockDim.x : size / blockDim.x + 1;

    for (unsigned int i = id * numLoopPerThread; i < size; i++)
    {
        data[i] = value;
    }
}

}  // namespace Motutapu::Compute::Cuda::Dense
