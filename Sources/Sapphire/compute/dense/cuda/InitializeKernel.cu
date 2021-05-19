// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/cudaUtil/CudaParams.cuh>
#include <Sapphire/compute/dense/cuda/InitializeKernel.cuh>

namespace Sapphire::Compute::Dense::Cuda
{
//! ByteSize should be divisible by gridDim.x*blockDim.x
//! Both block and thread should be in one dimension
__global__ void NormalKernel(float* data, float mean, float sd,
                             unsigned int size, int seed)
{
    const auto id = blockDim.x * blockIdx.x + threadIdx.x;
    const auto sizePerBlock = size / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    curandState localState;
    curand_init(seed, id, 0, &localState);

    for (unsigned int i = 0; i < numLoops; i++)
    {
        data[blockOffset + blockDim.x * i + threadIdx.x] =
            (curand_normal(&localState) + mean) / sd;
    }
}

__global__ void UniformKernel(float* data, float min, float max,
                              unsigned int size, int seed)
{
    const auto id = blockDim.x * blockIdx.x + threadIdx.x;
    const auto sizePerBlock = size / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    curandState localState;
    curand_init(seed, id, 0, &localState);

    for (unsigned int i = 0; i < numLoops; i++)
    {
        data[blockOffset + blockDim.x * i + threadIdx.x] =
            (curand_uniform(&localState) * (max - min) + min);
    }
}

__global__ void ScalarKernel(float* data, float value, unsigned int size)
{
    const auto sizePerBlock = size / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        data[blockOffset + blockDim.x * i + threadIdx.x] = value;
    }
}

}  // namespace Sapphire::Compute::Dense::Cuda
