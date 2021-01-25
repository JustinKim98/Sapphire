// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any unsigned
// intellectual property of any third parties.

#ifndef MOTUTAPU_COMPUTE_CUDA_DENSE_INITIALIZE_CUH
#define MOTUTAPU_COMPUTE_CUDA_DENSE_INITIALIZE_CUH

#include <Motutapu/compute/cuda/dense/InitializeKernel.cuh>
#include <Motutapu/compute/cuda/CudaParams.hpp>


namespace Motutapu::Compute::Cuda::Dense
{
__host__ bool InitRandom(curandState_t* state)
{
    auto successful = true;

    successful &= cudaMalloc(&state, MAX_THREAD_DIM_X) == cudaSuccess;
    initRandomKernel(state);

    return successful;
}

__host__ void NormalFloat(float* data, float mean, float sd, unsigned int size,
                          curandState_t* state)
{
    const auto numThreads = (size < MAX_THREAD_DIM_X) ? size : MAX_THREAD_DIM_X;

    NormalFloatKernel<<<1, numThreads>>>(data, mean, sd, size, state);
}
}

#endif
