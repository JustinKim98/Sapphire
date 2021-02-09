// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/cuda/dense/BasicKernel.cuh>

namespace Motutapu::Compute::Cuda::Dense
{
//! (x,y) : (TILE_DIM*8) threads per block
//! Assuming input is M x N, (nx, ny, nz) : (N/TILE_DIM, M/TILE_DIM, batchSize)
//! blocks required
__global__ void TransposeKernel(float* output, const float* input,
                                unsigned int inputNumRows,
                                unsigned int inputNumCols, bool broadcastInput)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    const unsigned int outputNumRows = inputNumCols;
    const unsigned int outputNumCols = inputNumRows;

    const int inputColIdx = blockIdx.x * TILE_DIM + threadIdx.x;
    const int inputRowIdx = blockIdx.y * TILE_DIM + threadIdx.y;

    const int outputColIdx = blockIdx.y * TILE_DIM + threadIdx.x;
    const int outputRowIdx = blockIdx.x * TILE_DIM + threadIdx.y;

    float* outputOffset = output + inputNumRows * inputNumCols * blockIdx.z;
    const float* inputOffset =
        input + (broadcastInput ? 0 : inputNumRows * inputNumCols * blockIdx.z);

    if (inputColIdx < inputNumCols && outputColIdx < outputNumCols)
    {
        for (int i = 0; (i < TILE_DIM) && (inputRowIdx * i < inputNumRows);
             i += 8)
            tile[threadIdx.y + i][threadIdx.x] =
                inputOffset[(inputRowIdx + i) * inputNumCols + inputColIdx];

        __syncthreads();

        for (int i = 0; (i < TILE_DIM) && (outputRowIdx * i < outputNumRows);
             i += 8)
            outputOffset[(outputRowIdx + i) * outputNumCols + outputColIdx] =
                tile[threadIdx.x][threadIdx.y + i];
    }
}

__global__ void AddKernel(float* output, const float* inputA,
                          const float* inputB, unsigned int offset,
                          unsigned int launchSize, unsigned int totalSize,
                          unsigned int inputStride, bool broadcastInputA,
                          bool broadcastInputB)
{
    const auto sizePerBlock = launchSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    unsigned int leftOverA = broadcastInputA ? inputStride : totalSize;
    unsigned int leftOverB = broadcastInputB ? inputStride : totalSize;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        output[offset + blockOffset + blockDim.x * i + threadIdx.x] =
            inputA[(offset + blockOffset + blockDim.x * i + threadIdx.x) %
                   leftOverA] +
            inputB[(offset + blockOffset + blockDim.x * i + threadIdx.x) %
                   leftOverB];
    }
}

__global__ void SubKernel(float* output, const float* inputA,
                          const float* inputB, unsigned int totalSize,
                          unsigned int inputStride, bool broadcastInputA,
                          bool broadcastInputB)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    unsigned int leftOverA = broadcastInputA ? inputStride : totalSize;
    unsigned int leftOverB = broadcastInputB ? inputStride : totalSize;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        output[blockOffset + blockDim.x * i + threadIdx.x] =
            inputA[(blockOffset + blockDim.x * i + threadIdx.x) % leftOverA] -
            inputB[(blockOffset + blockDim.x * i + threadIdx.x) % leftOverB];
    }
}

__global__ void DotKernel(float* output, const float* inputA,
                          const float* inputB, unsigned int totalSize,
                          unsigned int inputStride, bool broadcastInputA,
                          bool broadcastInputB)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    unsigned int leftOverA = broadcastInputA ? inputStride : totalSize;
    unsigned int leftOverB = broadcastInputB ? inputStride : totalSize;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        output[blockOffset + blockDim.x * i + threadIdx.x] =
            inputA[(blockOffset + blockDim.x * i + threadIdx.x) % leftOverA] *
            inputB[(blockOffset + blockDim.x * i + threadIdx.x) % leftOverB];
    }
}

__global__ void ScaleKernel(float* output, const float* input,
                            const float scaleFactor, unsigned int totalSize,
                            unsigned int inputStride, bool broadcastInput)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    unsigned int leftOver = broadcastInput ? inputStride : totalSize;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        output[blockOffset + blockDim.x * i + threadIdx.x] =
            input[(blockOffset + blockDim.x * i + threadIdx.x) % leftOver] *
            scaleFactor;
    }
}

}  // namespace Motutapu::Compute::Cuda::Dense