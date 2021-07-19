// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/dense/cuda/kernels/BasicKernel.cuh>
#include <Sapphire/compute/cudaUtil/CudaParams.cuh>

namespace Sapphire::Compute::Dense::Cuda
{

__global__ void AddKernel(float* y, const float* a,
                          const float* b, unsigned int offset,
                          unsigned int launchSize, unsigned int totalSize,
                          unsigned int inputStride, bool broadcastInputA,
                          bool broadcastInputB)
{
    const auto sizePerBlock = launchSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    const unsigned int leftOverA = broadcastInputA ? inputStride : totalSize;
    const unsigned int leftOverB = broadcastInputB ? inputStride : totalSize;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        const auto idx = blockOffset + blockDim.x * i + threadIdx.x;
        y[offset + idx] =
            a[(offset + idx) %
              leftOverA] +
            b[(offset + idx) %
              leftOverB];
    }
}

__global__ void SubKernel(float* y, const float* a,
                          const float* b, unsigned int offset,
                          unsigned int launchSize, unsigned int totalSize,
                          unsigned int inputStride, bool broadcastInputA,
                          bool broadcastInputB)
{
    const auto sizePerBlock = launchSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    const unsigned int leftOverA = broadcastInputA ? inputStride : totalSize;
    const unsigned int leftOverB = broadcastInputB ? inputStride : totalSize;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        const auto idx = blockOffset + blockDim.x * i + threadIdx.x;
        y[offset + idx] = a[(offset + idx) % leftOverA] - b[
                              (offset + idx) % leftOverB];
    }
}

__global__ void DotKernel(float* y, const float* a,
                          const float* b, unsigned int offset,
                          unsigned int launchSize, unsigned int totalSize,
                          unsigned int inputStride, bool broadcastInputA,
                          bool broadcastInputB)
{
    const auto sizePerBlock = launchSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    const unsigned int leftOverA = broadcastInputA ? inputStride : totalSize;
    const unsigned int leftOverB = broadcastInputB ? inputStride : totalSize;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        const auto idx = blockOffset + blockDim.x * i + threadIdx.x;
        y[idx] = a[(offset + idx) % leftOverA] * b[(offset + idx) % leftOverB];
    }
}

//! (x,y) : (TILE_DIM*8) threads per block
//! Assuming x is M x N, (nx, ny, nz) : (N/TILE_DIM, M/TILE_DIM, batchSize)
//! blocks required
__global__ void TransposeKernel(float* y, const float* x,
                                unsigned int inputNumRows,
                                unsigned int inputNumCols, bool broadcastInput)
{
    const auto tileDim = 8;
    __shared__ float tile[tileDim][tileDim + 1];

    const unsigned int outputNumRows = inputNumCols;
    const unsigned int outputNumCols = inputNumRows;

    const unsigned int inputColIdx = blockIdx.x * tileDim + threadIdx.x;
    const int unsigned inputRowIdx = blockIdx.y * tileDim + threadIdx.y;

    const unsigned int outputColIdx = blockIdx.y * tileDim + threadIdx.x;
    const unsigned int outputRowIdx = blockIdx.x * tileDim + threadIdx.y;

    float* outputOffset = y + inputNumRows * inputNumCols * blockIdx.z;
    const float* inputOffset =
        x + (broadcastInput ? 0 : inputNumRows * inputNumCols * blockIdx.z);

    for (int i = 0; (i < tileDim) && (inputRowIdx * i < inputNumRows); i += 8)
    {
        if (inputRowIdx + i < inputNumRows && inputColIdx < inputNumCols)
            tile[threadIdx.y + i][threadIdx.x] =
                inputOffset[(inputRowIdx + i) * inputNumCols + inputColIdx];
    }

    __syncthreads();

    for (int i = 0; (i < tileDim) && (outputRowIdx * i < outputNumRows); i += 8)
    {
        if (outputRowIdx + i < outputNumRows && outputColIdx < outputNumCols)
            outputOffset[(outputRowIdx + i) * outputNumCols + outputColIdx] =
                tile[threadIdx.x][threadIdx.y + i];
    }
}

__global__ void ScaleKernel(float* y, const float* x,
                            const float scaleFactor, unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        const auto idx = blockOffset + blockDim.x * i + threadIdx.x;
        y[idx] = x[idx] * scaleFactor;
    }
}

__global__ void PowKernel(float* y, const float* x, const float factor,
                          unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        const auto idx = blockOffset + blockDim.x * i + threadIdx.x;
        y[idx] = powf(x[idx], factor);
    }
}

__global__ void logKernel(float* y, const float* x, unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        const auto idx = blockOffset + blockDim.x * i + threadIdx.x;
        y[idx] = logf(x[idx]);
    }
}

__global__ void log10Kernel(float* y, const float* x, unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        const auto idx = blockOffset + blockDim.x * i + threadIdx.x;
        y[idx] = log10f(x[idx]);
    }
}


__global__ void MeanKernel(float* y, const float* x, unsigned int totalSize,
                           unsigned int unitSize)
{
    if (const auto unitId = blockIdx.x * blockDim.x + threadIdx.x;
        unitId < totalSize)
    {
        for (unsigned int i = 0; i < unitSize; i++)
        {
            y[unitId] += x[unitSize * unitId + i];
        }
        y[unitId] /= static_cast<float>(unitSize);
    }
}

__global__ void InverseKernel(float* y, const float* x,
                              unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        const auto idx = blockOffset + blockDim.x * i + threadIdx.x;
        y[idx] = 1 / x[idx];
    }
}

} // namespace Sapphire::Compute::Cuda::Dense
