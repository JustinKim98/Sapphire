// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/cuda/dense/BasicKernel.cuh>

namespace Motutapu::Compute::Cuda::Dense
{
//__global__ void transposeNaive(float* output, const float* input,
//                               unsigned int inputNumRows,
//                               unsigned int inputNumCols, bool broadcastInput)
//{
//    int x = blockIdx.x * TILE_DIM + threadIdx.x;
//    int y = blockIdx.y * TILE_DIM + threadIdx.y;
//    int width = gridDim.x * TILE_DIM;
//
//    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
//        output[x * width + (y + j)] = input[(y + j) * width + x];
//}

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

    for (int i = 0; (i < TILE_DIM) && (inputRowIdx * i < inputNumRows); i += 8)
    {
        if (inputRowIdx + i < inputNumRows && inputColIdx < inputNumCols)
            tile[threadIdx.y + i][threadIdx.x] =
                inputOffset[(inputRowIdx + i) * inputNumCols + inputColIdx];
    }

    __syncthreads();

    for (int i = 0; (i < TILE_DIM) && (outputRowIdx * i < outputNumRows);
         i += 8)
    {
        if (outputRowIdx + i < outputNumRows && outputColIdx < outputNumCols)
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

__global__ void AddKernelShared(float* output, const float* inputA,
                                const float* inputB, unsigned int offset,
                                unsigned int launchSize, unsigned int totalSize,
                                unsigned int inputStride, unsigned int numLoops,
                                bool broadcastInputA, bool broadcastInputB)
{
    __shared__ extern float temp[];

    const auto sizePerBlock = blockDim.x * numLoops;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    unsigned int leftOverA = broadcastInputA ? inputStride : totalSize;
    unsigned int leftOverB = broadcastInputB ? inputStride : totalSize;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        if (offset + blockOffset + blockDim.x * i + threadIdx.x < totalSize)
        {
            temp[blockDim.x * i + threadIdx.x] =
                inputA[(offset + blockOffset + blockDim.x * i + threadIdx.x) %
                       leftOverA];

            temp[sizePerBlock + blockDim.x * i + threadIdx.x] =
                inputB[(offset + blockOffset + blockDim.x * i + threadIdx.x) %
                       leftOverB];
        }
    }

    __syncthreads();

    for (unsigned int i = 0; i < numLoops; i++)
    {
        if (offset + blockOffset + blockDim.x * i + threadIdx.x < totalSize)
        {
            output[offset + blockOffset + blockDim.x * i + threadIdx.x] =
                temp[blockDim.x * i + threadIdx.x] +
                temp[sizePerBlock + blockDim.x * i + threadIdx.x];
        }
    }
}

__global__ void AddKernelBroadcast(float* output, const float* inputA,
                                   const float* inputB, unsigned int offset,
                                   unsigned int totalSize,
                                   unsigned int inputStride,
                                   bool broadcastInputA, bool broadcastInputB)
{
    __shared__ extern float temp[];
    const auto id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < inputStride)
    {
        if (broadcastInputA)
            temp[id % blockDim.x] = inputA[offset + id];

        if (broadcastInputB)
            temp[blockDim.x + id % blockDim.x] = inputB[offset + id];

        __syncthreads();

        for (int i = 0; i < totalSize; i += inputStride)
        {
            auto aValue = broadcastInputA ? temp[id % blockDim.x]
                                          : inputA[offset + id + i];
            auto bValue = broadcastInputB ? temp[blockDim.x + id % blockDim.x]
                                          : inputB[offset + id + i];
            output[offset + id + i] = aValue + bValue;
        }
    }
}

__global__ void SubKernel(float* output, const float* inputA,
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
                   leftOverA] -
            inputB[(offset + blockOffset + blockDim.x * i + threadIdx.x) %
                   leftOverB];
    }
}

__global__ void DotKernel(float* output, const float* inputA,
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
                   leftOverA] *
            inputB[(offset + blockOffset + blockDim.x * i + threadIdx.x) %
                   leftOverB];
    }
}

__global__ void ScaleKernel(float* output, const float* input,
                            const float scaleFactor, unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        output[blockOffset + blockDim.x * i + threadIdx.x] =
            input[blockOffset + blockDim.x * i + threadIdx.x] * scaleFactor;
    }
}

__global__ void PowKernel(float* output, const float* input, const float factor,
                          unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        output[blockOffset + blockDim.x * i + threadIdx.x] =
            powf(input[blockOffset + blockDim.x * i + threadIdx.x], factor);
    }
}

__global__ void cosKernel(float* output, const float* input,
                          unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        output[blockOffset + blockDim.x * i + threadIdx.x] =
            cosf(input[blockOffset + blockDim.x * i + threadIdx.x]);
    }
}

__global__ void sinKernel(float* output, const float* input,
                          unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        output[blockOffset + blockDim.x * i + threadIdx.x] =
            sinf(input[blockOffset + blockDim.x * i + threadIdx.x]);
    }
}

__global__ void tanKernel(float* output, const float* input,
                          unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        output[blockOffset + blockDim.x * i + threadIdx.x] =
            tanf(input[blockOffset + blockDim.x * i + threadIdx.x]);
    }
}

__global__ void coshKernel(float* output, const float* input,
                           unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        output[blockOffset + blockDim.x * i + threadIdx.x] =
            coshf(input[blockOffset + blockDim.x * i + threadIdx.x]);
    }
}

__global__ void sinhKernel(float* output, const float* input,
                           unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        output[blockOffset + blockDim.x * i + threadIdx.x] =
            sinhf(input[blockOffset + blockDim.x * i + threadIdx.x]);
    }
}

__global__ void tanhKernel(float* output, const float* input,
                           unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        output[blockOffset + blockDim.x * i + threadIdx.x] =
            tanhf(input[blockOffset + blockDim.x * i + threadIdx.x]);
    }
}

__global__ void logKernel(float* output, const float* input,
                          unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        output[blockOffset + blockDim.x * i + threadIdx.x] =
            logf(input[blockOffset + blockDim.x * i + threadIdx.x]);
    }
}

__global__ void log10Kernel(float* output, const float* input,
                            unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        output[blockOffset + blockDim.x * i + threadIdx.x] =
            log10f(input[blockOffset + blockDim.x * i + threadIdx.x]);
    }
}

__global__ void ReLUKernel(float* output, const float* input,
                           unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        output[blockOffset + blockDim.x * i + threadIdx.x] =
            input[blockOffset + blockDim.x * i + threadIdx.x] > 0.0f
                ? input[blockOffset + blockDim.x * i + threadIdx.x]
                : 0.0f;
    }
}

__global__ void ReLUDerivativeKernel(float* output, const float* input,
                                     unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        output[blockOffset + blockDim.x * i + threadIdx.x] =
            input[blockOffset + blockDim.x * i + threadIdx.x] > 0.0f ? 1.0f
                                                                     : 0.0f;
    }
}

__global__ void LeakyReLUKernel(float* output, const float* input, float a,
                                unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        output[blockOffset + blockDim.x * i + threadIdx.x] =
            input[blockOffset + blockDim.x * i + threadIdx.x] > 0
                ? input[blockOffset + blockDim.x * i + threadIdx.x]
                : a * input[blockOffset + blockDim.x * i + threadIdx.x];
    }
}

__global__ void LeakyReLUDerivativeKernel(float* output, const float* input,
                                          const float a, unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        output[blockOffset + blockDim.x * i + threadIdx.x] =
            input[blockOffset + blockDim.x * i + threadIdx.x] > 0.0f ? 1.0f : a;
    }
}

__global__ void InverseKernel(float* output, const float* input,
                              unsigned int totalSize)
{
    const auto sizePerBlock = totalSize / gridDim.x;
    const auto numLoops = sizePerBlock / blockDim.x;
    const auto blockOffset = sizePerBlock * blockIdx.x;

    for (unsigned int i = 0; i < numLoops; i++)
    {
        output[blockOffset + blockDim.x * i + threadIdx.x] =
            1 / input[blockOffset + blockDim.x * i + threadIdx.x];
    }
}

__global__ void MeanKernel(float* output, const float* input,
                           unsigned int totalSize, unsigned int unitSize)
{
    const auto unitId = blockIdx.x * blockDim.x + threadIdx.x;

    if (unitId < totalSize)
    {
        for (unsigned int i = 0; i < unitSize; i++)
        {
            output[unitId] += input[unitSize * unitId + i];
        }
        output[unitId] /= static_cast<float>(unitSize);
    }
}
}  // namespace Motutapu::Compute::Cuda::Dense