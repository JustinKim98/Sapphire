// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any unsigned
// intellectual property of any third parties.

#include <cuda_fp16.h>
#include <mma.h>
#include <Sapphire/compute/dense/cuda/GemmKernel.cuh>

#define WARP_SIZE 32

namespace Sapphire::Compute::Cuda::Dense
{

__global__ void Gemm(float* out, const float* A, const float* B, const float* C,
                     unsigned int paddedM, unsigned int paddedN,
                     unsigned int paddedK, unsigned int chunkIdxK,
                     const unsigned int chunkSize, const unsigned int chunkDimN)
{
    extern __shared__ float sharedMem[];
    const unsigned int tileDim = 8;

    const unsigned int sharedMemColSize = tileDim * chunkSize + 1;
    const unsigned int matrixBOffset = (tileDim * chunkSize) * sharedMemColSize;

    const unsigned int chunkIdxM = blockIdx.x / chunkDimN;
    const unsigned int chunkIdxN = blockIdx.x % chunkDimN;

    const unsigned int rowIdx = threadIdx.x / (chunkSize * tileDim);
    const unsigned int colIdx = threadIdx.x % (chunkSize * tileDim);

    const unsigned int blockIdxA = paddedM * chunkSize * tileDim * chunkIdxM +
                                   chunkSize * tileDim * chunkIdxK;
    const unsigned int blockIdxB = paddedK * chunkSize * tileDim * chunkIdxN +
                                   chunkSize * tileDim * chunkSize;
    const unsigned int blockIdxC = paddedM * chunkSize * tileDim * chunkIdxM +
                                   chunkSize * tileDim * chunkIdxN;
    const unsigned int blockIdxOut = paddedM * chunkSize * tileDim * chunkIdxM +
                                     chunkSize * tileDim * chunkIdxN;

    const float* chunkPtrA = A + blockIdxA;
    const float* chunkPtrB = B + blockIdxB;
    const float* chunkPtrC = C + blockIdxC;
    float* chunkPtrOut = out + blockIdxOut;

    sharedMem[rowIdx * sharedMemColSize + colIdx] =
        *(chunkPtrA + paddedM * rowIdx + colIdx);
    sharedMem[matrixBOffset + rowIdx * sharedMemColSize + colIdx] =
        *(chunkPtrB + paddedK * rowIdx + colIdx);

    __syncthreads();

    float output = *(chunkPtrC + paddedM * rowIdx + colIdx);

    for (unsigned int i = 0; i < tileDim * chunkSize; ++i)
    {
        output += sharedMem[rowIdx * sharedMemColSize + i] *
                  sharedMem[matrixBOffset + i * sharedMemColSize + colIdx];
    }

    __syncthreads();

    *(chunkPtrOut + paddedM * rowIdx + colIdx) = output;
}

__global__ void GemmSimple(float* out, const float* A, const float* B,
                           const float* C, unsigned int paddedM,
                           unsigned int paddedN, unsigned int paddedK)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned int m = idx / paddedN;
    const unsigned int n = idx % paddedN;

    if (m < paddedM)
    {
        out[paddedN * m + n] = C[paddedN * m + n];

        for (unsigned int k = 0; k < paddedK; k++)
        {
            out[paddedN * m + n] += A[paddedM * m + k] * B[paddedK * k + n];
        }
    }
}
}  // namespace Sapphire::Compute::Cuda::Dense
