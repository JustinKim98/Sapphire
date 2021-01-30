// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any unsigned intellectual
// property of any third parties.

#include <Motutapu/compute/cuda/dense/GemmKernel.cuh>
#include <mma.h>
#include <cuda_fp16.h>

#define WARP_SIZE 32

namespace Motutapu::Compute::Cuda::Dense
{
using namespace nvcuda;

__global__ void WmmaGemm(float* Out, const float* A, const float* B,
                         const float* C, unsigned int paddedK,
                         unsigned int paddedN,
                         unsigned int chunkIdxK,
                         const unsigned int chunkSize)
{
    constexpr unsigned int tileDim = 16;

    // Minimum shift we can use with 256bit alignment while protecting from bank
    // conflicts;
    constexpr unsigned shift = 32 / sizeof(half);
    constexpr unsigned shiftedSharedMemoryColSize = 4 * tileDim + shift;

    //! If chunk size is 4, and tile dimension is fixed to 16
    //! This makes each block hold 64 x 64 submatrix (chunk)
    __shared__ half sharedMemoryA[4 * tileDim][shiftedSharedMemoryColSize];
    __shared__ half sharedMemoryB[4 * tileDim][shiftedSharedMemoryColSize];
    __shared__ float sharedMemoryC[4 * tileDim][shiftedSharedMemoryColSize];

    //! Each block identifies its chunk to compute using 2 dimensional
    const unsigned int chunkIdxM = blockIdx.x;
    const unsigned int chunkIdxN = blockIdx.y;

    const unsigned int warpIdx = threadIdx.x / WARP_SIZE;
    const unsigned int laneIdx = threadIdx.x % WARP_SIZE;

    const unsigned int tileRowIdx = warpIdx / 4;
    const unsigned int tileColIdx = warpIdx % 4;

    //! Pointer that indicates starting address of chunk
    const float* chunkPtrA = A + paddedK * chunkIdxM * tileDim * chunkSize +
                             chunkIdxK * tileDim * chunkSize;
    const float* chunkPtrB = B + paddedN * chunkIdxK * tileDim * chunkSize +
                             chunkIdxN * tileDim * chunkSize;
    const float* chunkPtrC = C + paddedN * chunkIdxM * tileDim * chunkSize +
                             chunkIdxN * tileDim * chunkSize;

    float* chunkPtrOut = Out + paddedN * chunkIdxM * tileDim * chunkSize +
                         chunkIdxN * tileDim * chunkSize;

    const float* tilePtrA = chunkPtrA + paddedK * tileRowIdx * tileDim;
    const float* tilePtrB = chunkPtrB + tileColIdx * tileDim;
    const float* tilePtrC =
        chunkPtrC + paddedN * tileRowIdx * tileDim + tileColIdx * tileDim;
    float* tilePtrOut =
        chunkPtrOut + paddedN * tileRowIdx * tileDim + tileColIdx * tileDim;

    //! Load matrix onto shared memory
    //! For half of the warps, copy matrix A while other half copies B
    const float* copyPtr;
    unsigned int sharedMemCopyRowIdx;
    if (laneIdx % 2)
    {
        copyPtr = tilePtrA + paddedK * (laneIdx / 2);
        sharedMemCopyRowIdx = tileRowIdx + laneIdx / 2;

        const float* copyPtrC = tilePtrC + paddedN * (laneIdx / 2);
        const unsigned int sharedMemCopyRowIdxC = tileRowIdx + laneIdx / 2;

#pragma unroll
        for (unsigned int i = 0; i < tileDim; i++)
        {
            const unsigned int sharedMemCopyColIdx = tileColIdx * tileDim + i;
            sharedMemoryC[sharedMemCopyRowIdxC][sharedMemCopyColIdx] =
                *(copyPtrC + i);
        }

#pragma unroll
        for (unsigned int i = 0; i < tileDim; i++)
        {
            const unsigned int sharedMemCopyColIdx = tileColIdx * tileDim + i;
            sharedMemoryA[sharedMemCopyRowIdx][sharedMemCopyColIdx + i] =
                __float2half(*(copyPtr + i));
        }
    }
    else
    {
        copyPtr = tilePtrB + paddedN * (laneIdx / 2);
        sharedMemCopyRowIdx = tileRowIdx + laneIdx / 2;

#pragma unroll
        for (unsigned int i = 0; i < tileDim; i++)
        {
            const unsigned int sharedMemCopyColIdx = tileColIdx * tileDim + i;
            sharedMemoryB[sharedMemCopyRowIdx][sharedMemCopyColIdx + i] =
                __float2half(*(copyPtr + i));
        }
    }

    //! Load the matrix to shared memory
    //! each thread copies consecutive row from their src determined previously

    //! Load shared memory to fragments and accumulate
    wmma::fragment<wmma::matrix_a, tileDim, tileDim, tileDim, half,
                   wmma::row_major>
        fragA;
    wmma::fragment<wmma::matrix_b, tileDim, tileDim, tileDim, half,
                   wmma::row_major>
        fragB;

    wmma::fragment<wmma::accumulator, tileDim, tileDim, tileDim, float> fragAcc;

    wmma::fragment<wmma::accumulator, tileDim, tileDim, tileDim, float> fragOut;

    wmma::load_matrix_sync(fragAcc, tilePtrOut, paddedN, wmma::mem_row_major);

#pragma unroll
    for (unsigned int i = 0; i < chunkSize; ++i)
    {
        wmma::load_matrix_sync(fragA,
                               &sharedMemoryA[tileDim * tileRowIdx][tileDim * i
                               ],
                               shiftedSharedMemoryColSize);
        wmma::load_matrix_sync(
            fragB,
            &sharedMemoryB[tileDim * i][tileDim * tileColIdx],
            shiftedSharedMemoryColSize);
        wmma::load_matrix_sync(
            fragAcc,
            &sharedMemoryC[tileDim * i][tileDim * tileColIdx],
            shiftedSharedMemoryColSize, wmma::mem_row_major);
        wmma::mma_sync(fragOut, fragA, fragB, fragAcc);
    }

    wmma::store_matrix_sync(tilePtrOut, fragOut, paddedN, wmma::mem_row_major);
}

__global__ void Gemm(float* out, const float* A, const float* B,
                     const float* C, unsigned int paddedK,
                     unsigned int paddedN, unsigned int chunkIdxK)
{
    constexpr unsigned int tileDim = 8;
    constexpr unsigned int chunkSize = 4;
    __shared__ float matrixA[tileDim * chunkSize][tileDim * chunkSize + 1];
    __shared__ float matrixB[tileDim * chunkSize][tileDim * chunkSize + 1];
    __shared__ float matrixC[tileDim * chunkSize][tileDim * chunkSize + 1];

    const unsigned int chunkIdxM = blockIdx.x;
    const unsigned int chunkIdxN = blockIdx.y;

    const unsigned int rowIdx = threadIdx.x;
    const unsigned int colIdx = threadIdx.y;

    const unsigned int blockIdxA = chunkIdxM * paddedK * chunkSize * tileDim +
                                   chunkIdxK * chunkSize * tileDim;
    const unsigned int blockIdxB = chunkIdxK * paddedN * chunkSize * tileDim +
                                   chunkIdxN * chunkSize * tileDim;
    const unsigned int blockIdxC = chunkIdxM * paddedK * chunkSize * tileDim +
                                   chunkIdxN * chunkSize * tileDim;
    const unsigned int blockIdxOut = chunkIdxM * paddedK * chunkSize * tileDim +
                                     chunkIdxN * chunkSize * tileDim;

    const float* chunkPtrA = A + blockIdxA;
    const float* chunkPtrB = B + blockIdxB;
    const float* chunkPtrC = C + blockIdxC;

    float* chunkPtrOut = out + blockIdxOut;

    matrixA[rowIdx][colIdx] = *(chunkPtrA + paddedK * rowIdx + colIdx);
    matrixB[rowIdx][colIdx] = *(chunkPtrB + paddedN * rowIdx + colIdx);
    matrixC[rowIdx][colIdx] = *(chunkPtrC + paddedN * rowIdx + colIdx);

    float output = 0.0f;

#pragma unroll
    for (unsigned int i = 0; i < tileDim * chunkSize; ++i)
    {
        output =
            matrixC[rowIdx][colIdx] + matrixA[rowIdx][i] * matrixB[i][colIdx];
    }

    *(chunkPtrOut + paddedK * rowIdx + colIdx) = output;
}
} // namespace Motutapu::Cuda::Dense
