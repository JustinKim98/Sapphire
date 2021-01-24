// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any unsigned intellectual
// property of any third parties.

#include <Motutapu/compute/cuda/dense/GemmKernel.cuh>
#include <mma.h>

#define WARP_SIZE 32

namespace Motutapu::Compute::Cuda::Dense
{
using namespace nvcuda;

__global__ void WmmaGemmHalf(half* Out, const half* A, const half* B,
                             const half* C, unsigned int paddedK,
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
    __shared__ half
        sharedMemory[4 * tileDim * 3][shiftedSharedMemoryColSize];

    //! Each block identifies its chunk to compute using 2 dimensional
    const unsigned int chunkIdxM = blockIdx.x;
    const unsigned int chunkIdxN = blockIdx.y;

    const unsigned int warpIdx = threadIdx.x / WARP_SIZE;
    const unsigned int laneIdx = threadIdx.x % WARP_SIZE;

    const unsigned int tileRowIdx = warpIdx / 4;
    const unsigned int tileColIdx = warpIdx % 4;

    //! Pointer that indicates starting address of chunk
    const half* chunkPtrA = A + paddedK * chunkIdxM * tileDim * chunkSize +
                            chunkIdxK * tileDim * chunkSize;
    const half* chunkPtrB = B + paddedN * chunkIdxK * tileDim * chunkSize +
                            chunkIdxN * tileDim * chunkSize;
    const half* chunkPtrC = C + paddedN * chunkIdxM * tileDim * chunkSize +
                            chunkIdxN * tileDim * chunkSize;

    half* chunkPtrOut = Out + paddedN * chunkIdxM * tileDim * chunkSize +
                        chunkIdxN * tileDim * chunkSize;

    const half* tilePtrA = chunkPtrA + paddedK * tileRowIdx * tileDim;
    const half* tilePtrB = chunkPtrB + tileColIdx * tileDim;
    const half* tilePtrC =
        chunkPtrC + paddedN * tileRowIdx * tileDim + tileColIdx * tileDim;
    half* tilePtrOut =
        chunkPtrOut + paddedN * tileRowIdx * tileDim + tileColIdx * tileDim;

    const unsigned int matrixBOffset = chunkSize * tileDim;
    const unsigned int matrixCOffset = chunkSize * tileDim * 2;

    //! For half of the warps, copy matrix A while other half copies B
    const half* copyPtr;

    unsigned int sharedMemCopyRowIdx;
    if (laneIdx % 2)
    {
        copyPtr = tilePtrA + paddedK * (laneIdx / 2);
        sharedMemCopyRowIdx = tileRowIdx + laneIdx / 2;

        const half* copyPtrC = tilePtrC + paddedN * (laneIdx / 2);
        unsigned int sharedMemCopyRowIdxC =
            matrixCOffset + tileRowIdx + laneIdx / 2;

#pragma unroll
        for (unsigned int i = 0; i < tileDim; i++)
        {
            const unsigned int sharedMemCopyColIdx = tileColIdx * tileDim + i;
            sharedMemory[sharedMemCopyRowIdxC][sharedMemCopyColIdx] =
                *(copyPtrC + i);
        }
    }
    else
    {
        copyPtr = tilePtrB + paddedN * (laneIdx / 2);
        sharedMemCopyRowIdx = matrixBOffset + tileRowIdx + laneIdx / 2;
    }

    //! Load the matrix to shared memory
    //! each thread copies consecutive row from their src determined previously
#pragma unroll
    for (unsigned int i = 0; i < tileDim; i++)
    {
        const unsigned int sharedMemCopyColIdx = tileColIdx * tileDim + i;
        sharedMemory[sharedMemCopyRowIdx][sharedMemCopyColIdx + i] =
            *(copyPtr + i);
    }

    //! Load shared memory to fragments and accumulate
    wmma::fragment<wmma::matrix_a, tileDim, tileDim, tileDim, half,
                   wmma::row_major>
        fragA;
    wmma::fragment<wmma::matrix_b, tileDim, tileDim, tileDim, half,
                   wmma::row_major>
        fragB;

    wmma::fragment<wmma::accumulator, tileDim, tileDim, tileDim, half> fragAcc;

    wmma::fragment<wmma::accumulator, tileDim, tileDim, tileDim, half> fragOut;

    wmma::load_matrix_sync(fragAcc, tilePtrOut, paddedN, wmma::mem_row_major);

#pragma unroll
    for (unsigned int i = 0; i < chunkSize; ++i)
    {
        wmma::load_matrix_sync(fragA,
                               &sharedMemory[tileDim * tileRowIdx][tileDim * i],
                               shiftedSharedMemoryColSize);
        wmma::load_matrix_sync(
            fragB,
            &sharedMemory[tileDim * i + matrixBOffset][tileDim * tileColIdx],
            shiftedSharedMemoryColSize);
        wmma::load_matrix_sync(
            fragAcc,
            &sharedMemory[tileDim * i + matrixCOffset][tileDim * tileColIdx],
            shiftedSharedMemoryColSize, wmma::mem_row_major);
        wmma::mma_sync(fragOut, fragA, fragB, fragAcc);
    }

    wmma::store_matrix_sync(tilePtrOut, fragOut, paddedN, wmma::mem_row_major);
}

__global__ void GemmFloat(float* out, const float* A, const float* B,
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

__global__ void GemmHalf(half* out, const half* A, const half* B,
                          const half* C, unsigned int paddedK,
                          unsigned int paddedN, unsigned int chunkIdxK)
{
    constexpr unsigned int tileDim = 8;
    constexpr unsigned int chunkSize = 4;
    __shared__ half matrixA[tileDim * chunkSize][tileDim * chunkSize + 1];
    __shared__ half matrixB[tileDim * chunkSize][tileDim * chunkSize + 1];
    __shared__ half matrixC[tileDim * chunkSize][tileDim * chunkSize + 1];

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

    const half* chunkPtrA = A + blockIdxA;
    const half* chunkPtrB = B + blockIdxB;
    const half* chunkPtrC = C + blockIdxC;

    half* chunkPtrOut = out + blockIdxOut;

    matrixA[rowIdx][colIdx] = *(chunkPtrA + paddedK * rowIdx + colIdx);
    matrixB[rowIdx][colIdx] = *(chunkPtrB + paddedN * rowIdx + colIdx);
    matrixC[rowIdx][colIdx] = *(chunkPtrC + paddedN * rowIdx + colIdx);

    half output = 0.0f;

#pragma unroll
    for (unsigned int i = 0; i < tileDim * chunkSize; ++i)
    {
        output =
            matrixC[rowIdx][colIdx] + matrixA[rowIdx][i] * matrixB[i][colIdx];
    }

    *(chunkPtrOut + paddedK * rowIdx + colIdx) = output;
}
} // namespace Motutapu::Cuda::Dense
