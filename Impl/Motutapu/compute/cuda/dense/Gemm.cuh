// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any size_tellectual
// property of any third parties.

#ifndef MOTUTAPU_CUDA_DENSEMATMUL_HPP
#define MOTUTAPU_CUDA_DENSEMATMUL_HPP

#include <Motutapu/compute/cuda/CudaParams.hpp>
#include <cuda_fp16.h>
#define __CUDACC__
#include <mma.h>
#define WARP_SIZE 32


namespace Motutapu::Cuda::Dense
{
using namespace nvcuda;

//! Computes Gemm operation on the chunk (Out = A x B + Out)
//! Chunk represents sub matrix that can be computed using one block
//! Each chunk is composed of tiles which contains 16x16 elements each
//! Size of the chunk is configurable by chunkSize template parameter
//! For example, if chunk size is 4, each block will compute (4x4) x (16x16) = 64x64
//! chunks each.
//!
//! However, chunk size should be carefully configured considering these constraints
//!     A. size of the shared memory on each SM.
//!         2 x (chunkSize x chunkSize) x (16x16) x sizeof(half) should be smaller than shared memory size
//!         since shared memory is shifted to prevent bank conflicts
//!     B. Memory alignment on shared memory
//!         Each row in shared memory is aligned to 256 bit (32 bytes).
//!         meaning, 16 x chunkSize x sizeof(half) should be multiple of 32 bytes
//!     Otherwise, static assertion will fail
//!
//! Warp size and block Size should be allocated considering chunk size
//! This kernel requires chunkSize x chunkSize warps in block.x dimension
//! meaning, we need to allocate chunkSize x chunkSize x 32(warp size) threads.
//! (y and z dimension is not used for block)
//! Grid x and y dimension depends on chunks in direction of M and N
//! It should be allocated as (x, y) = (M/chunkSize, N/chunkSize)
//!
//! \tparam chunkSize : size of the chunk
//! \param Out : output matrix pointer containing whole matrix
//! \param A : matrix A
//! \param B : matrix B
//! \param paddedK : padded column size of A
//! \param paddedN : padded column size of B
//! \param chunkIdxK : index of K
template <size_t chunkSize>
__global__ void WmmaGemmHalf(half* Out, const half* A, const half* B,
                             const half* C,
                             size_t paddedK, size_t paddedN,
                             size_t chunkIdxK)
{
    static constexpr size_t tileDim = 16;

    if constexpr (chunkSize * tileDim * sizeof(half) % 32 != 0)
    {
        static_assert(
            false &&
            "chunkSize * tileDim * sizeof(half) should be multiple of 32");
    }

    // Minimum shift we can use with 256bit alignment while protecting from bank
    // conflicts;
    constexpr size_t shift = 32 / sizeof(half);
    constexpr size_t shiftedSharedMemoryColSize = chunkSize * tileDim + shift;

    //! If chunk size is 4, and tile dimension is fixed to 16
    //! This makes each block hold 64 x 64 submatrix (chunk)
    __shared__ half
        sharedMemory[chunkSize * tileDim * 3][shiftedSharedMemoryColSize];

    //! Each block identifies its chunk to compute using 2 dimensional
    const size_t chunkIdxM = blockIdx.x;
    const size_t chunkIdxN = blockIdx.y;

    const size_t warpIdx = threadIdx.x / WARP_SIZE;
    const size_t laneIdx = threadIdx.x % WARP_SIZE;

    const size_t tileRowIdx = warpIdx / 4;
    const size_t tileColIdx = warpIdx % 4;

    //! Pointer that indicates starting address of chunk
    const half* chunkPtrA = A +
                            paddedK * chunkIdxM * tileDim * chunkSize +
                            chunkIdxK * tileDim * chunkSize;
    const half* chunkPtrB = B +
                            paddedN * chunkIdxK * tileDim * chunkSize +
                            chunkIdxN * tileDim * chunkSize;
    const half* chunkPtrC = C + paddedN * chunkIdxM * tileDim * chunkSize +
                            chunkIdxN * tileDim * chunkSize;

    half* chunkPtrOut = Out +
                        paddedN * chunkIdxM * tileDim * chunkSize +
                        chunkIdxN * tileDim * chunkSize;

    const half* tilePtrA = chunkPtrA + paddedK * tileRowIdx * tileDim;
    const half* tilePtrB = chunkPtrB + tileColIdx * tileDim;
    const half* tilePtrC =
        chunkPtrC + paddedN * tileRowIdx * tileDim + tileColIdx * tileDim;
    half* tilePtrOut =
        chunkPtrOut + paddedN * tileRowIdx * tileDim + tileColIdx * tileDim;

    const size_t matrixBOffset = chunkSize * tileDim;
    const size_t matrixCOffset = chunkSize * tileDim * 2;

    //! For half of the warps, copy matrix A while other half copies B
    const half* copyPtr;

    size_t sharedMemCopyRowIdx;
    if (laneIdx % 2)
    {
        copyPtr = tilePtrA + paddedK * (laneIdx / 2);
        sharedMemCopyRowIdx = tileRowIdx + laneIdx / 2;

        const half* copyPtrC = tilePtrC + paddedN * (laneIdx / 2);
        size_t sharedMemCopyRowIdxC = matrixCOffset + tileRowIdx + laneIdx / 2;

#pragma unroll
        for (size_t i = 0; i < tileDim; i++)
        {
            const size_t sharedMemCopyColIdx = tileColIdx * tileDim + i;
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
    for (size_t i = 0; i < tileDim; i++)
    {
        const size_t sharedMemCopyColIdx = tileColIdx * tileDim + i;
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

    wmma::load_matrix_sync(fragAcc, tilePtrOut, paddedN,
                           wmma::mem_row_major);

#pragma unroll
    for (size_t i = 0; i < chunkSize; ++i)
    {
        wmma::load_matrix_sync(
            fragA, &sharedMemory[tileDim * tileRowIdx][tileDim * i],
            shiftedSharedMemoryColSize);
        wmma::load_matrix_sync(
            fragB,
            &sharedMemory[tileDim * i + matrixBOffset][tileDim * tileColIdx],
            shiftedSharedMemoryColSize);
        wmma::load_matrix_sync(
            fragAcc,
            &sharedMemory[tileDim * i + matrixCOffset][tileDim * tileColIdx],
            shiftedSharedMemoryColSize);
        wmma::mma_sync(fragOut, fragA, fragB, fragAcc);
    }

    wmma::store_matrix_sync(tilePtrOut, fragOut, paddedN,
                            wmma::mem_row_major);
}

//! \tparam T : template data type
//! \tparam tileDim : dimension of the tile
//! \tparam chunkSize : size of the chunk should be under 2 for float types
//! Required thread dimension is (tileDim x chunkSize) x (tileDim x chunkSize)
//! each thread computes each output entry in the chunk
//! Required block size is chunkDimM x chunkDimN
template <typename T, size_t tileDim, size_t chunkSize>
__global__ void Gemm(T* out, const T* A, const T* B, const T* C,
                     size_t paddedK, size_t paddedN,
                     size_t chunkIdxK)
{
    __shared__ T matrixA[tileDim * chunkSize][tileDim * chunkSize + 1];
    __shared__ T matrixB[tileDim * chunkSize][tileDim * chunkSize + 1];
    __shared__ T matrixC[tileDim * chunkSize][tileDim * chunkSize + 1];

    const size_t chunkIdxM = blockIdx.x;
    const size_t chunkIdxN = blockIdx.y;

    const size_t rowIdx = threadIdx.x;
    const size_t colIdx = threadIdx.y;

    const size_t blockIdxA =
        chunkIdxM * paddedK * chunkSize * tileDim + chunkIdxK * chunkSize *
        tileDim;
    const size_t blockIdxB =
        chunkIdxK * paddedN * chunkSize * tileDim + chunkIdxN * chunkSize *
        tileDim;
    const size_t blockIdxC =
        chunkIdxM * paddedK * chunkSize * tileDim +
        chunkIdxN * chunkSize * tileDim;
    const size_t blockIdxOut =
        chunkIdxM * paddedK * chunkSize * tileDim + chunkIdxN * chunkSize *
        tileDim;

    const T* chunkPtrA = A + blockIdxA;
    const T* chunkPtrB = B + blockIdxB;
    const T* chunkPtrC = C + blockIdxC;

    T* chunkPtrOut = out + blockIdxOut;

    matrixA[rowIdx][colIdx] = *(chunkPtrA + paddedK * rowIdx + colIdx);
    matrixB[rowIdx][colIdx] = *(chunkPtrB + paddedN * rowIdx + colIdx);
    matrixC[rowIdx][colIdx] = *(chunkPtrC + paddedN * rowIdx + colIdx);

    T output = static_cast<T>(0.0f);

#pragma unroll
    for (size_t i = 0; i < tileDim * chunkSize; ++i)
    {
        output = matrixC[rowIdx][colIdx] + matrixA[rowIdx][i] * matrixB[i][colIdx];
    }

    *(chunkPtrOut + paddedK * rowIdx + colIdx) = output;
}
}

#endif
