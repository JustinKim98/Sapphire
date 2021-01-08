// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_CUDA_SPARSEMATRIXCOPY_CUH
#define MOTUTAPU_CUDA_SPARSEMATRIXCOPY_CUH

#include <cstdint>
#include <Motutapu/compute/cuda/CudaParams.hpp>
#include <Motutapu/util/SparseMatrixDecl.hpp>

namespace Motutapu::Cuda::Sparse
{
template <typename T>
__device__ void AllocateSparseMatrix(SparseMatrix<T>* dest)
{
    const auto nnz = dest->NNZ;
    const auto rowArraySize = dest->NumRows + 1;
    dest->ColIndex = static_cast<T*>(malloc(nnz * sizeof(T)));
    dest->RowIndex = static_cast<T*>(malloc(rowArraySize * sizeof(T)));
    dest->V = static_cast<T*>(malloc(nnz * sizeof(T)));
}

template <typename T>
__device__ void FreeSparseMatrix(SparseMatrix<T>* target)
{
    free(static_cast<void*>(target->ColIndex));
    free(static_cast<void*>(target->RowINdex));
    free(static_cast<void*>(target->V));
}

template <typename T>
__device__ void CopySparseMatrix(SparseMatrix<T>* dest, SparseMatrix<T>* src,
                                 uint32_t rowOffset)
{
    const auto numRows = src->NumRows;
    const auto srcRowArray = src->RowIndex;
    auto destRowArray = dest->RowIndex + rowOffset;
    const auto srcColArray = src->ColIndex;
    auto destColArray = dest->ColIndex;
    const auto srcValue = src->V;
    const auto destValue = dest->V;

    const auto totalNumWorkers = gridDim.x * blockDim.x;
    auto index = blockDim.x * blockIdx.x + threadIdx.x;

    while (index < numRows)
    {
        destRowArray[index] = srcRowArray[index];
        auto cur = srcRowArray[index];
        auto to = srcRowArray[index + 1];

        while (cur < to)
        {
            destColArray[cur] = srcColArray[cur];
            destValue[cur] = srcValue[cur];
        }
        index += totalNumWorkers;
    }
}

template <typename T>
__global__ void FreeKernel(SparseMatrix<T>* targetArray,
                           uint32_t batchSize)
{
    const auto totalNumWorkers = gridDim.x * blockDim.x;
    auto index = blockDim.x * blockIdx.x + threadIdx.x;
    while (index < batchSize)

    {
        SparseMatrix<T>* destMatrix = targetArray + index;
        FreeSparseMatrix<T>(destMatrix);

        index += totalNumWorkers;
    }
}

template <typename T>
__global__ void AllocateKernel(SparseMatrix<T>* targetArray, uint32_t batchSize)
{
    const auto totalNumWorkers = gridDim.x * blockDim.x;
    auto index = blockDim.x * blockIdx.x + threadIdx.x;
    while (index < batchSize)
    {
        SparseMatrix<T>* destMatrix = targetArray + index;
        AllocateSparseMatrix<T>(destMatrix);

        index += totalNumWorkers;
    }
}

template <typename T>
__global__ void CopyKernel(SparseMatrix<T>* destArray,
                           SparseMatrix<T>* srcArray, uint32_t batchSize)
{
    const auto totalNumWorkers = gridDim.x * blockDim.x;
    auto index = blockDim.x * blockIdx.x + threadIdx.x;

    while (index < batchSize)
    {
        SparseMatrix<T>* destMatrix = destArray + index;
        SparseMatrix<T>* srcMatrix = srcArray + index;

        uint32_t numCopied = 0;
        if (destMatrix->NumRows > DEFAULT_DIM_X)
        {
            CopySparseMatrix<T>
                <<<destMatrix->NumRows / DEFAULT_DIM_X, DEFAULT_DIM_X>>>(
                    destMatrix, srcMatrix, 0);
            numCopied +=
                (destMatrix->NumRows / DEFAULT_DIM_X) * DEFAULT_DIM_X;
        }

        if (destMatrix->NumRows % DEFAULT_DIM_X > 0)
            CopySparseMatrix<T><<<1, destMatrix->NumRows % DEFAULT_DIM_X>>>(
                destMatrix, srcMatrix, numCopied);

        index += totalNumWorkers;
    }
}

template <typename T>
__host__ void AllocateGpu(SparseMatrix<T>* targetArray,
                          SparseMatrix<T>* hostRefArray, uint32_t size)
{
    cudaMalloc(reinterpret_cast<void**>(&targetArray),
               size * SPARSEMATRIX_PADDED_SIZE);

    cudaMemcpy(reinterpret_cast<void**>(&targetArray),
               reinterpret_cast<void**>(&hostRefArray),
               size * SPARSEMATRIX_PADDED_SIZE, cudaMemcpyHostToDevice);

    cudaStream_t streamPool[size / DEFAULT_DIM_X];

    uint32_t idx = 0;
    uint32_t streamIdx = 0;
    for (; idx < size; idx += DEFAULT_DIM_X, streamIdx++)
    {
        cudaStreamCreate(&streamPool[streamIdx]);
        AllocateKernel<T><<<1, DEFAULT_DIM_X, 0, streamPool[streamIdx]>>>(
            targetArray + idx,
            DEFAULT_DIM_X);
    }

    if (idx > 0)
        idx -= DEFAULT_DIM_X;

    AllocateKernel<T><<<1, (size - idx)>>>(targetArray + idx,
                                           size - idx);

    for (int i = 0; i < size / DEFAULT_DIM_X; i ++)
    {
        cudaStreamSynchronize(streamPool[i]);
        cudaStreamDestroy(streamPool[i]);
    }
}

template <typename T>
__host__ void CopyHostToGpu(SparseMatrix<T>* deviceArray,
                            SparseMatrix<T>* hostArray, uint32_t size)
{
    cudaStream_t streamPool[size];
    for (uint32_t i = 0; i < size; ++i)
    {
        SparseMatrix<T>* curDestPtr = deviceArray + i;
        SparseMatrix<T>* curSrcPtr = deviceArray + i;
        cudaStreamCreate(&streamPool[i]);
        cudaMemcpyAsync(curDestPtr->RowIndex, curSrcPtr->RowIndex,
                        (curSrcPtr->NumRows + 1) * sizeof(uint32_t),
                        &streamPool[i]);
        cudaMemcpyAsync(curDestPtr->ColIndex, curSrcPtr->ColIndex,
                        (curSrcPtr->NNZ + 1) * sizeof(uint32_t),
                        &streamPool[i]);
        cudaMemcpyAsync(curDestPtr->V, curSrcPtr->V,
                        (curSrcPtr->NNZ + 1) * sizeof(T), &streamPool[i]);
    }
}

template <typename T>
__global__ void CopySparseMatrixOnGpu(SparseMatrix<T>* destArray,
                                      SparseMatrix<T>* srcArray,
                                      uint32_t size)
{
    const auto totalNumWorkers = gridDim.x * blockDim.x;
    auto index = blockDim.x * blockIdx.x + threadIdx.x;
    while (index < size)

    {
        SparseMatrix<T>* destMatrix = destArray + index;
        SparseMatrix<T>* srcMatrix = srcArray + index;
        FreeSparseMatrix<T>(destMatrix);

        destMatrix->NNZ = srcMatrix->NNZ;
        destMatrix->NumRows = srcMatrix->NumRows;
        AllocateSparseMatrix<T>(destMatrix);

        uint32_t numCopied = 0;
        if (destMatrix->NumRows > DEFAULT_DIM_X)
        {
            CopySparseMatrix<T>
                <<<destMatrix->NumRows / DEFAULT_DIM_X, DEFAULT_DIM_X>>>(
                    destMatrix, srcMatrix, 0);
            numCopied +=
                (destMatrix->NumRows / DEFAULT_DIM_X) * DEFAULT_DIM_X;
        }

        if (destMatrix->NumRows % DEFAULT_DIM_X > 0)
            CopySparseMatrix<T><<<1, destMatrix->NumRows % DEFAULT_DIM_X>>>(
                destMatrix, srcMatrix, numCopied);

        index += totalNumWorkers;
    }
}
}
#endif
