// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_CUDA_SPARSEMATRIXMANAGE_CUH
#define MOTUTAPU_CUDA_SPARSEMATRIXMANAGE_CUH

#include <cstdint>
#include <Motutapu/compute/cuda/CudaParams.hpp>
#include <Motutapu/util/SparseMatrixDecl.hpp>
#include <Motutapu/compute/cuda/sparse/SparseMatrixManageDecl.cuh>

namespace Motutapu::Cuda::Sparse
{

template <typename T>
__device__ void DeepAllocateSparseMatrix(SparseMatrix<T>* dest)
{
    const auto nnz = dest->NNZ;
    const auto rowArraySize = dest->NumRows + 1;
    dest->ColIndex = static_cast<T*>(malloc(nnz * sizeof(T)));
    dest->RowIndex = static_cast<T*>(malloc(rowArraySize * sizeof(T)));
    dest->V = static_cast<T*>(malloc(nnz * sizeof(T)));
}

template <typename T>
__device__ void DeepFreeSparseMatrix(SparseMatrix<T>* target)
{
    free(static_cast<void*>(target->ColIndex));
    free(static_cast<void*>(target->RowINdex));
    free(static_cast<void*>(target->V));
}

template <typename T>
__device__ void ShallowFreeSparseMatrix(SparseMatrix<T>* target)
{
    free(static_cast<void*>(target));
}

template <typename T>
__device__ void DeepCopySparseMatrix(SparseMatrix<T>* dest,
                                     SparseMatrix<T>* src, uint32_t rowOffset)
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
__global__ void DeepFreeKernel(SparseMatrix<T>* targetArray, uint32_t size)
{
    const auto totalNumWorkers = gridDim.x * blockDim.x;
    auto index = blockDim.x * blockIdx.x + threadIdx.x;
    while (index < size)

    {
        SparseMatrix<T>* destMatrix = targetArray + index;
        DeepFreeSparseMatrix<T>(destMatrix);

        index += totalNumWorkers;
    }
}

template <typename T>
__global__ void DeepAllocateKernel(SparseMatrix<T>* targetArray, uint32_t size)
{
    const auto totalNumWorkers = gridDim.x * blockDim.x;
    auto index = blockDim.x * blockIdx.x + threadIdx.x;
    while (index < size)
    {
        SparseMatrix<T>* destMatrix = targetArray + index;
        DeepAllocateSparseMatrix<T>(destMatrix);

        index += totalNumWorkers;
    }
}

template <typename T>
__global__ void DeepCopySparseMatrixOnGpu(SparseMatrix<T>* destArray,
                                          SparseMatrix<T>* srcArray,
                                          uint32_t size)
{
    const auto totalNumWorkers = gridDim.x * blockDim.x;
    auto index = blockDim.x * blockIdx.x + threadIdx.x;

    while (index < size)
    {
        SparseMatrix<T>* destMatrix = destArray + index;
        SparseMatrix<T>* srcMatrix = srcArray + index;
        DeepFreeSparseMatrix<T>(destMatrix);

        destMatrix->NNZ = srcMatrix->NNZ;
        destMatrix->NumRows = srcMatrix->NumRows;
        DeepAllocateSparseMatrix<T>(destMatrix);

        uint32_t numCopied = 0;
        if (destMatrix->NumRows > DEFAULT_DIM_X)
        {
            DeepCopySparseMatrix<T>
                <<<destMatrix->NumRows / DEFAULT_DIM_X, DEFAULT_DIM_X>>>(
                    destMatrix, srcMatrix, 0);
            numCopied += (destMatrix->NumRows / DEFAULT_DIM_X) * DEFAULT_DIM_X;
        }

        if (destMatrix->NumRows % DEFAULT_DIM_X > 0)
            DeepCopySparseMatrix<T><<<1, destMatrix->NumRows % DEFAULT_DIM_X>>>(
                destMatrix, srcMatrix, numCopied);

        index += totalNumWorkers;
    }
}

template <typename T>
__host__ void DeepCopyHostToGpu(SparseMatrix<T>* deviceArray,
                                SparseMatrix<T>* hostArray, uint32_t size)
{
    cudaStream_t streamPool[size];
    for (uint32_t i = 0; i < size; ++i)
    {
        SparseMatrix<T>* curDestPtr = deviceArray + i;
        SparseMatrix<T>* curSrcPtr = hostArray + i;
        cudaStreamCreate(&streamPool[i]);
        cudaMemcpyAsync(curDestPtr->RowIndex, curSrcPtr->RowIndex,
                        (curSrcPtr->NumRows + 1) * sizeof(uint32_t),
                        streamPool[i]);
        cudaMemcpyAsync(curDestPtr->ColIndex, curSrcPtr->ColIndex,
                        (curSrcPtr->NNZ) * sizeof(uint32_t), streamPool[i]);
        cudaMemcpyAsync(curDestPtr->V, curSrcPtr->V,
                        (curSrcPtr->NNZ) * sizeof(T), streamPool[i]);
    }

    for (uint32_t i = 0; i < size; ++i)
    {
        cudaStreamSynchronize(streamPool[i]);
        cudaStreamDestroy(streamPool[i]);
    }
}

template <typename T>
__host__ void ShallowAllocateGpu(SparseMatrix<T>* targetArray, uint32_t size)
{
    cudaMalloc(reinterpret_cast<void**>(&targetArray),
               size * SPARSEMATRIX_PADDED_SIZE);
}

template <typename T>
__host__ void DeepAllocateGpu(SparseMatrix<T>* targetArray,
                              SparseMatrix<T>* hostRefArray, uint32_t size)
{
    cudaMemcpy(reinterpret_cast<void**>(&targetArray),
               reinterpret_cast<void**>(&hostRefArray),
               size * SPARSEMATRIX_PADDED_SIZE, cudaMemcpyHostToDevice);

    cudaStream_t streamPool[size / DEFAULT_DIM_X];

    uint32_t idx = 0;
    uint32_t streamIdx = 0;
    for (; idx < size; idx += DEFAULT_DIM_X, streamIdx++)
    {
        cudaStreamCreate(&streamPool[streamIdx]);
        DeepAllocateKernel<T><<<1, DEFAULT_DIM_X, 0, streamPool[streamIdx]>>>(
            targetArray + idx, DEFAULT_DIM_X);
    }

    if (idx > 0)
        idx -= DEFAULT_DIM_X;

    DeepAllocateKernel<T><<<1, size - idx>>>(targetArray + idx, size - idx);

    for (int i = 0; i < size / DEFAULT_DIM_X; i++)
    {
        cudaStreamSynchronize(streamPool[i]);
        cudaStreamDestroy(streamPool[i]);
    }
}
}

#endif