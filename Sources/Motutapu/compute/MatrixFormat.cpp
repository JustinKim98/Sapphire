// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/MatrixFormat.hpp>
#include <Motutapu/compute/cuda/Memory.cuh>
#include <Motutapu/util/MemoryManager.hpp>

namespace Motutapu::Compute
{
void DeepAllocateSparseHost(SparseMatrix* target, size_t m, size_t nnz,
                            size_t numMatrices)
{
    target = new SparseMatrix[numMatrices];
    target->M = m;
    target->NNZ = nnz;
    target->V = static_cast<float*>(
        Util::MemoryManager::GetMemoryHost(nnz * sizeof(float)));
    target->COL = static_cast<size_t*>(
        Util::MemoryManager::GetMemoryHost(nnz * sizeof(size_t)));
    target->ROW = static_cast<size_t*>(
        Util::MemoryManager::GetMemoryHost((m + 1) * sizeof(size_t)));
}

void DeepFreeSparseHost(SparseMatrix* target)
{
    delete[] target;
    Util::MemoryManager::DeReferenceHost(static_cast<void*>(target->V));
    Util::MemoryManager::DeReferenceHost(static_cast<void*>(target->COL));
    Util::MemoryManager::DeReferenceHost(static_cast<void*>(target->ROW));
}

void DeepAllocateSparseCuda(SparseMatrix* cudaTarget, SparseMatrix* hostTarget,
                            size_t numMatrices, int deviceId)
{
    Cuda::CudaMalloc((void**)(&cudaTarget), sizeof(SparseMatrix) * numMatrices);
    Cuda::MemcpyHostToGpu(cudaTarget, hostTarget, sizeof(SparseMatrix));

    cudaTarget->V = static_cast<float*>(Util::MemoryManager::GetMemoryCuda(
        hostTarget->NNZ * sizeof(float), deviceId));
    cudaTarget->COL = static_cast<size_t*>(Util::MemoryManager::GetMemoryCuda(
        hostTarget->NNZ * sizeof(size_t), deviceId));
    cudaTarget->ROW = static_cast<size_t*>(Util::MemoryManager::GetMemoryHost(
        (hostTarget->M + 1) * sizeof(size_t)));
}

void DeepFreeSparseCuda(SparseMatrix* cudaTarget, int deviceId)
{
    Cuda::CudaFree((void*)cudaTarget);
    Util::MemoryManager::DeReferenceCuda(static_cast<void*>(cudaTarget->V),
                                         deviceId);
    Util::MemoryManager::DeReferenceCuda(static_cast<void*>(cudaTarget->COL),
                                         deviceId);
    Util::MemoryManager::DeReferenceCuda(static_cast<void*>(cudaTarget->ROW),
                                         deviceId);
}

void ConvertDenseToSparseHost(SparseMatrix* dst, float* src, size_t numRows,
                              size_t numCols, size_t numMatrices)
{
    const auto matrixSize = numRows * numCols;
    for (size_t matrixIdx = 0; matrixIdx < numMatrices; matrixIdx++)
    {
        size_t nnz = 0;
        dst[matrixIdx].ROW[0] = 0;
        for (size_t rowIdx = 0; rowIdx < numRows; ++rowIdx)
        {
            for (size_t colIdx = 0; colIdx < numCols; ++colIdx)
            {
                if (src[matrixIdx * matrixSize + rowIdx] != 0)
                {
                    dst[matrixIdx].V[nnz] =
                        src[matrixIdx * matrixSize + rowIdx * numRows + colIdx];
                    dst[matrixIdx].COL[nnz] = colIdx;
                    nnz++;
                }
            }
            dst[matrixIdx].NNZ = nnz;
            dst[matrixIdx].M = numRows;
        }
    }
}

void ConvertDenseToSparseCuda(SparseMatrix* dst, float* src, size_t numRows,
                              size_t numCols, size_t numMatrices)
{
    
}

void ConvertSparseToDenseHost(float* dst, SparseMatrix* src, size_t numRows,
                              size_t numCols, size_t numMatrices)
{
    const auto matrixSize = numRows * numCols;
    for (size_t matrixIdx = 0; matrixIdx < numMatrices; ++matrixIdx)
    {
        for (size_t rowIdx = 0; rowIdx < src[matrixIdx].M; ++rowIdx)
        {
            const auto sparseColIdxBegin = src[matrixIdx].ROW[rowIdx];
            const auto sparseColIdxEnd = src[matrixIdx].ROW[rowIdx + 1];
            for (auto sparseColIdx = sparseColIdxBegin;
                 sparseColIdx < sparseColIdxEnd; ++sparseColIdx)
            {
                const auto denseColIdx = src[matrixIdx].ROW[sparseColIdx];
                const auto value = src[matrixIdx].V[sparseColIdx];
                dst[matrixIdx * matrixSize + rowIdx * numCols + denseColIdx] =
                    value;
            }
        }
    }
}

void ConvertSparseToDenseCuda(float* dst, SparseMatrix* src, size_t numRows,
                              size_t numCols, size_t numMatrices)
{
}

}  // namespace Motutapu::Compute