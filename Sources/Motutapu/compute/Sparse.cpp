// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/Sparse.hpp>
#include <Motutapu/compute/SparseMatrix.hpp>
#include <Motutapu/compute/cuda/Memory.cuh>
#include <Motutapu/compute/cuda/sparse/CalculateLoad.cuh>
#include <Motutapu/compute/cuda/sparse/MatrixManage.cuh>
#include <Motutapu/util/MemoryManager.hpp>

namespace Motutapu::Compute
{
void DeepAllocateSparseHost(SparseMatrix** targetPtr, uint32_t m, uint32_t nnz,
                            uint32_t numMatrices)
{
    SparseMatrix* target = *targetPtr;
    target = new SparseMatrix[numMatrices];
    target->M = m;
    target->NNZ = nnz;
    target->V = static_cast<float*>(
        Util::MemoryManager::GetMemoryHost(nnz * sizeof(float)));
    target->COL = static_cast<uint32_t*>(
        Util::MemoryManager::GetMemoryHost(nnz * sizeof(uint32_t)));
    target->ROW = static_cast<uint32_t*>(
        Util::MemoryManager::GetMemoryHost((m + 1) * sizeof(uint32_t)));
}

void DeepFreeSparseHost(SparseMatrix* target)
{
    delete[] target;
    Util::MemoryManager::DeReferenceHost(static_cast<void*>(target->V));
    Util::MemoryManager::DeReferenceHost(static_cast<void*>(target->COL));
    Util::MemoryManager::DeReferenceHost(static_cast<void*>(target->ROW));
}

void DeepAllocateSparseCuda(SparseMatrix** cudaTargetPtr,
                            SparseMatrix* hostTarget, uint32_t numMatrices,
                            int deviceId)
{
    SparseMatrix* cudaTarget = *cudaTargetPtr;
    Cuda::CudaMalloc((void**)(cudaTarget), sizeof(SparseMatrix) * numMatrices);
    Cuda::MemcpyHostToGpu(cudaTarget, hostTarget, sizeof(SparseMatrix));

    cudaTarget->V = static_cast<float*>(Util::MemoryManager::GetMemoryCuda(
        hostTarget->NNZ * sizeof(float), deviceId));
    cudaTarget->COL = static_cast<uint32_t*>(Util::MemoryManager::GetMemoryCuda(
        hostTarget->NNZ * sizeof(uint32_t), deviceId));
    cudaTarget->ROW = static_cast<uint32_t*>(Util::MemoryManager::GetMemoryHost(
        (hostTarget->M + 1) * sizeof(uint32_t)));
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

void ConvertDenseToSparseHost(SparseMatrix* dst, float* src, uint32_t numRows,
                              uint32_t numCols, uint32_t numMatrices)
{
    const auto matrixSize = numRows * numCols;
    for (uint32_t matrixIdx = 0; matrixIdx < numMatrices; matrixIdx++)
    {
        uint32_t nnz = 0;
        dst[matrixIdx].ROW[0] = 0;
        for (uint32_t rowIdx = 0; rowIdx < numRows; ++rowIdx)
        {
            for (uint32_t colIdx = 0; colIdx < numCols; ++colIdx)
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

void ConvertDenseToSparseCuda(SparseMatrix* dst, float* src, uint32_t numRows,
                              uint32_t numCols, uint32_t numMatrices)
{
}

void ConvertSparseToDenseHost(float* dst, SparseMatrix* src, uint32_t numRows,
                              uint32_t numCols, uint32_t numMatrices)
{
    const auto matrixSize = numRows * numCols;
    for (uint32_t matrixIdx = 0; matrixIdx < numMatrices; ++matrixIdx)
    {
        for (uint32_t rowIdx = 0; rowIdx < src[matrixIdx].M; ++rowIdx)
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

void ConvertSparseToDenseCuda(float* dst, SparseMatrix* src, uint32_t numRows,
                              uint32_t numCols, uint32_t numMatrices)
{
}

void CopySparseDeviceToHost(SparseMatrix* dst, SparseMatrix* src,
                            size_t numMatrices)
{
    Cuda::MemcpyGpuToHost(dst, src, numMatrices * sizeof(SparseMatrix));

    for (uint32_t matrixIdx = 0; matrixIdx < numMatrices; ++matrixIdx)
    {
        Cuda::MemcpyGpuToHost(dst[matrixIdx].ROW, src[matrixIdx].ROW,
                              sizeof(uint32_t) * (src[matrixIdx].M + 1));

        Util::MemoryManager::DeReferenceHost(dst[matrixIdx].COL);
        Util::MemoryManager::DeReferenceHost(dst[matrixIdx].V);

        dst[matrixIdx].COL =
            static_cast<uint32_t*>(Util::MemoryManager::GetMemoryHost(
                sizeof(uint32_t) * src[matrixIdx].NNZ));
        dst[matrixIdx].V =
            static_cast<float*>(Util::MemoryManager::GetMemoryHost(
                sizeof(float) * src[matrixIdx].NNZ));

        Cuda::MemcpyGpuToHost(dst[matrixIdx].V, src[matrixIdx].V,
                              sizeof(float) * dst[matrixIdx].NNZ);
        Cuda::MemcpyGpuToHost(dst[matrixIdx].COL, src[matrixIdx].COL,
                              sizeof(uint32_t) * dst[matrixIdx].NNZ);
    }
}

void CopySparseHostToDevice(SparseMatrix* dst, SparseMatrix* src,
                            size_t numMatrices, int deviceId)
{
    Cuda::MemcpyHostToGpu(dst, src, numMatrices * sizeof(SparseMatrix));

    for (uint32_t matrixIdx = 0; matrixIdx < numMatrices; ++matrixIdx)
    {
        Cuda::MemcpyHostToGpu(dst[matrixIdx].ROW, src[matrixIdx].ROW,
                              sizeof(uint32_t) * (src[matrixIdx].M + 1));

        Util::MemoryManager::DeReferenceCuda(dst[matrixIdx].COL, deviceId);
        Util::MemoryManager::DeReferenceCuda(dst[matrixIdx].V, deviceId);

        dst[matrixIdx].COL =
            static_cast<uint32_t*>(Util::MemoryManager::GetMemoryCuda(
                sizeof(uint32_t) * src[matrixIdx].NNZ, deviceId));
        dst[matrixIdx].V =
            static_cast<float*>(Util::MemoryManager::GetMemoryCuda(
                sizeof(float) * src[matrixIdx].NNZ, deviceId));

        Cuda::MemcpyHostToGpu(dst[matrixIdx].V, src[matrixIdx].V,
                              sizeof(float) * dst[matrixIdx].NNZ);
        Cuda::MemcpyHostToGpu(dst[matrixIdx].COL, src[matrixIdx].COL,
                              sizeof(uint32_t) * dst[matrixIdx].NNZ);
    }
}

void LaunchSparseGemm(SparseMatrix* A, SparseMatrix* B, uint32_t numMatrices,
                      uint32_t numRows, uint32_t numCols, int deviceId)
{
    SparseMatrix* loadDiffHost = nullptr;
    SparseMatrix* loadDiffCuda = nullptr;

    DeepAllocateSparseHost(&loadDiffHost, A->M, A->NNZ, numMatrices);
    DeepAllocateSparseCuda(&loadDiffCuda, loadDiffHost, numMatrices, deviceId);

    CalculateLoad(A, B, loadDiffCuda, numMatrices);

    Cuda::Sparse::DeepCopyGpuToHost(loadDiffHost, loadDiffCuda, numMatrices);

    uint32_t nnzArray[numMatrices][numRows];

    for (uint32_t matrixIdx = 0; matrixIdx < numMatrices; ++matrixIdx)
        for (uint32_t rowIdx = 0; rowIdx < loadDiffCuda[matrixIdx].M; ++rowIdx)
        {
            const auto from = loadDiffHost[matrixIdx].ROW[rowIdx];
            const auto to = loadDiffHost[matrixIdx].ROW[rowIdx + 1];

            uint32_t nnz = 0;

            for (uint32_t sparseColIdx = from; sparseColIdx < to;
                 ++sparseColIdx)
            {
                nnz += loadDiffHost[matrixIdx].V[sparseColIdx];
            }

            nnzArray[matrixIdx][rowIdx] = nnz;
        }

    for (uint32_t matrixIdx = 0; matrixIdx < numMatrices; ++matrixIdx)
        for (uint32_t rowIdx = 0; rowIdx < A[matrixIdx].M; ++rowIdx)
        {
            //! Launch calculation Kernel based on load distribution
        }
}

void AllocateGemm(SparseMatrix* Out, SparseMatrix* A, SparseMatrix* B)
{
}

}  // namespace Motutapu::Compute