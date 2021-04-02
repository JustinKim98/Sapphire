// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/SparseMatrix.hpp>
#include <Motutapu/compute/cuda/Memory.cuh>
#include <Motutapu/compute/cuda/sparse/MatrixManage.cuh>
#include <Motutapu/compute/cuda/sparse/Sparse.hpp>
#include <Motutapu/compute/cuda/sparse/SparseGemm.cuh>
#include <Motutapu/util/MemoryManager.hpp>

namespace Motutapu::Compute::Cuda::Sparse
{
void DeepAllocateSparseHost(SparseMatrix** targetPtr, const uint32_t m[],
                            const uint32_t nnz[], uint32_t numMatrices)
{
    *targetPtr = new SparseMatrix[numMatrices];
    SparseMatrix* target = *targetPtr;

    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        target[i].M = m[i];
        target[i].NNZ = nnz[i];
        target[i].V = static_cast<float*>(
            Util::MemoryManager::GetMemoryHost(nnz[i] * sizeof(float)));
        target[i].COL = static_cast<uint32_t*>(
            Util::MemoryManager::GetMemoryHost(nnz[i] * sizeof(uint32_t)));
        target[i].ROW = static_cast<uint32_t*>(
            Util::MemoryManager::GetMemoryHost((m[i] + 1) * sizeof(uint32_t)));
    }
}

void DeepAllocateLoadDistMatrixHost(LoadDistMatrix** loadDistArray,
                                    SparseMatrix* sparseArray,
                                    uint32_t numMatrices)
{
    *loadDistArray = new LoadDistMatrix[numMatrices];
    LoadDistMatrix* target = *loadDistArray;

    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        target[i].M = sparseArray[i].M;
        target[i].NNZ = sparseArray[i].NNZ;
        target[i].Load =
            static_cast<uint32_t*>(Util::MemoryManager::GetMemoryHost(
                sparseArray[i].NNZ * sizeof(uint32_t)));
        target[i].COL =
            static_cast<uint32_t*>(Util::MemoryManager::GetMemoryHost(
                sparseArray[i].NNZ * sizeof(uint32_t)));
        target[i].ROW =
            static_cast<uint32_t*>(Util::MemoryManager::GetMemoryHost(
                (sparseArray[i].M + 1) * sizeof(uint32_t)));
    }
}

void DeepFreeSparseHost(SparseMatrix* target, uint32_t numMatrices)
{
    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        Util::MemoryManager::DeReferenceHost(static_cast<void*>(target[i].V));
        Util::MemoryManager::DeReferenceHost(static_cast<void*>(target[i].COL));
        Util::MemoryManager::DeReferenceHost(static_cast<void*>(target[i].ROW));
    }

    delete[] target;
}

void DeepFreeLoadDistHost(LoadDistMatrix* target, uint32_t numMatrices)
{
    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        Util::MemoryManager::DeReferenceHost(
            static_cast<void*>(target[i].Load));
        Util::MemoryManager::DeReferenceHost(static_cast<void*>(target[i].COL));
        Util::MemoryManager::DeReferenceHost(static_cast<void*>(target[i].ROW));
    }

    delete[] target;
}

void DeepAllocateSparseCuda(SparseMatrix** targetPtr, SparseMatrix* hostPtr,
                            uint32_t numMatrices, int deviceId)
{
    Cuda::CudaMalloc((void**)(targetPtr), sizeof(SparseMatrix) * numMatrices);
    SparseMatrix* cudaTarget = *targetPtr;
    Cuda::CopyHostToGpu(cudaTarget, hostPtr,
                        sizeof(SparseMatrix) * numMatrices);

    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        cudaTarget[i].V =
            static_cast<float*>(Util::MemoryManager::GetMemoryCuda(
                hostPtr[i].NNZ * sizeof(float), deviceId));
        cudaTarget[i].COL =
            static_cast<uint32_t*>(Util::MemoryManager::GetMemoryCuda(
                hostPtr[i].NNZ * sizeof(uint32_t), deviceId));
        cudaTarget[i].ROW =
            static_cast<uint32_t*>(Util::MemoryManager::GetMemoryHost(
                (hostPtr[i].M + 1) * sizeof(uint32_t)));
    }
}

void DeepAllocateLoadDistCuda(LoadDistMatrix** targetPtr,
                              LoadDistMatrix* hostPtr, uint32_t numMatrices,
                              int deviceId)
{
    Cuda::CudaMalloc((void**)(targetPtr), sizeof(SparseMatrix) * numMatrices);
    LoadDistMatrix* cudaTarget = *targetPtr;
    Cuda::CopyHostToGpu(cudaTarget, hostPtr,
                        sizeof(SparseMatrix) * numMatrices);

    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        cudaTarget[i].Load =
            static_cast<uint32_t*>(Util::MemoryManager::GetMemoryCuda(
                hostPtr[i].NNZ * sizeof(uint32_t), deviceId));
        cudaTarget[i].COL =
            static_cast<uint32_t*>(Util::MemoryManager::GetMemoryCuda(
                hostPtr[i].NNZ * sizeof(uint32_t), deviceId));
        cudaTarget[i].ROW =
            static_cast<uint32_t*>(Util::MemoryManager::GetMemoryHost(
                (hostPtr[i].M + 1) * sizeof(uint32_t)));
    }
}

void DeepFreeSparseCuda(SparseMatrix* targetPtr, int deviceId)
{
    Util::MemoryManager::DeReferenceCuda(static_cast<void*>(targetPtr->V),
                                         deviceId);
    Util::MemoryManager::DeReferenceCuda(static_cast<void*>(targetPtr->COL),
                                         deviceId);
    Util::MemoryManager::DeReferenceCuda(static_cast<void*>(targetPtr->ROW),
                                         deviceId);
    Cuda::CudaFree((void*)targetPtr);
}

void DeepFreeLoadDistCuda(LoadDistMatrix* cudaTarget, int deviceId)
{
    Util::MemoryManager::DeReferenceCuda(static_cast<void*>(cudaTarget->Load),
                                         deviceId);
    Util::MemoryManager::DeReferenceCuda(static_cast<void*>(cudaTarget->COL),
                                         deviceId);
    Util::MemoryManager::DeReferenceCuda(static_cast<void*>(cudaTarget->ROW),
                                         deviceId);
    Cuda::CudaFree((void*)cudaTarget);
}

void DeepCopyGpuToGpu(SparseMatrix* gpuDstArray, SparseMatrix* gpuSrcArray,
                      uint32_t numMatrices, int deviceId)
{
    auto* hostDstArray =
        static_cast<SparseMatrix*>(malloc(numMatrices * sizeof(SparseMatrix)));
    auto* hostSrcArray =
        static_cast<SparseMatrix*>(malloc(numMatrices * sizeof(SparseMatrix)));

    CopyGpuToHost(hostDstArray, gpuDstArray,
                  numMatrices * sizeof(SparseMatrix));
    CopyGpuToHost(hostSrcArray, gpuSrcArray,
                  numMatrices * sizeof(SparseMatrix));

    for (int idx = 0; idx < numMatrices; ++idx)
    {
        if (hostDstArray[idx].NNZ != hostSrcArray[idx].NNZ)
        {
            Util::MemoryManager::DeReferenceCuda(hostDstArray[idx].COL,
                                                 deviceId);
            Util::MemoryManager::DeReferenceCuda(hostDstArray[idx].V, deviceId);
            hostDstArray[idx].COL =
                static_cast<uint32_t*>(Util::MemoryManager::GetMemoryCuda(
                    gpuSrcArray[idx].NNZ * sizeof(uint32_t), deviceId));
            hostDstArray[idx].V =
                static_cast<float*>(Util::MemoryManager::GetMemoryCuda(
                    gpuSrcArray[idx].NNZ * sizeof(float), deviceId));
            hostDstArray[idx].NNZ = hostSrcArray[idx].NNZ;
        }
        if (hostDstArray[idx].M != hostSrcArray[idx].M)
        {
            Util::MemoryManager::DeReferenceCuda(hostDstArray[idx].ROW,
                                                 deviceId);
            hostDstArray[idx].ROW =
                static_cast<uint32_t*>(Util::MemoryManager::GetMemoryCuda(
                    (gpuSrcArray[idx].M + 1) * sizeof(uint32_t), deviceId));
            hostDstArray[idx].M = hostSrcArray[idx].M;
        }

        Cuda::CopyGpuToGpu(gpuDstArray[idx].COL, gpuSrcArray[idx].COL,
                           gpuSrcArray[idx].NNZ * sizeof(uint32_t));
        Cuda::CopyGpuToGpu(gpuDstArray[idx].V, gpuSrcArray[idx].V,
                           gpuSrcArray[idx].NNZ * sizeof(float));
        Cuda::CopyGpuToGpu(gpuDstArray[idx].ROW, gpuSrcArray[idx].ROW,
                           gpuSrcArray[idx].NNZ * sizeof(uint32_t));
    }

    CopyHostToGpu(gpuDstArray, hostDstArray,
                  numMatrices * sizeof(SparseMatrix));
}

void DeepCopyGpuToGpu(LoadDistMatrix* gpuDstArray, LoadDistMatrix* gpuSrcArray,
                      uint32_t numMatrices, int deviceId)
{
    auto* hostDstArray = static_cast<LoadDistMatrix*>(
        malloc(numMatrices * sizeof(LoadDistMatrix)));
    auto* hostSrcArray = static_cast<LoadDistMatrix*>(
        malloc(numMatrices * sizeof(LoadDistMatrix)));

    CopyGpuToHost(hostDstArray, gpuDstArray,
                  numMatrices * sizeof(LoadDistMatrix));
    CopyGpuToHost(hostSrcArray, gpuSrcArray,
                  numMatrices * sizeof(LoadDistMatrix));

    for (int idx = 0; idx < numMatrices; ++idx)
    {
        if (hostDstArray[idx].NNZ != hostSrcArray[idx].NNZ)
        {
            Util::MemoryManager::DeReferenceCuda(hostDstArray[idx].COL,
                                                 deviceId);
            Util::MemoryManager::DeReferenceCuda(hostDstArray[idx].Load,
                                                 deviceId);
            hostDstArray[idx].COL =
                static_cast<uint32_t*>(Util::MemoryManager::GetMemoryCuda(
                    gpuSrcArray[idx].NNZ * sizeof(uint32_t), deviceId));
            hostDstArray[idx].Load =
                static_cast<uint32_t*>(Util::MemoryManager::GetMemoryCuda(
                    gpuSrcArray[idx].NNZ * sizeof(uint32_t), deviceId));
            hostDstArray[idx].NNZ = hostSrcArray[idx].NNZ;
        }
        if (hostDstArray[idx].M != hostSrcArray[idx].M)
        {
            Util::MemoryManager::DeReferenceCuda(hostDstArray[idx].ROW,
                                                 deviceId);
            hostDstArray[idx].ROW =
                static_cast<uint32_t*>(Util::MemoryManager::GetMemoryCuda(
                    (gpuSrcArray[idx].M + 1) * sizeof(uint32_t), deviceId));
            hostDstArray[idx].M = hostSrcArray[idx].M;
        }

        Cuda::CopyGpuToGpu(gpuDstArray[idx].COL, gpuSrcArray[idx].COL,
                           gpuSrcArray[idx].NNZ * sizeof(uint32_t));
        Cuda::CopyGpuToGpu(gpuDstArray[idx].Load, gpuSrcArray[idx].Load,
                           gpuSrcArray[idx].NNZ * sizeof(uint32_t));
        Cuda::CopyGpuToGpu(gpuDstArray[idx].ROW, gpuSrcArray[idx].ROW,
                           gpuSrcArray[idx].NNZ * sizeof(uint32_t));
    }

    CopyHostToGpu(gpuDstArray, hostDstArray,
                  numMatrices * sizeof(SparseMatrix));
}

void DeepCopyHostToGpu(SparseMatrix* gpuDstArray, SparseMatrix* hostSrcArray,
                       uint32_t numMatrices, int deviceId)
{
    auto* hostDstArray =
        static_cast<SparseMatrix*>(malloc(numMatrices * sizeof(SparseMatrix)));

    CopyGpuToHost(hostDstArray, gpuDstArray,
                  numMatrices * sizeof(SparseMatrix));
    CopyHostToGpu(gpuDstArray, hostSrcArray,
                  numMatrices * sizeof(SparseMatrix));

    for (int idx = 0; idx < numMatrices; ++idx)
    {
        if (hostDstArray[idx].NNZ != hostSrcArray[idx].NNZ)
        {
            const auto nnz = hostSrcArray[idx].NNZ;
            Util::MemoryManager::DeReferenceCuda(gpuDstArray[idx].V, deviceId);
            Util::MemoryManager::DeReferenceCuda(gpuDstArray[idx].COL,
                                                 deviceId);

            gpuDstArray[idx].COL =
                static_cast<uint32_t*>(Util::MemoryManager::GetMemoryCuda(
                    sizeof(uint32_t) * nnz, deviceId));
            gpuDstArray[idx].V =
                static_cast<float*>(Util::MemoryManager::GetMemoryCuda(
                    sizeof(float) * nnz, deviceId));
            hostDstArray[idx].NNZ = nnz;
        }
        if (hostDstArray[idx].M != hostSrcArray[idx].M)
        {
            const auto M = hostSrcArray[idx].M;
            Util::MemoryManager::DeReferenceCuda(gpuDstArray[idx].ROW,
                                                 deviceId);
            gpuDstArray[idx].ROW =
                static_cast<uint32_t*>(Util::MemoryManager::GetMemoryCuda(
                    M * sizeof(uint32_t), deviceId));
            hostDstArray[idx].M = M;
        }

        CopyHostToGpu(gpuDstArray, hostDstArray,
                      numMatrices * sizeof(SparseMatrix));
    }
}

void DeepCopyHostToGpu(LoadDistMatrix* gpuDstArray,
                       LoadDistMatrix* hostSrcArray, uint32_t numMatrices,
                       int deviceId)
{
    auto* hostDstArray = static_cast<LoadDistMatrix*>(
        malloc(numMatrices * sizeof(LoadDistMatrix)));

    CopyGpuToHost(hostDstArray, gpuDstArray,
                  numMatrices * sizeof(LoadDistMatrix));
    CopyHostToGpu(gpuDstArray, hostSrcArray,
                  numMatrices * sizeof(LoadDistMatrix));

    for (int idx = 0; idx < numMatrices; ++idx)
    {
        if (hostDstArray[idx].NNZ != hostSrcArray[idx].NNZ)
        {
            const auto nnz = hostSrcArray[idx].NNZ;
            Util::MemoryManager::DeReferenceCuda(gpuDstArray[idx].Load,
                                                 deviceId);
            Util::MemoryManager::DeReferenceCuda(gpuDstArray[idx].COL,
                                                 deviceId);

            gpuDstArray[idx].COL =
                static_cast<uint32_t*>(Util::MemoryManager::GetMemoryCuda(
                    sizeof(uint32_t) * nnz, deviceId));
            gpuDstArray[idx].Load =
                static_cast<uint32_t*>(Util::MemoryManager::GetMemoryCuda(
                    sizeof(uint32_t) * nnz, deviceId));
            hostDstArray[idx].NNZ = nnz;
        }
        if (hostDstArray[idx].M != hostSrcArray[idx].M)
        {
            const auto M = hostSrcArray[idx].M;
            Util::MemoryManager::DeReferenceCuda(gpuDstArray[idx].ROW,
                                                 deviceId);
            gpuDstArray[idx].ROW =
                static_cast<uint32_t*>(Util::MemoryManager::GetMemoryCuda(
                    M * sizeof(uint32_t), deviceId));
            hostDstArray[idx].M = M;
        }

        CopyHostToGpu(gpuDstArray, hostDstArray,
                      numMatrices * sizeof(LoadDistMatrix));
    }
}

void DeepCopyGpuToHost(SparseMatrix* hostDstArray, SparseMatrix* gpuSrcArray,
                       uint32_t numMatrices)
{
    auto* hostSrcArray =
        static_cast<SparseMatrix*>(malloc(numMatrices * sizeof(SparseMatrix)));
    CopyGpuToHost(hostSrcArray, gpuSrcArray,
                  numMatrices * sizeof(SparseMatrix));

    for (int idx = 0; idx < numMatrices; ++idx)
    {
        if (hostDstArray[idx].NNZ != hostSrcArray[idx].NNZ)
        {
            Util::MemoryManager::DeReferenceHost(hostDstArray[idx].COL);
            Util::MemoryManager::DeReferenceHost(hostDstArray[idx].V);
            hostDstArray[idx].COL =
                static_cast<uint32_t*>(Util::MemoryManager::GetMemoryHost(
                    hostSrcArray[idx].NNZ * sizeof(uint32_t)));
            hostDstArray[idx].V =
                static_cast<float*>(Util::MemoryManager::GetMemoryHost(
                    hostSrcArray[idx].NNZ * sizeof(float)));
            hostDstArray[idx].NNZ = hostSrcArray[idx].NNZ;
        }
        if (hostDstArray[idx].M != hostSrcArray[idx].M)
        {
            Util::MemoryManager::DeReferenceHost(hostDstArray[idx].ROW);
            hostDstArray[idx].ROW =
                static_cast<uint32_t*>(Util::MemoryManager::GetMemoryHost(
                    (hostDstArray[idx].M + 1) * sizeof(uint32_t)));
            hostDstArray[idx].M = hostSrcArray[idx].M;
        }

        Cuda::CopyGpuToHost(hostDstArray[idx].COL, gpuSrcArray[idx].COL,
                            hostDstArray[idx].NNZ * sizeof(uint32_t));
        Cuda::CopyGpuToHost(hostDstArray[idx].V, gpuSrcArray[idx].V,
                            hostDstArray[idx].NNZ * sizeof(uint32_t));
        Cuda::CopyGpuToHost(hostDstArray[idx].ROW, gpuSrcArray[idx].ROW,
                            (hostDstArray[idx].M + 1) * sizeof(uint32_t));
    }
}

void DeepCopyGpuToHost(LoadDistMatrix* hostDstArray,
                       LoadDistMatrix* gpuSrcArray, uint32_t numMatrices)
{
    auto* hostSrcArray = static_cast<LoadDistMatrix*>(
        malloc(numMatrices * sizeof(LoadDistMatrix)));
    CopyGpuToHost(hostSrcArray, gpuSrcArray,
                  numMatrices * sizeof(LoadDistMatrix));

    for (int idx = 0; idx < numMatrices; ++idx)
    {
        if (hostDstArray[idx].NNZ != hostSrcArray[idx].NNZ)
        {
            Util::MemoryManager::DeReferenceHost(hostDstArray[idx].COL);
            Util::MemoryManager::DeReferenceHost(hostDstArray[idx].Load);
            hostDstArray[idx].COL =
                static_cast<uint32_t*>(Util::MemoryManager::GetMemoryHost(
                    hostSrcArray[idx].NNZ * sizeof(uint32_t)));
            hostDstArray[idx].Load =
                static_cast<uint32_t*>(Util::MemoryManager::GetMemoryHost(
                    hostSrcArray[idx].NNZ * sizeof(uint32_t)));
            hostDstArray[idx].NNZ = hostSrcArray[idx].NNZ;
        }
        if (hostDstArray[idx].M != hostSrcArray[idx].M)
        {
            Util::MemoryManager::DeReferenceHost(hostDstArray[idx].ROW);
            hostDstArray[idx].ROW =
                static_cast<uint32_t*>(Util::MemoryManager::GetMemoryHost(
                    (hostDstArray[idx].M + 1) * sizeof(uint32_t)));
            hostDstArray[idx].M = hostSrcArray[idx].M;
        }

        Cuda::CopyGpuToHost(hostDstArray[idx].COL, gpuSrcArray[idx].COL,
                            hostDstArray[idx].NNZ * sizeof(uint32_t));
        Cuda::CopyGpuToHost(hostDstArray[idx].Load, gpuSrcArray[idx].Load,
                            hostDstArray[idx].NNZ * sizeof(uint32_t));
        Cuda::CopyGpuToHost(hostDstArray[idx].ROW, gpuSrcArray[idx].ROW,
                            (hostDstArray[idx].M + 1) * sizeof(uint32_t));
    }
}

void ConvertDenseToSparseHost(SparseMatrix* dst, const float* src,
                              uint32_t numRows, uint32_t numCols,
                              uint32_t numMatrices)
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

void ConvertSparseToDenseHost(float* dst, const SparseMatrix* src,
                              uint32_t numRows, uint32_t numCols,
                              uint32_t numMatrices)
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
    Cuda::CopyGpuToHost(dst, src, numMatrices * sizeof(SparseMatrix));

    for (uint32_t matrixIdx = 0; matrixIdx < numMatrices; ++matrixIdx)
    {
        Cuda::CopyGpuToHost(dst[matrixIdx].ROW, src[matrixIdx].ROW,
                            sizeof(uint32_t) * (src[matrixIdx].M + 1));

        Util::MemoryManager::DeReferenceHost(dst[matrixIdx].COL);
        Util::MemoryManager::DeReferenceHost(dst[matrixIdx].V);

        dst[matrixIdx].COL =
            static_cast<uint32_t*>(Util::MemoryManager::GetMemoryHost(
                sizeof(uint32_t) * src[matrixIdx].NNZ));
        dst[matrixIdx].V =
            static_cast<float*>(Util::MemoryManager::GetMemoryHost(
                sizeof(float) * src[matrixIdx].NNZ));

        Cuda::CopyGpuToHost(dst[matrixIdx].V, src[matrixIdx].V,
                            sizeof(float) * dst[matrixIdx].NNZ);
        Cuda::CopyGpuToHost(dst[matrixIdx].COL, src[matrixIdx].COL,
                            sizeof(uint32_t) * dst[matrixIdx].NNZ);
    }
}

void CopySparseHostToDevice(SparseMatrix* dst, SparseMatrix* src,
                            size_t numMatrices, int deviceId)
{
    Cuda::CopyHostToGpu(dst, src, numMatrices * sizeof(SparseMatrix));

    for (uint32_t matrixIdx = 0; matrixIdx < numMatrices; ++matrixIdx)
    {
        Cuda::CopyHostToGpu(dst[matrixIdx].ROW, src[matrixIdx].ROW,
                            sizeof(uint32_t) * (src[matrixIdx].M + 1));

        Util::MemoryManager::DeReferenceCuda(dst[matrixIdx].COL, deviceId);
        Util::MemoryManager::DeReferenceCuda(dst[matrixIdx].V, deviceId);

        dst[matrixIdx].COL =
            static_cast<uint32_t*>(Util::MemoryManager::GetMemoryCuda(
                sizeof(uint32_t) * src[matrixIdx].NNZ, deviceId));
        dst[matrixIdx].V =
            static_cast<float*>(Util::MemoryManager::GetMemoryCuda(
                sizeof(float) * src[matrixIdx].NNZ, deviceId));

        Cuda::CopyHostToGpu(dst[matrixIdx].V, src[matrixIdx].V,
                            sizeof(float) * dst[matrixIdx].NNZ);
        Cuda::CopyHostToGpu(dst[matrixIdx].COL, src[matrixIdx].COL,
                            sizeof(uint32_t) * dst[matrixIdx].NNZ);
    }
}

void LaunchSparseGemm(SparseMatrix* A, SparseMatrix* B, uint32_t numMatrices,
                      uint32_t numRows, uint32_t numCols, int deviceId)
{
    LoadDistMatrix* loadDiffHost = nullptr;
    LoadDistMatrix* loadDiffCuda = nullptr;

    DeepAllocateLoadDistMatrixHost(&loadDiffHost, A, numMatrices);
    DeepAllocateLoadDistCuda(&loadDiffCuda, loadDiffHost, numMatrices,
                             deviceId);

    CalculateLoad(A, B, loadDiffCuda, numMatrices);

    //! todo : Avoid copying load data from GPU to Host
    DeepCopyHostToGpu(loadDiffHost, loadDiffCuda, numMatrices, 0);

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
                nnz += loadDiffHost[matrixIdx].Load[sparseColIdx];
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

}  // namespace Motutapu::Compute::Cuda::Sparse