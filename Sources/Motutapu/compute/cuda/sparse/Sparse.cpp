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
using namespace Util;

void DeepAllocateSparseHost(SparseMatrix** sparseMatrixArray,
                            const uint32_t m[], const uint32_t nnz[],
                            uint32_t numMatrices)
{
    *sparseMatrixArray = (SparseMatrix*)MemoryManager::GetMemoryHost(
        sizeof(SparseMatrix) * numMatrices);
    SparseMatrix* targetArray = *sparseMatrixArray;

    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        targetArray[i].M = m[i];
        targetArray[i].NNZ = nnz[i];
        targetArray[i].V = static_cast<float*>(
            Util::MemoryManager::GetMemoryHost(nnz[i] * sizeof(float)));
        targetArray[i].COL = static_cast<uint32_t*>(
            Util::MemoryManager::GetMemoryHost(nnz[i] * sizeof(uint32_t)));
        targetArray[i].ROW = static_cast<uint32_t*>(
            Util::MemoryManager::GetMemoryHost((m[i] + 1) * sizeof(uint32_t)));
    }
}

void DeepAllocateLoadDistMatrixHost(LoadDistMatrix** loadDistArray,
                                    SparseMatrix* sparseArray,
                                    uint32_t numMatrices)
{
    *loadDistArray = (LoadDistMatrix*)MemoryManager::GetMemoryHost(
        sizeof(LoadDistMatrix) * numMatrices);
    LoadDistMatrix* targetArray = *loadDistArray;

    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        targetArray[i].M = sparseArray[i].M;
        targetArray[i].NNZ = sparseArray[i].NNZ;
        targetArray[i].Load =
            static_cast<uint32_t*>(Util::MemoryManager::GetMemoryHost(
                sparseArray[i].NNZ * sizeof(uint32_t)));
        targetArray[i].COL =
            static_cast<uint32_t*>(Util::MemoryManager::GetMemoryHost(
                sparseArray[i].NNZ * sizeof(uint32_t)));
        targetArray[i].ROW =
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
        cudaSetDevice(deviceId);
        cudaMalloc(&cudaTarget[i].Load, hostPtr[i].NNZ * sizeof(uint32_t));
        cudaMalloc(&cudaTarget[i].COL, hostPtr[i].NNZ * sizeof(uint32_t));
        cudaMalloc(&cudaTarget[i].ROW, (hostPtr[i].M + 1) * sizeof(uint32_t));
    }
}

void DeepFreeSparseCuda(SparseMatrix* targetPtr, uint32_t numMatrices,
                        int deviceId)
{
    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        cudaFree(targetPtr[i].V);
        cudaFree(targetPtr[i].COL);
        cudaFree(targetPtr[i].ROW);
    }
    Cuda::CudaFree(targetPtr);
}

void DeepFreeLoadDistCuda(LoadDistMatrix* targetPtr, uint32_t numMatrices,
                          int deviceId)
{
    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        cudaFree(targetPtr[i].Load);
        cudaFree(targetPtr[i].COL);
        cudaFree(targetPtr[i].ROW);
    }
    Cuda::CudaFree(targetPtr);
}

void DeepCopyGpuToGpu(SparseMatrix* gpuDstArray, SparseMatrix* gpuSrcArray,
                      uint32_t numMatrices, int deviceId)
{
    auto* dstArrayBuffer = static_cast<SparseMatrix*>(
        MemoryManager::GetMemoryHost(numMatrices * sizeof(SparseMatrix)));
    auto* srcArrayBuffer = static_cast<SparseMatrix*>(
        MemoryManager::GetMemoryHost(numMatrices * sizeof(SparseMatrix)));

    cudaSetDevice(deviceId);
    CopyGpuToHost(dstArrayBuffer, gpuDstArray,
                  numMatrices * sizeof(SparseMatrix));
    CopyGpuToHost(srcArrayBuffer, gpuSrcArray,
                  numMatrices * sizeof(SparseMatrix));

    for (int idx = 0; idx < numMatrices; ++idx)
    {
        if (dstArrayBuffer[idx].NNZ != srcArrayBuffer[idx].NNZ)
        {
            MemoryManager::DeReferenceCuda(dstArrayBuffer[idx].COL, deviceId);
            MemoryManager::DeReferenceCuda(dstArrayBuffer[idx].V, deviceId);
            dstArrayBuffer[idx].COL = (uint32_t*)MemoryManager::GetMemoryCuda(
                srcArrayBuffer[idx].NNZ * sizeof(uint32_t), deviceId);
            dstArrayBuffer[idx].V = (float*)MemoryManager::GetMemoryCuda(
                srcArrayBuffer[idx].NNZ * sizeof(float), deviceId);
        }
        if (dstArrayBuffer[idx].M != srcArrayBuffer[idx].M)
        {
            MemoryManager::DeReferenceCuda(dstArrayBuffer[idx].ROW, deviceId);
            dstArrayBuffer[idx].ROW = (uint32_t*)MemoryManager::GetMemoryCuda(
                (srcArrayBuffer[idx].M + 1) * sizeof(uint32_t), deviceId);
        }

        dstArrayBuffer[idx].M = srcArrayBuffer[idx].M;
        dstArrayBuffer[idx].N = srcArrayBuffer[idx].N;
        dstArrayBuffer[idx].NNZ = srcArrayBuffer[idx].NNZ;

        Cuda::CopyGpuToGpu(dstArrayBuffer[idx].COL, srcArrayBuffer[idx].COL,
                           srcArrayBuffer[idx].NNZ * sizeof(uint32_t));
        Cuda::CopyGpuToGpu(dstArrayBuffer[idx].V, srcArrayBuffer[idx].V,
                           srcArrayBuffer[idx].NNZ * sizeof(float));
        Cuda::CopyGpuToGpu(dstArrayBuffer[idx].ROW, srcArrayBuffer[idx].ROW,
                           (srcArrayBuffer[idx].M + 1) * sizeof(uint32_t));
    }

    CopyHostToGpu(gpuDstArray, dstArrayBuffer,
                  numMatrices * sizeof(SparseMatrix));

    MemoryManager::DeReferenceHost(dstArrayBuffer);
    MemoryManager::DeReferenceHost(dstArrayBuffer);
}

void DeepCopyGpuToGpu(LoadDistMatrix* gpuDstArray, LoadDistMatrix* gpuSrcArray,
                      uint32_t numMatrices, int deviceId)
{
    auto* dstArrayBuffer = static_cast<LoadDistMatrix*>(
        MemoryManager::GetMemoryHost(numMatrices * sizeof(LoadDistMatrix)));
    auto* srcArrayBuffer = static_cast<LoadDistMatrix*>(
        MemoryManager::GetMemoryHost(numMatrices * sizeof(LoadDistMatrix)));

    cudaSetDevice(deviceId);
    CopyGpuToHost(dstArrayBuffer, gpuDstArray,
                  numMatrices * sizeof(LoadDistMatrix));
    CopyGpuToHost(srcArrayBuffer, gpuSrcArray,
                  numMatrices * sizeof(LoadDistMatrix));

    for (int idx = 0; idx < numMatrices; ++idx)
    {
        if (dstArrayBuffer[idx].NNZ != srcArrayBuffer[idx].NNZ)
        {
            MemoryManager::DeReferenceCuda(dstArrayBuffer[idx].COL, deviceId);
            MemoryManager::DeReferenceCuda(dstArrayBuffer[idx].Load, deviceId);
            dstArrayBuffer[idx].COL = (uint32_t*)MemoryManager::GetMemoryCuda(
                srcArrayBuffer[idx].NNZ * sizeof(uint32_t), deviceId);
            dstArrayBuffer[idx].Load = (uint32_t*)MemoryManager::GetMemoryCuda(
                srcArrayBuffer[idx].NNZ * sizeof(uint32_t), deviceId);
        }
        if (dstArrayBuffer[idx].M != srcArrayBuffer[idx].M)
        {
            MemoryManager::DeReferenceCuda(dstArrayBuffer[idx].ROW, deviceId);
            dstArrayBuffer[idx].ROW = (uint32_t*)MemoryManager::GetMemoryCuda(
                (srcArrayBuffer[idx].M + 1) * sizeof(uint32_t), deviceId);
        }

        dstArrayBuffer[idx].M = srcArrayBuffer[idx].M;
        dstArrayBuffer[idx].N = srcArrayBuffer[idx].N;
        dstArrayBuffer[idx].NNZ = srcArrayBuffer[idx].NNZ;

        Cuda::CopyGpuToGpu(dstArrayBuffer[idx].COL, srcArrayBuffer[idx].COL,
                           srcArrayBuffer[idx].NNZ * sizeof(uint32_t));
        Cuda::CopyGpuToGpu(dstArrayBuffer[idx].Load, srcArrayBuffer[idx].Load,
                           srcArrayBuffer[idx].NNZ * sizeof(uint32_t));
        Cuda::CopyGpuToGpu(dstArrayBuffer[idx].ROW, srcArrayBuffer[idx].ROW,
                           (srcArrayBuffer[idx].M + 1) * sizeof(uint32_t));
    }

    CopyHostToGpu(gpuDstArray, dstArrayBuffer,
                  numMatrices * sizeof(LoadDistMatrix));

    MemoryManager::DeReferenceHost(dstArrayBuffer);
    MemoryManager::DeReferenceHost(dstArrayBuffer);
}

void DeepCopyHostToGpu(SparseMatrix* gpuDstArray, SparseMatrix* hostSrcArray,
                       uint32_t numMatrices, int deviceId)
{
    auto* dstArrayBuffer = static_cast<SparseMatrix*>(
        MemoryManager::GetMemoryHost(numMatrices * sizeof(SparseMatrix)));

    cudaSetDevice(deviceId);
    CopyGpuToHost(dstArrayBuffer, gpuDstArray,
                  numMatrices * sizeof(SparseMatrix));

    for (int idx = 0; idx < numMatrices; ++idx)
    {
        const auto m = hostSrcArray[idx].M;
        const auto n = hostSrcArray[idx].N;
        const auto nnz = hostSrcArray[idx].NNZ;

        if (dstArrayBuffer[idx].NNZ != nnz)
        {
            MemoryManager::DeReferenceCuda(dstArrayBuffer[idx].V, deviceId);
            MemoryManager::DeReferenceCuda(dstArrayBuffer[idx].COL, deviceId);
            dstArrayBuffer[idx].V = (float*)MemoryManager::GetMemoryCuda(
                sizeof(float) * nnz, deviceId);
            dstArrayBuffer[idx].COL = (uint32_t*)MemoryManager::GetMemoryCuda(
                sizeof(uint32_t) * nnz, deviceId);
        }
        if (dstArrayBuffer[idx].M != m)
        {
            MemoryManager::DeReferenceCuda(dstArrayBuffer[idx].ROW, deviceId);
            dstArrayBuffer[idx].ROW = (uint32_t*)MemoryManager::GetMemoryCuda(
                sizeof(uint32_t) * (m + 1), deviceId);
        }

        dstArrayBuffer[idx].M = m;
        dstArrayBuffer[idx].N = n;
        dstArrayBuffer[idx].NNZ = nnz;

        CopyHostToGpu(dstArrayBuffer[idx].V, hostSrcArray[idx].V,
                      sizeof(float) * nnz);
        CopyHostToGpu(dstArrayBuffer[idx].COL, hostSrcArray[idx].COL,
                      sizeof(uint32_t) * nnz);
        CopyHostToGpu(dstArrayBuffer[idx].ROW, hostSrcArray[idx].ROW,
                      sizeof(uint32_t) * (m + 1));
    }
    CopyHostToGpu(gpuDstArray, dstArrayBuffer,
                  sizeof(SparseMatrix) * numMatrices);
    MemoryManager::DeReferenceHost(dstArrayBuffer);
}

void DeepCopyHostToGpu(LoadDistMatrix* gpuDstArray,
                       LoadDistMatrix* hostSrcArray, uint32_t numMatrices,
                       int deviceId)
{
    auto* dstArrayBuffer = static_cast<LoadDistMatrix*>(
        MemoryManager::GetMemoryHost(numMatrices * sizeof(LoadDistMatrix)));

    cudaSetDevice(deviceId);
    CopyGpuToHost(dstArrayBuffer, gpuDstArray,
                  numMatrices * sizeof(LoadDistMatrix));

    for (int idx = 0; idx < numMatrices; ++idx)
    {
        const auto m = hostSrcArray[idx].M;
        const auto n = hostSrcArray[idx].N;
        const auto nnz = hostSrcArray[idx].NNZ;

        if (dstArrayBuffer[idx].NNZ != nnz)
        {
            MemoryManager::DeReferenceCuda(dstArrayBuffer[idx].Load, deviceId);
            MemoryManager::DeReferenceCuda(dstArrayBuffer[idx].COL, deviceId);
            dstArrayBuffer[idx].Load = (uint32_t*)MemoryManager::GetMemoryCuda(
                sizeof(uint32_t) * nnz, deviceId);
            dstArrayBuffer[idx].COL = (uint32_t*)MemoryManager::GetMemoryCuda(
                sizeof(uint32_t) * nnz, deviceId);
        }
        if (dstArrayBuffer[idx].M != m)
        {
            MemoryManager::DeReferenceCuda(dstArrayBuffer[idx].ROW, deviceId);
            dstArrayBuffer[idx].ROW = (uint32_t*)MemoryManager::GetMemoryCuda(
                sizeof(uint32_t) * (m + 1), deviceId);
        }

        dstArrayBuffer[idx].M = m;
        dstArrayBuffer[idx].N = n;
        dstArrayBuffer[idx].NNZ = nnz;

        CopyHostToGpu(dstArrayBuffer[idx].Load, hostSrcArray[idx].Load,
                      sizeof(uint32_t) * nnz);
        CopyHostToGpu(dstArrayBuffer[idx].COL, hostSrcArray[idx].COL,
                      sizeof(uint32_t) * nnz);
        CopyHostToGpu(dstArrayBuffer[idx].ROW, hostSrcArray[idx].ROW,
                      sizeof(uint32_t) * (m + 1));
    }
    CopyHostToGpu(gpuDstArray, dstArrayBuffer,
                  sizeof(LoadDistMatrix) * numMatrices);
    MemoryManager::DeReferenceHost(dstArrayBuffer);
}

void DeepCopyGpuToHost(SparseMatrix* hostDstArray, SparseMatrix* gpuSrcArray,
                       uint32_t numMatrices)
{
    auto* srcArrayBuffer = static_cast<SparseMatrix*>(
        MemoryManager::GetMemoryHost(numMatrices * sizeof(SparseMatrix)));

    CopyGpuToHost(srcArrayBuffer, gpuSrcArray,
                  numMatrices * sizeof(SparseMatrix));

    for (int idx = 0; idx < numMatrices; ++idx)
    {
        if (hostDstArray[idx].NNZ != srcArrayBuffer[idx].NNZ)
        {
            Util::MemoryManager::DeReferenceHost(hostDstArray[idx].COL);
            Util::MemoryManager::DeReferenceHost(hostDstArray[idx].V);
            hostDstArray[idx].COL =
                static_cast<uint32_t*>(Util::MemoryManager::GetMemoryHost(
                    srcArrayBuffer[idx].NNZ * sizeof(uint32_t)));
            hostDstArray[idx].V =
                static_cast<float*>(Util::MemoryManager::GetMemoryHost(
                    srcArrayBuffer[idx].NNZ * sizeof(float)));
        }
        if (hostDstArray[idx].M != srcArrayBuffer[idx].M)
        {
            Util::MemoryManager::DeReferenceHost(hostDstArray[idx].ROW);
            hostDstArray[idx].ROW =
                static_cast<uint32_t*>(Util::MemoryManager::GetMemoryHost(
                    (hostDstArray[idx].M + 1) * sizeof(uint32_t)));
        }

        hostDstArray[idx].M = srcArrayBuffer[idx].M;
        hostDstArray[idx].N = srcArrayBuffer[idx].N;
        hostDstArray[idx].NNZ = srcArrayBuffer[idx].NNZ;

        Cuda::CopyGpuToHost(hostDstArray[idx].COL, gpuSrcArray[idx].COL,
                            hostDstArray[idx].NNZ * sizeof(uint32_t));
        Cuda::CopyGpuToHost(hostDstArray[idx].V, gpuSrcArray[idx].V,
                            hostDstArray[idx].NNZ * sizeof(float));
        Cuda::CopyGpuToHost(hostDstArray[idx].ROW, gpuSrcArray[idx].ROW,
                            (hostDstArray[idx].M + 1) * sizeof(uint32_t));
    }

    MemoryManager::DeReferenceHost(srcArrayBuffer);
}

void DeepCopyGpuToHost(LoadDistMatrix* hostDstArray,
                       LoadDistMatrix* gpuSrcArray, uint32_t numMatrices)
{
    auto* srcArrayBuffer = static_cast<LoadDistMatrix*>(
        MemoryManager::GetMemoryHost(numMatrices * sizeof(LoadDistMatrix)));

    CopyGpuToHost(srcArrayBuffer, gpuSrcArray,
                  numMatrices * sizeof(LoadDistMatrix));

    for (int idx = 0; idx < numMatrices; ++idx)
    {
        if (hostDstArray[idx].NNZ != srcArrayBuffer[idx].NNZ)
        {
            Util::MemoryManager::DeReferenceHost(hostDstArray[idx].COL);
            Util::MemoryManager::DeReferenceHost(hostDstArray[idx].Load);
            hostDstArray[idx].COL =
                static_cast<uint32_t*>(Util::MemoryManager::GetMemoryHost(
                    srcArrayBuffer[idx].NNZ * sizeof(uint32_t)));
            hostDstArray[idx].Load =
                static_cast<uint32_t*>(Util::MemoryManager::GetMemoryHost(
                    srcArrayBuffer[idx].NNZ * sizeof(uint32_t)));
        }
        if (hostDstArray[idx].M != srcArrayBuffer[idx].M)
        {
            Util::MemoryManager::DeReferenceHost(hostDstArray[idx].ROW);
            hostDstArray[idx].ROW =
                static_cast<uint32_t*>(Util::MemoryManager::GetMemoryHost(
                    (hostDstArray[idx].M + 1) * sizeof(uint32_t)));
        }

        hostDstArray[idx].M = srcArrayBuffer[idx].M;
        hostDstArray[idx].N = srcArrayBuffer[idx].N;
        hostDstArray[idx].NNZ = srcArrayBuffer[idx].NNZ;

        Cuda::CopyGpuToHost(hostDstArray[idx].COL, gpuSrcArray[idx].COL,
                            hostDstArray[idx].NNZ * sizeof(uint32_t));
        Cuda::CopyGpuToHost(hostDstArray[idx].Load, gpuSrcArray[idx].Load,
                            hostDstArray[idx].NNZ * sizeof(uint32_t));
        Cuda::CopyGpuToHost(hostDstArray[idx].ROW, gpuSrcArray[idx].ROW,
                            (hostDstArray[idx].M + 1) * sizeof(uint32_t));
    }

    MemoryManager::DeReferenceHost(srcArrayBuffer);
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

void LaunchSparseGemm(SparseMatrix* C, SparseMatrix* A, SparseMatrix* B,
                      uint32_t numMatrices, uint32_t numRows, uint32_t numCols,
                      int deviceId)
{
    LoadDistMatrix* loadDiffHost = nullptr;
    LoadDistMatrix* loadDiffCuda = nullptr;

    DeepAllocateLoadDistMatrixHost(&loadDiffHost, A, numMatrices);
    DeepAllocateLoadDistCuda(&loadDiffCuda, loadDiffHost, numMatrices,
                             deviceId);


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