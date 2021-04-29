// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/Sparse.hpp>
#include <Motutapu/compute/SparseMatrix.hpp>
#include <Motutapu/compute/cuda/Memory.cuh>
#include <Motutapu/util/MemoryManager.hpp>

namespace Motutapu::Compute::Sparse
{
using namespace Util;

void DeepAllocateSparseHost(SparseMatrix** sparseMatrixArray, const uint32_t m,
                            const uint32_t n, const uint32_t nnz[],
                            uint32_t numMatrices)
{
    *sparseMatrixArray = static_cast<SparseMatrix*>(
        MemoryManager::GetMemoryHost(sizeof(SparseMatrix) * numMatrices));
    SparseMatrix* targetArray = *sparseMatrixArray;

    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        targetArray[i].M = m;
        targetArray[i].N = n;
        targetArray[i].NNZ = nnz[i];
        targetArray[i].V = static_cast<float*>(
            Util::MemoryManager::GetMemoryHost(nnz[i] * sizeof(float)));
        targetArray[i].COL = static_cast<uint32_t*>(
            Util::MemoryManager::GetMemoryHost(nnz[i] * sizeof(uint32_t)));
        targetArray[i].ROW = static_cast<uint32_t*>(
            Util::MemoryManager::GetMemoryHost((m + 1) * sizeof(uint32_t)));
    }
}

void DeepAllocateLoadDistHost(LoadDistMatrix** loadDistArray,
                              SparseMatrix* sparseArray, uint32_t numMatrices)
{
    *loadDistArray = static_cast<LoadDistMatrix*>(
        MemoryManager::GetMemoryHost(sizeof(LoadDistMatrix) * numMatrices));
    LoadDistMatrix* targetArray = *loadDistArray;

    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        targetArray[i].M = sparseArray[i].M;
        targetArray[i].N = sparseArray[i].N;
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

void DeepFreeSparseHost(SparseMatrix* sparseMatrixArray, uint32_t numMatrices)
{
    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        Util::MemoryManager::DeReferenceHost(
            static_cast<void*>(sparseMatrixArray[i].V));
        Util::MemoryManager::DeReferenceHost(
            static_cast<void*>(sparseMatrixArray[i].COL));
        Util::MemoryManager::DeReferenceHost(
            static_cast<void*>(sparseMatrixArray[i].ROW));
    }

    delete[] sparseMatrixArray;
}

void DeepFreeLoadDistHost(LoadDistMatrix* loadDistArray, uint32_t numMatrices)
{
    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        Util::MemoryManager::DeReferenceHost(
            static_cast<void*>(loadDistArray[i].Load));
        Util::MemoryManager::DeReferenceHost(
            static_cast<void*>(loadDistArray[i].COL));
        Util::MemoryManager::DeReferenceHost(
            static_cast<void*>(loadDistArray[i].ROW));
    }

    delete[] loadDistArray;
}

void DeepAllocateSparseCuda(SparseMatrix** deviceSparseMatrixArray,
                            SparseMatrix* hostSparseMatrixArray,
                            uint32_t numMatrices, int deviceId)
{
    Cuda::CudaMalloc((void**)(deviceSparseMatrixArray),
                     sizeof(SparseMatrix) * numMatrices);
    SparseMatrix* cudaTarget = *deviceSparseMatrixArray;
    Cuda::CopyHostToGpu(cudaTarget, hostSparseMatrixArray,
                        sizeof(SparseMatrix) * numMatrices);

    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        cudaTarget[i].V =
            static_cast<float*>(Util::MemoryManager::GetMemoryCuda(
                hostSparseMatrixArray[i].NNZ * sizeof(float), deviceId));
        cudaTarget[i].COL =
            static_cast<uint32_t*>(Util::MemoryManager::GetMemoryCuda(
                hostSparseMatrixArray[i].NNZ * sizeof(uint32_t), deviceId));
        cudaTarget[i].ROW =
            static_cast<uint32_t*>(Util::MemoryManager::GetMemoryHost(
                (hostSparseMatrixArray[i].M + 1) * sizeof(uint32_t)));
    }
}

void DeepAllocateLoadDistCuda(LoadDistMatrix** deviceLoadDistArray,
                              LoadDistMatrix* hostLoadDistArray,
                              uint32_t numMatrices, int deviceId)
{
    Cuda::CudaMalloc((void**)(deviceLoadDistArray),
                     sizeof(SparseMatrix) * numMatrices);
    LoadDistMatrix* cudaTarget = *deviceLoadDistArray;
    Cuda::CopyHostToGpu(cudaTarget, hostLoadDistArray,
                        sizeof(SparseMatrix) * numMatrices);

    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        cudaSetDevice(deviceId);
        cudaMalloc(&(cudaTarget[i].Load),
                   hostLoadDistArray[i].NNZ * sizeof(uint32_t));
        cudaMalloc(&(cudaTarget[i].COL),
                   hostLoadDistArray[i].NNZ * sizeof(uint32_t));
        cudaMalloc(&(cudaTarget[i].ROW),
                   (hostLoadDistArray[i].M + 1) * sizeof(uint32_t));
    }
}

void DeepFreeSparseCuda(SparseMatrix* sparseMatrixArray, uint32_t numMatrices,
                        int deviceId)
{
    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        cudaFree(sparseMatrixArray[i].V);
        cudaFree(sparseMatrixArray[i].COL);
        cudaFree(sparseMatrixArray[i].ROW);
    }
    Cuda::CudaFree(sparseMatrixArray);
}

void DeepFreeLoadDistCuda(LoadDistMatrix* loadDistArray, uint32_t numMatrices,
                          int deviceId)
{
    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        cudaFree(loadDistArray[i].Load);
        cudaFree(loadDistArray[i].COL);
        cudaFree(loadDistArray[i].ROW);
    }
    Cuda::CudaFree(loadDistArray);
}

void DeepCopyGpuToGpu(SparseMatrix* deviceDstArray,
                      SparseMatrix* deviceSrcArray, uint32_t numMatrices,
                      int deviceId)
{
    auto* dstArrayBuffer = static_cast<SparseMatrix*>(
        MemoryManager::GetMemoryHost(numMatrices * sizeof(SparseMatrix)));
    auto* srcArrayBuffer = static_cast<SparseMatrix*>(
        MemoryManager::GetMemoryHost(numMatrices * sizeof(SparseMatrix)));

    cudaSetDevice(deviceId);
    Cuda::CopyGpuToHost(dstArrayBuffer, deviceDstArray,
                        numMatrices * sizeof(SparseMatrix));
    Cuda::CopyGpuToHost(srcArrayBuffer, deviceSrcArray,
                        numMatrices * sizeof(SparseMatrix));

    for (uint32_t idx = 0; idx < numMatrices; ++idx)
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

    Cuda::CopyHostToGpu(deviceDstArray, dstArrayBuffer,
                        numMatrices * sizeof(SparseMatrix));

    MemoryManager::DeReferenceHost(dstArrayBuffer);
    MemoryManager::DeReferenceHost(dstArrayBuffer);
}

void DeepCopyGpuToGpu(LoadDistMatrix* deviceDstArray,
                      LoadDistMatrix* deviceSrcArray, uint32_t numMatrices,
                      int deviceId)
{
    auto* dstArrayBuffer = static_cast<LoadDistMatrix*>(
        MemoryManager::GetMemoryHost(numMatrices * sizeof(LoadDistMatrix)));
    auto* srcArrayBuffer = static_cast<LoadDistMatrix*>(
        MemoryManager::GetMemoryHost(numMatrices * sizeof(LoadDistMatrix)));

    cudaSetDevice(deviceId);
    Cuda::CopyGpuToHost(dstArrayBuffer, deviceDstArray,
                        numMatrices * sizeof(LoadDistMatrix));
    Cuda::CopyGpuToHost(srcArrayBuffer, deviceSrcArray,
                        numMatrices * sizeof(LoadDistMatrix));

    for (uint32_t idx = 0; idx < numMatrices; ++idx)
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

    Cuda::CopyHostToGpu(deviceDstArray, dstArrayBuffer,
                        numMatrices * sizeof(LoadDistMatrix));

    MemoryManager::DeReferenceHost(dstArrayBuffer);
    MemoryManager::DeReferenceHost(dstArrayBuffer);
}

void DeepCopyHostToGpu(SparseMatrix* deviceDstArray, SparseMatrix* hostSrcArray,
                       uint32_t numMatrices, int deviceId)
{
    auto* dstArrayBuffer = static_cast<SparseMatrix*>(
        MemoryManager::GetMemoryHost(numMatrices * sizeof(SparseMatrix)));

    cudaSetDevice(deviceId);
    Cuda::CopyGpuToHost(dstArrayBuffer, deviceDstArray,
                        numMatrices * sizeof(SparseMatrix));

    for (uint32_t idx = 0; idx < numMatrices; ++idx)
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

        Cuda::CopyHostToGpu(dstArrayBuffer[idx].V, hostSrcArray[idx].V,
                            sizeof(float) * nnz);
        Cuda::CopyHostToGpu(dstArrayBuffer[idx].COL, hostSrcArray[idx].COL,
                            sizeof(uint32_t) * nnz);
        Cuda::CopyHostToGpu(dstArrayBuffer[idx].ROW, hostSrcArray[idx].ROW,
                            sizeof(uint32_t) * (m + 1));
    }
    Cuda::CopyHostToGpu(deviceDstArray, dstArrayBuffer,
                        sizeof(SparseMatrix) * numMatrices);
    MemoryManager::DeReferenceHost(dstArrayBuffer);
}

void DeepCopyHostToGpu(LoadDistMatrix* deviceDstArray,
                       LoadDistMatrix* hostSrcArray, uint32_t numMatrices,
                       int deviceId)
{
    auto* dstArrayBuffer = static_cast<LoadDistMatrix*>(
        MemoryManager::GetMemoryHost(numMatrices * sizeof(LoadDistMatrix)));

    cudaSetDevice(deviceId);
    Cuda::CopyGpuToHost(dstArrayBuffer, deviceDstArray,
                        numMatrices * sizeof(LoadDistMatrix));

    for (uint32_t idx = 0; idx < numMatrices; ++idx)
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

        Cuda::CopyHostToGpu(dstArrayBuffer[idx].Load, hostSrcArray[idx].Load,
                            sizeof(uint32_t) * nnz);
        Cuda::CopyHostToGpu(dstArrayBuffer[idx].COL, hostSrcArray[idx].COL,
                            sizeof(uint32_t) * nnz);
        Cuda::CopyHostToGpu(dstArrayBuffer[idx].ROW, hostSrcArray[idx].ROW,
                            sizeof(uint32_t) * (m + 1));
    }
    Cuda::CopyHostToGpu(deviceDstArray, dstArrayBuffer,
                        sizeof(LoadDistMatrix) * numMatrices);
    MemoryManager::DeReferenceHost(dstArrayBuffer);
}

void DeepCopyGpuToHost(SparseMatrix* hostDstArray, SparseMatrix* deviceSrcArray,
                       uint32_t numMatrices, int deviceId)
{
    auto* srcArrayBuffer = static_cast<SparseMatrix*>(
        MemoryManager::GetMemoryHost(numMatrices * sizeof(SparseMatrix)));

    cudaSetDevice(deviceId);
    Cuda::CopyGpuToHost(srcArrayBuffer, deviceSrcArray,
                        numMatrices * sizeof(SparseMatrix));

    for (uint32_t idx = 0; idx < numMatrices; ++idx)
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

        Cuda::CopyGpuToHost(hostDstArray[idx].COL, deviceSrcArray[idx].COL,
                            hostDstArray[idx].NNZ * sizeof(uint32_t));
        Cuda::CopyGpuToHost(hostDstArray[idx].V, deviceSrcArray[idx].V,
                            hostDstArray[idx].NNZ * sizeof(float));
        Cuda::CopyGpuToHost(hostDstArray[idx].ROW, deviceSrcArray[idx].ROW,
                            (hostDstArray[idx].M + 1) * sizeof(uint32_t));
    }

    MemoryManager::DeReferenceHost(srcArrayBuffer);
}

void DeepCopyGpuToHost(LoadDistMatrix* hostDstArray,
                       LoadDistMatrix* deviceSrcArray, uint32_t numMatrices)
{
    auto* srcArrayBuffer = static_cast<LoadDistMatrix*>(
        MemoryManager::GetMemoryHost(numMatrices * sizeof(LoadDistMatrix)));

    Cuda::CopyGpuToHost(srcArrayBuffer, deviceSrcArray,
                        numMatrices * sizeof(LoadDistMatrix));

    for (uint32_t idx = 0; idx < numMatrices; ++idx)
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

        Cuda::CopyGpuToHost(hostDstArray[idx].COL, deviceSrcArray[idx].COL,
                            hostDstArray[idx].NNZ * sizeof(uint32_t));
        Cuda::CopyGpuToHost(hostDstArray[idx].Load, deviceSrcArray[idx].Load,
                            hostDstArray[idx].NNZ * sizeof(uint32_t));
        Cuda::CopyGpuToHost(hostDstArray[idx].ROW, deviceSrcArray[idx].ROW,
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

}  // namespace Motutapu::Compute::Sparse