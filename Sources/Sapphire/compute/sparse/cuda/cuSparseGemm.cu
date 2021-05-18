// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/cudaUtil/Memory.hpp>
#include <Sapphire/compute/sparse/cuda/cuSparseGemm.cuh>
#include <Sapphire/util/MemoryManager.hpp>
#include <iostream>

namespace Sapphire::Compute::Sparse::Cuda
{
using namespace Sapphire::Util;
size_t cuSparseGemm(SparseMatrix** hostOutput, SparseMatrix** cudaOutput,
                    SparseMatrix* cudaA, SparseMatrix* cudaB, uint32_t m,
                    uint32_t n, size_t numMatrices, int deviceId,
                    bool copyResultToHost)
{
    size_t totalElapsedTime;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    float alpha = 1.0f;
    float beta = 0.0f;
    cudaDataType computeType = CUDA_R_32F;

    *cudaOutput = static_cast<SparseMatrix*>(MemoryManager::GetMemoryCuda(
        sizeof(SparseMatrix) * numMatrices, deviceId));

    auto* hostBufferA = static_cast<SparseMatrix*>(
        MemoryManager::GetMemoryHost(sizeof(SparseMatrix) * numMatrices));
    auto* hostBufferB = static_cast<SparseMatrix*>(
        MemoryManager::GetMemoryHost(sizeof(SparseMatrix) * numMatrices));
    auto* outputBuffer = static_cast<SparseMatrix*>(
        MemoryManager::GetMemoryHost(sizeof(SparseMatrix) * numMatrices));

    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        outputBuffer[i].ROW = static_cast<uint32_t*>(
            MemoryManager::GetMemoryCuda((m + 1) * sizeof(uint32_t), deviceId));
    }

    Compute::Cuda::CopyDeviceToHost(hostBufferA, cudaA,
                                    sizeof(SparseMatrix) * numMatrices);
    Compute::Cuda::CopyDeviceToHost(hostBufferB, cudaB,
                                    sizeof(SparseMatrix) * numMatrices);

    for (size_t matrixIdx = 0; matrixIdx < numMatrices; ++matrixIdx)
    {
        const auto k = hostBufferA[matrixIdx].N;
        const auto nnzA = hostBufferA[matrixIdx].NNZ;
        const auto nnzB = hostBufferB[matrixIdx].NNZ;
        cusparseHandle_t handle = nullptr;
        cusparseSpMatDescr_t matA, matB, matOut;
        void *buffer1 = nullptr, *buffer2 = nullptr;
        size_t bufferSize1 = 0, bufferSize2 = 0;

        CHECK_CUSPARSE(cusparseCreate(&handle))
        CHECK_CUSPARSE(cusparseCreateCsr(
            &matA, m, k, nnzA, hostBufferA[matrixIdx].ROW,
            hostBufferA[matrixIdx].COL, hostBufferA[matrixIdx].V,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F))
        CHECK_CUSPARSE(cusparseCreateCsr(
            &matB, k, n, nnzB, hostBufferB[matrixIdx].ROW,
            hostBufferB[matrixIdx].COL, hostBufferB[matrixIdx].V,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F))
        CHECK_CUSPARSE(cusparseCreateCsr(
            &matOut, m, n, 0, nullptr, nullptr, nullptr, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))

        cusparseSpGEMMDescr_t spgemmDesc;
        CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc))
        auto cuSparseBegin = std::chrono::system_clock::now();

        CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(
            handle, opA, opB, &alpha, matA, matB, &beta, matOut, computeType,
            CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, nullptr))
        buffer1 = MemoryManager::GetMemoryCuda(bufferSize1, deviceId);
        CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(
            handle, opA, opB, &alpha, matA, matB, &beta, matOut, computeType,
            CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, buffer1))

        CHECK_CUSPARSE(cusparseSpGEMM_compute(
            handle, opA, opB, &alpha, matA, matB, &beta, matOut, computeType,
            CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, nullptr))
        buffer2 = MemoryManager::GetMemoryCuda(bufferSize2, 0);

        CHECK_CUSPARSE(cusparseSpGEMM_compute(
            handle, opA, opB, &alpha, matA, matB, &beta, matOut, computeType,
            CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, buffer2))

        int64_t outNumRows, outNumCols, OutNNZ;
        CHECK_CUSPARSE(
            cusparseSpMatGetSize(matOut, &outNumRows, &outNumCols, &OutNNZ))
        outputBuffer[matrixIdx].COL = static_cast<uint32_t*>(
            MemoryManager::GetMemoryCuda(OutNNZ * sizeof(uint32_t), deviceId));
        outputBuffer[matrixIdx].V = static_cast<float*>(
            MemoryManager::GetMemoryCuda(OutNNZ * sizeof(float), deviceId));

        CHECK_CUSPARSE(cusparseCsrSetPointers(
            matOut, outputBuffer[matrixIdx].ROW, outputBuffer[matrixIdx].COL,
            outputBuffer[matrixIdx].V))
        CHECK_CUSPARSE(cusparseSpGEMM_copy(handle, opA, opB, &alpha, matA, matB,
                                           &beta, matOut, computeType,
                                           CUSPARSE_SPGEMM_DEFAULT, spgemmDesc))
        auto cuSparseEnd = std::chrono::system_clock::now();

        CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDesc))
        CHECK_CUSPARSE(cusparseDestroySpMat(matA))
        CHECK_CUSPARSE(cusparseDestroySpMat(matB))
        CHECK_CUSPARSE(cusparseDestroySpMat(matOut))
        CHECK_CUSPARSE(cusparseDestroy(handle))

        totalElapsedTime +=
            std::chrono::duration_cast<std::chrono::microseconds>(cuSparseEnd -
                                                                  cuSparseBegin)
                .count();
    }

    Compute::Cuda::CopyHostToDevice(*cudaOutput, outputBuffer,
                                    sizeof(SparseMatrix) * numMatrices);

    if (copyResultToHost)
    {
        *hostOutput = static_cast<SparseMatrix*>(
            MemoryManager::GetMemoryHost(sizeof(SparseMatrix) * numMatrices));
        for (uint32_t i = 0; i < numMatrices; ++i)
        {
            (*hostOutput + i)->ROW = static_cast<uint32_t*>(
                MemoryManager::GetMemoryHost(sizeof(uint32_t) * (m + 1)));
            Compute::Cuda::CopyDeviceToHost((*hostOutput + i)->ROW,
                                            (outputBuffer + i)->ROW,
                                            sizeof(uint32_t) * (m + 1));

            const auto nnz = (*hostOutput + i)->ROW[m];
            (*hostOutput + i)->V = static_cast<float*>(
                MemoryManager::GetMemoryHost(sizeof(float) * nnz));
            (*hostOutput + i)->COL = static_cast<uint32_t*>(
                MemoryManager::GetMemoryHost(sizeof(float) * nnz));

            Compute::Cuda::CopyDeviceToHost((*hostOutput + i)->V,
                                            (outputBuffer + i)->V,
                                            sizeof(float) * nnz);
            Compute::Cuda::CopyDeviceToHost((*hostOutput + i)->COL,
                                            (outputBuffer + i)->COL,
                                            sizeof(uint32_t) * nnz);

            (*hostOutput + i)->M = m;
            (*hostOutput + i)->N = n;
            (*hostOutput + i)->NNZ = nnz;
        }
    }

    MemoryManager::DeReferenceHost(outputBuffer);
    MemoryManager::DeReferenceHost(hostBufferA);
    MemoryManager::DeReferenceHost(hostBufferB);

    return totalElapsedTime;
}

}  // namespace Sapphire::Compute::Sparse::Cuda