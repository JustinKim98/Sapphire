// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Tests/SparseGemmTest.hpp>
#include <Sapphire/Tests/SparseMemoryTest.hpp>
#include <Sapphire/Tests/TestUtil.hpp>
#include <Sapphire/compute/dense/cuda/Gemm.cuh>
#include <Sapphire/compute/dense/naive/NaiveGemm.hpp>
#include <Sapphire/compute/sparse/cuda/SparseGemm.cuh>
#include <Sapphire/compute/sparse/cuda/cuSparseGemm.cuh>
#include <Sapphire/compute/sparse/naive/SparseGemm.hpp>
#include <chrono>
#include <iostream>
#include <random>

namespace Sapphire::Test
{
void PrintSparseMatrix(SparseMatrix *sparseMatrix, bool printVerbose)
{
    std::cout << "# rows : " << sparseMatrix->M
              << " # columns : " << sparseMatrix->N
              << " # nnz : " << sparseMatrix->NNZ << std::endl;
    for (uint32_t rowIdx = 0; rowIdx < sparseMatrix->M; ++rowIdx)
    {
        const auto sparseColIdxBegin = sparseMatrix->ROW[rowIdx];
        const auto sparseColIdxEnd = sparseMatrix->ROW[rowIdx + 1];
        uint32_t colIdxPrev = 0;
        for (uint32_t sparseColIdx = sparseColIdxBegin;
             sparseColIdx < sparseColIdxEnd; ++sparseColIdx)
        {
            const auto colIdx = sparseMatrix->COL[sparseColIdx];
            CHECK(colIdx >= colIdxPrev);
            colIdxPrev = colIdx;
            const auto value = sparseMatrix->V[sparseColIdx];

            if (printVerbose)
                std::cout << "rowIdx : " << rowIdx
                          << " rowOffset : " << sparseMatrix->ROW[rowIdx]
                          << " colIdx : " << colIdx << " value : " << value
                          << std::endl;
        }
    }
    CHECK_EQ(sparseMatrix->NNZ, sparseMatrix->ROW[sparseMatrix->M]);
    if (printVerbose)
        std::cout << "rowOffset : " << sparseMatrix->ROW[sparseMatrix->M]
                  << std::endl;
}

void PrintLoadDistMatrix(LoadDistMatrix *loadDistMatrix, bool printVerbose)
{
    std::cout << "# rows : " << loadDistMatrix->M
              << " # columns : " << loadDistMatrix->N
              << " # nnz : " << loadDistMatrix->NNZ << std::endl;
    for (uint32_t rowIdx = 0; rowIdx < loadDistMatrix->M; ++rowIdx)
    {
        const auto sparseColIdxBegin = loadDistMatrix->ROW[rowIdx];
        const auto sparseColIdxEnd = loadDistMatrix->ROW[rowIdx + 1];

        uint32_t colIdxPrev = 0;
        for (uint32_t sparseColIdx = sparseColIdxBegin;
             sparseColIdx < sparseColIdxEnd; ++sparseColIdx)
        {
            const auto colIdx = loadDistMatrix->COL[sparseColIdx];
            const auto value = loadDistMatrix->Load[sparseColIdx];
            CHECK(colIdx >= colIdxPrev);
            colIdxPrev = colIdx;
            if (printVerbose)
                std::cout << "rowIdx : " << rowIdx
                          << " rowOffset : " << loadDistMatrix->ROW[rowIdx]
                          << " colIdx : " << colIdx << " load : " << value
                          << std::endl;
        }
    }

    CHECK_EQ(loadDistMatrix->NNZ, loadDistMatrix->ROW[loadDistMatrix->M]);
    if (printVerbose)
        std::cout << "rowOffset : " << loadDistMatrix->ROW[loadDistMatrix->M]
                  << std::endl;
}

void LoadDistTest(bool printVerbose)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> uniform(1, 50);
    std::normal_distribution<float> normal(0, 100);

    SparseMatrix *hostA, *hostB;
    SparseMatrix *cudaA, *cudaB;
    const uint32_t numMatrices = uniform(gen) % 3 + 1;
    const auto m = uniform(gen);
    const auto n = uniform(gen);
    const auto k = uniform(gen);

    GenerateRandomSparseArray(&hostA, m, k, numMatrices);
    GenerateRandomSparseArray(&hostB, k, n, numMatrices);

    Compute::DeepAllocateSparseCuda(&cudaA, hostA, numMatrices, 0);
    Compute::DeepAllocateSparseCuda(&cudaB, hostB, numMatrices, 0);

    Compute::DeepCopyHostToDevice(cudaA, hostA, numMatrices, 0);
    Compute::DeepCopyHostToDevice(cudaB, hostB, numMatrices, 0);

    LoadDistMatrix *loadDist;
    Compute::DeepAllocateLoadDistHost(&loadDist, hostA, numMatrices);
    Compute::Sparse::Cuda::GetLoadDist(loadDist, cudaA, cudaB, m, numMatrices,
                                       0);

    CHECK_EQ(loadDist->ROW[loadDist->M], loadDist->NNZ);

    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        std::cout << "\nMatrix " << i << std::endl;
        PrintLoadDistMatrix(loadDist + i, printVerbose);
    }
}

void LoadDistTestFixed(bool printVerbose)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> uniform(1, 50);
    std::normal_distribution<float> normal(0, 100);

    SparseMatrix *hostA, *hostB;
    SparseMatrix *cudaA, *cudaB;
    const uint32_t numMatrices = 1;
    const auto m = 5;
    const auto n = 5;
    const auto k = 5;

    GenerateFixedSparseArray(&hostA, m, k, numMatrices);
    GenerateFixedSparseArray(&hostB, k, n, numMatrices);

    Compute::DeepAllocateSparseCuda(&cudaA, hostA, numMatrices, 0);
    Compute::DeepAllocateSparseCuda(&cudaB, hostB, numMatrices, 0);

    Compute::DeepCopyHostToDevice(cudaA, hostA, numMatrices, 0);
    Compute::DeepCopyHostToDevice(cudaB, hostB, numMatrices, 0);

    LoadDistMatrix *loadDist;
    Compute::DeepAllocateLoadDistHost(&loadDist, hostA, numMatrices);
    Compute::Sparse::Cuda::GetLoadDist(loadDist, cudaA, cudaB, m, numMatrices,
                                       0);

    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        std::cout << "\nMatrix " << i << std::endl;
        PrintLoadDistMatrix(loadDist + i, printVerbose);
    }

    Util::MemoryManager::ClearHostMemoryPool();
    Util::MemoryManager::ClearCudaMemoryPool();
}

long SparseGemmTestSimple(bool printVerbose)
{
    SparseMatrix *A, *B, *C = nullptr;
    SparseMatrix *cudaA, *cudaB, *cudaC = nullptr;
    const uint32_t numMatrices = 2;
    const auto m = 10;
    const auto n = 5;
    const auto k = 7;

    GenerateFixedSparseArray(&A, m, k, numMatrices);
    GenerateFixedSparseArray(&B, k, n, numMatrices);

    Compute::DeepAllocateSparseCuda(&cudaA, A, numMatrices, 0);
    Compute::DeepAllocateSparseCuda(&cudaB, B, numMatrices, 0);

    Compute::DeepCopyHostToDevice(cudaA, A, numMatrices, 0);
    Compute::DeepCopyHostToDevice(cudaB, B, numMatrices, 0);

    auto start = std::chrono::system_clock::now();
    Compute::Sparse::Cuda::Gemm(&C, &cudaC, cudaA, cudaB, m, n, numMatrices, 0,
                                true);
    auto end = std::chrono::system_clock::now();

    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        std::cout << "\nMatrix " << i << std::endl;
        PrintSparseMatrix(C + i, printVerbose);
    }

    Compute::DeepFreeSparseHost(C, numMatrices);
    Compute::DeepFreeSparseCuda(cudaC, numMatrices, 0);

    Util::MemoryManager::ClearHostMemoryPool();
    Util::MemoryManager::ClearCudaMemoryPool();

    auto elapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    return elapsedTime;
}

long SparseGemmTestComplex(bool printVerbose, size_t minimumNumMatrices)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> uniform(32, 64);

    SparseMatrix *A, *B, *C = nullptr;
    SparseMatrix *cudaA, *cudaB, *cudaC = nullptr;
    const uint32_t numMatrices = uniform(gen) % 50 + minimumNumMatrices;
    const auto m = uniform(gen);
    const auto n = uniform(gen);
    const auto k = uniform(gen);

    GenerateRandomSparseArray(&A, m, k, numMatrices);
    GenerateRandomSparseArray(&B, k, n, numMatrices);

    Compute::DeepAllocateSparseCuda(&cudaA, A, numMatrices, 0);
    Compute::DeepAllocateSparseCuda(&cudaB, B, numMatrices, 0);

    Compute::DeepCopyHostToDevice(cudaA, A, numMatrices, 0);
    Compute::DeepCopyHostToDevice(cudaB, B, numMatrices, 0);

    auto start = std::chrono::system_clock::now();
    Compute::Sparse::Cuda::Gemm(&C, &cudaC, cudaA, cudaB, m, n, numMatrices, 0,
                                true);
    auto end = std::chrono::system_clock::now();

    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        std::cout << "\nMatrix " << i << std::endl;
        PrintSparseMatrix(C + i, printVerbose);
    }

    Compute::DeepFreeSparseHost(C, numMatrices);
    Compute::DeepFreeSparseCuda(cudaC, numMatrices, 0);

    Util::MemoryManager::ClearHostMemoryPool();
    Util::MemoryManager::ClearCudaMemoryPool();

    auto elapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    return elapsedTime;
}

long SparseGemmTestCorrectness(bool printVerbose, size_t minimumNumMatrices)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> uniform(32, 64);

    SparseMatrix *A, *B, *C = nullptr;
    SparseMatrix *cudaA, *cudaB, *cudaC = nullptr;
    const uint32_t numMatrices = uniform(gen) % 10 + minimumNumMatrices;
    const auto m = uniform(gen);
    const auto n = uniform(gen);
    const auto k = uniform(gen);

    GenerateRandomSparseArray(&A, m, k, numMatrices);
    GenerateRandomSparseArray(&B, k, n, numMatrices);

    Compute::DeepAllocateSparseCuda(&cudaA, A, numMatrices, 0);
    Compute::DeepAllocateSparseCuda(&cudaB, B, numMatrices, 0);

    Compute::DeepCopyHostToDevice(cudaA, A, numMatrices, 0);
    Compute::DeepCopyHostToDevice(cudaB, B, numMatrices, 0);

    auto start = std::chrono::system_clock::now();
    Compute::Sparse::Cuda::Gemm(&C, &cudaC, cudaA, cudaB, m, n, numMatrices, 0,
                                true);
    auto end = std::chrono::system_clock::now();

    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        std::cout << "\nMatrix " << i << std::endl;
        PrintSparseMatrix(C + i, printVerbose);
    }

    Compute::DeepFreeSparseHost(C, numMatrices);
    Compute::DeepFreeSparseCuda(cudaC, numMatrices, 0);

    Util::MemoryManager::ClearHostMemoryPool();
    Util::MemoryManager::ClearCudaMemoryPool();

    auto elapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    return elapsedTime;
}

void SparseTestCorrectnessCuda(size_t m, size_t n, size_t k, size_t numMatrices,
                               float sparsity, bool printResult)
{
    size_t paddedK = k;
    size_t paddedN = n;
    auto *hostDenseA = static_cast<float *>(Util::MemoryManager::GetMemoryHost(
        sizeof(float) * m * paddedK * numMatrices));
    auto *hostDenseB = static_cast<float *>(Util::MemoryManager::GetMemoryHost(
        sizeof(float) * k * paddedN * numMatrices));
    auto *hostDenseOut =
        static_cast<float *>(Util::MemoryManager::GetMemoryHost(
            sizeof(float) * m * paddedN * numMatrices));
    auto *hostSparseConverted =
        static_cast<float *>(Util::MemoryManager::GetMemoryHost(
            sizeof(float) * m * paddedN * numMatrices));

    auto *cudaDenseA = static_cast<float *>(Util::MemoryManager::GetMemoryCuda(
        sizeof(float) * m * k * numMatrices, 0));
    auto *cudaDenseB = static_cast<float *>(Util::MemoryManager::GetMemoryCuda(
        sizeof(float) * k * n * numMatrices, 0));
    auto *cudaDenseOut =
        static_cast<float *>(Util::MemoryManager::GetMemoryCuda(
            sizeof(float) * m * n * numMatrices, 0));

    InitIntegerDenseMatrix(hostDenseA, m, k, paddedK, numMatrices, sparsity);
    InitIntegerDenseMatrix(hostDenseB, k, n, paddedN, numMatrices, sparsity);

#pragma omp parallel for default(none) \
    shared(hostDenseOut, m, paddedN, numMatrices) schedule(static)
    for (size_t i = 0; i < m * paddedN * numMatrices; ++i)
        hostDenseOut[i] = 0.0f;
    //! Copy data to device
    Compute::Cuda::CopyHostToDevice(cudaDenseA, hostDenseA,
                                    sizeof(float) * m * k * numMatrices);
    Compute::Cuda::CopyHostToDevice(cudaDenseB, hostDenseB,
                                    sizeof(float) * k * n * numMatrices);
    Compute::Cuda::CopyHostToDevice(cudaDenseOut, hostDenseOut,
                                    sizeof(float) * m * n * numMatrices);

    SparseMatrix *hostSparseA = nullptr, *hostSparseB = nullptr,
                 *hostSparseOut = nullptr;
    SparseMatrix *cudaSparseA = nullptr, *cudaSparseB = nullptr,
                 *cudaSparseOut = nullptr;

    Compute::CreateSparseMatrixWithDenseMatrix(&hostSparseA, hostDenseA, m, k,
                                               paddedK, numMatrices);
    Compute::CreateSparseMatrixWithDenseMatrix(&hostSparseB, hostDenseB, k, n,
                                               paddedN, numMatrices);

    Compute::DeepAllocateSparseCuda(&cudaSparseA, hostSparseA, numMatrices, 0);
    Compute::DeepAllocateSparseCuda(&cudaSparseB, hostSparseB, numMatrices, 0);

    Compute::DeepCopyHostToDevice(cudaSparseA, hostSparseA, numMatrices, 0);
    Compute::DeepCopyHostToDevice(cudaSparseB, hostSparseB, numMatrices, 0);

    cublasHandle_t handle;
    cublasCreate(&handle);
    Compute::Dense::Cuda::Gemm(m * n * numMatrices, cudaDenseOut, cudaDenseA,
                               cudaDenseB, cudaDenseOut, m, n, k, &handle);
    cublasDestroy(handle);

    Compute::Sparse::Cuda::Gemm(&hostSparseOut, &cudaSparseOut, cudaSparseA,
                                cudaSparseB, m, n, numMatrices, 0, true);

    Compute::Cuda::CopyDeviceToHost(hostDenseOut, cudaDenseOut,
                                    sizeof(float) * m * n * numMatrices);

    Compute::ConvertSparseMatrixToDenseMatrix(
        hostSparseConverted, hostSparseOut, m, n, paddedN, numMatrices);

    for (uint32_t matrixIdx = 0; matrixIdx < numMatrices; ++matrixIdx)
        for (uint32_t rowIdx = 0; rowIdx < m; ++rowIdx)
            for (uint32_t colIdx = 0; colIdx < n; ++colIdx)
            {
                const auto index =
                    matrixIdx * m * paddedN + rowIdx * paddedN + colIdx;
                const auto denseResult = hostDenseOut[index];
                const auto sparseResult = hostSparseConverted[index];

                CHECK_EQ(denseResult, sparseResult);

                if (printResult)
                    std::cout << "matrix : " << matrixIdx << " row : " << rowIdx
                              << " col : " << colIdx
                              << " dense : " << denseResult
                              << " sparse : " << sparseResult << std::endl;
            }
    Util::MemoryManager::ClearCudaMemoryPool();
    Util::MemoryManager::ClearHostMemoryPool();
}

void SparseTestCorrectnessHost(size_t m, size_t n, size_t k, size_t numMatrices,
                               float sparsity, bool printResult)
{
    size_t paddedK = k;
    size_t paddedN = n;
    auto *hostDenseA = static_cast<float *>(Util::MemoryManager::GetMemoryHost(
        sizeof(float) * m * paddedK * numMatrices));
    auto *hostDenseB = static_cast<float *>(Util::MemoryManager::GetMemoryHost(
        sizeof(float) * k * paddedN * numMatrices));
    auto *hostDenseOut =
        static_cast<float *>(Util::MemoryManager::GetMemoryHost(
            sizeof(float) * m * paddedN * numMatrices));
    auto *hostSparseConverted =
        static_cast<float *>(Util::MemoryManager::GetMemoryHost(
            sizeof(float) * m * paddedN * numMatrices));

    InitIntegerDenseMatrix(hostDenseA, m, k, paddedK, numMatrices, sparsity);
    InitIntegerDenseMatrix(hostDenseB, k, n, paddedN, numMatrices, sparsity);

#pragma omp parallel for default(none) \
    shared(hostDenseOut, m, paddedN, numMatrices) schedule(static)
    for (size_t i = 0; i < m * paddedN * numMatrices; ++i)
        hostDenseOut[i] = 0.0f;

    SparseMatrix *hostSparseA = nullptr, *hostSparseB = nullptr,
                 *hostSparseOut = nullptr;

    Compute::CreateSparseMatrixWithDenseMatrix(&hostSparseA, hostDenseA, m, k,
                                               paddedK, numMatrices);
    Compute::CreateSparseMatrixWithDenseMatrix(&hostSparseB, hostDenseB, k, n,
                                               paddedN, numMatrices);

    Compute::Dense::Naive::NaiveGemm(m * paddedN * numMatrices, hostDenseOut,
                                     hostDenseA, hostDenseB, hostDenseOut, m, n,
                                     paddedN, k, paddedK);

    Compute::Sparse::Naive::Gemm(&hostSparseOut, hostSparseA, hostSparseB, m, n,
                                 numMatrices);

    Compute::ConvertSparseMatrixToDenseMatrix(
        hostSparseConverted, hostSparseOut, m, n, paddedN, numMatrices);

    for (uint32_t matrixIdx = 0; matrixIdx < numMatrices; ++matrixIdx)
        for (uint32_t rowIdx = 0; rowIdx < m; ++rowIdx)
            for (uint32_t colIdx = 0; colIdx < n; ++colIdx)
            {
                const auto index =
                    matrixIdx * m * paddedN + rowIdx * paddedN + colIdx;
                const auto denseResult = hostDenseOut[index];
                const auto sparseResult = hostSparseConverted[index];

                CHECK_EQ(denseResult, sparseResult);

                if (printResult)
                    std::cout << "matrix : " << matrixIdx << " row : " << rowIdx
                              << " col : " << colIdx
                              << " dense : " << denseResult
                              << " sparse : " << sparseResult << std::endl;
            }
    Util::MemoryManager::ClearHostMemoryPool();
}

void SparseMatrixConversionTest(size_t m, size_t n, size_t numMatrices,
                                float sparsity, bool printResult)
{
    size_t paddedN = n;
    auto *hostDenseOrigin =
        static_cast<float *>(Util::MemoryManager::GetMemoryHost(
            sizeof(float) * m * paddedN * numMatrices));

    auto *hostSparseConverted =
        static_cast<float *>(Util::MemoryManager::GetMemoryHost(
            sizeof(float) * m * paddedN * numMatrices));

    InitIntegerDenseMatrix(hostDenseOrigin, m, n, paddedN, numMatrices,
                           sparsity);

    SparseMatrix *convertedSparse;
    Compute::CreateSparseMatrixWithDenseMatrix(
        &convertedSparse, hostDenseOrigin, m, n, paddedN, numMatrices);

    Compute::ConvertSparseMatrixToDenseMatrix(
        hostSparseConverted, convertedSparse, m, n, paddedN, numMatrices);

    for (uint32_t matrixIdx = 0; matrixIdx < numMatrices; ++matrixIdx)
        for (uint32_t rowIdx = 0; rowIdx < m; ++rowIdx)
            for (uint32_t colIdx = 0; colIdx < n; ++colIdx)
            {
                const auto index =
                    matrixIdx * m * paddedN + rowIdx * paddedN + colIdx;
                const auto denseResult = hostDenseOrigin[index];
                const auto sparseResult = hostSparseConverted[index];

                CHECK_EQ(denseResult, sparseResult);

                if (printResult)
                    std::cout << "matrix : " << matrixIdx << " row : " << rowIdx
                              << " col : " << colIdx
                              << " dense : " << denseResult
                              << " sparse : " << sparseResult << std::endl;
            }

    Util::MemoryManager::ClearHostMemoryPool();
}

void PerformanceTest(size_t m, size_t n, size_t k, size_t numMatrices,
                     float sparsity)
{
    size_t paddedK = k;
    size_t paddedN = n;
    auto *hostDenseA = static_cast<float *>(Util::MemoryManager::GetMemoryHost(
        sizeof(float) * m * paddedK * numMatrices));
    auto *hostDenseB = static_cast<float *>(Util::MemoryManager::GetMemoryHost(
        sizeof(float) * k * paddedN * numMatrices));
    auto *hostDenseOut =
        static_cast<float *>(Util::MemoryManager::GetMemoryHost(
            sizeof(float) * m * paddedN * numMatrices));

    auto *cudaDenseA = static_cast<float *>(Util::MemoryManager::GetMemoryCuda(
        sizeof(float) * m * k * numMatrices, 0));
    auto *cudaDenseB = static_cast<float *>(Util::MemoryManager::GetMemoryCuda(
        sizeof(float) * k * n * numMatrices, 0));
    auto *cudaDenseOut =
        static_cast<float *>(Util::MemoryManager::GetMemoryCuda(
            sizeof(float) * m * n * numMatrices, 0));

    InitIntegerDenseMatrix(hostDenseA, m, k, paddedK, numMatrices, sparsity);
    InitIntegerDenseMatrix(hostDenseB, k, n, paddedN, numMatrices, sparsity);

#pragma omp parallel for default(none) \
    shared(hostDenseOut, m, paddedN, numMatrices) schedule(static)
    for (size_t i = 0; i < m * paddedN * numMatrices; ++i)
        hostDenseOut[i] = 0.0f;
    //! Copy data to device
    Compute::Cuda::CopyHostToDevice(cudaDenseA, hostDenseA,
                                    sizeof(float) * m * k * numMatrices);
    Compute::Cuda::CopyHostToDevice(cudaDenseB, hostDenseB,
                                    sizeof(float) * k * n * numMatrices);
    Compute::Cuda::CopyHostToDevice(cudaDenseOut, hostDenseOut,
                                    sizeof(float) * m * n * numMatrices);

    SparseMatrix *hostSparseA = nullptr, *hostSparseB = nullptr,
                 *hostSparseOut = nullptr;
    SparseMatrix *cudaSparseA = nullptr, *cudaSparseB = nullptr,
                 *cudaSparseOut = nullptr;

    Compute::CreateSparseMatrixWithDenseMatrix(&hostSparseA, hostDenseA, m, k,
                                               paddedK, numMatrices);
    Compute::CreateSparseMatrixWithDenseMatrix(&hostSparseB, hostDenseB, k, n,
                                               paddedN, numMatrices);

    Compute::DeepAllocateSparseCuda(&cudaSparseA, hostSparseA, numMatrices, 0);
    Compute::DeepAllocateSparseCuda(&cudaSparseB, hostSparseB, numMatrices, 0);

    Compute::DeepCopyHostToDevice(cudaSparseA, hostSparseA, numMatrices, 0);
    Compute::DeepCopyHostToDevice(cudaSparseB, hostSparseB, numMatrices, 0);

    auto naiveDenseBegin = std::chrono::system_clock::now();
    Compute::Dense::Naive::NaiveGemm(m * paddedN * numMatrices, hostDenseOut,
                                     hostDenseA, hostDenseB, hostDenseOut, m, n,
                                     paddedN, k, paddedK);
    auto naiveDenseEnd = std::chrono::system_clock::now();
    auto naiveDenseElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(naiveDenseEnd -
                                                              naiveDenseBegin)
            .count();

    auto cudaDenseBegin = std::chrono::system_clock::now();
    cublasHandle_t handle;
    cublasCreate(&handle);
    Compute::Dense::Cuda::Gemm(m * n * numMatrices, cudaDenseOut, cudaDenseA,
                               cudaDenseB, cudaDenseOut, m, n, k, &handle);
    cublasDestroy(handle);
    auto cudaDenseEnd = std::chrono::system_clock::now();

    auto naiveSparseBegin = std::chrono::system_clock::now();
    Compute::Sparse::Naive::Gemm(&hostSparseOut, hostSparseA, hostSparseB, m, n,
                                 numMatrices);
    auto naiveSparseEnd = std::chrono::system_clock::now();

    auto cuSparseBegin = std::chrono::system_clock::now();
    Compute::Sparse::Cuda::cuSparseGemm(&hostSparseOut, &cudaSparseOut,
                                        cudaSparseA, cudaSparseB, m, n,
                                        numMatrices, 0, false);
    auto cuSparseEnd = std::chrono::system_clock::now();

    auto cudaSparseBegin = std::chrono::system_clock::now();
    Compute::Sparse::Cuda::Gemm(&hostSparseOut, &cudaSparseOut, cudaSparseA,
                                cudaSparseB, m, n, numMatrices, 0, false);
    auto cudaSparseEnd = std::chrono::system_clock::now();

    auto cudaDenseElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(cudaDenseEnd -
                                                              cudaDenseBegin)
            .count();
    auto naiveSparseElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(naiveSparseEnd -
                                                              naiveSparseBegin)
            .count();
    auto cuSparseGemmElapsedTimeTotal =
        std::chrono::duration_cast<std::chrono::microseconds>(cuSparseEnd -
                                                              cuSparseBegin)
            .count();
    auto cudaSparseGemmElapsedTime =
        std::chrono::duration_cast<std::chrono::microseconds>(cudaSparseEnd -
                                                              cudaSparseBegin)
            .count();

    std::cout << "--- Results (time in microseconds) ---" << std::endl;
    std::cout << "* Sparsity : " << sparsity << std::endl;
    std::cout << "* Naive Dense : " << naiveDenseElapsedTime << std::endl;
    std::cout << "* Naive Sparse : " << naiveSparseElapsedTime << std::endl;
    std::cout << "* Cuda Dense : " << cudaDenseElapsedTime << std::endl;
    std::cout << "* Cuda Sparse (custom) : " << cudaSparseGemmElapsedTime
              << std::endl;
    std::cout << "* Cuda Sparse (cuSparse) : " << cuSparseGemmElapsedTimeTotal
              << std::endl;
    std::cout << "-------------------------------------\n" << std::endl;

    Util::MemoryManager::ClearHostMemoryPool();
    Util::MemoryManager::ClearCudaMemoryPool();
}

}  // namespace Sapphire::Test