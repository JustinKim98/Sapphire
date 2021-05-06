// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/Tests/SparseGemmTest.hpp>
#include <Motutapu/Tests/SparseMemoryTest.hpp>
#include <Motutapu/compute/cuda/sparse/SparseGemm.cuh>
#include <iostream>
#include <random>

namespace Motutapu::Test
{
void PrintSparseMatrix(SparseMatrix *sparseMatrix)
{
    std::cout << "# rows : " << sparseMatrix->M
              << " # columns : " << sparseMatrix->N
              << " # nnz : " << sparseMatrix->NNZ << std::endl;
    for (uint32_t rowIdx = 0; rowIdx < sparseMatrix->M; ++rowIdx)
    {
        const auto sparseColIdxBegin = sparseMatrix->ROW[rowIdx];
        const auto sparseColIdxEnd = sparseMatrix->ROW[rowIdx + 1];
        for (uint32_t sparseColIdx = sparseColIdxBegin;
             sparseColIdx < sparseColIdxEnd; ++sparseColIdx)
        {
            const auto colIdx = sparseMatrix->COL[sparseColIdx];
            const auto value = sparseMatrix->V[sparseColIdx];
            std::cout << "rowIdx : " << rowIdx
                      << " rowOffset : " << sparseMatrix->ROW[rowIdx]
                      << " colIdx : " << colIdx << " value : " << value
                      << std::endl;
        }
    }

    std::cout << "rowOffset : " << sparseMatrix->ROW[sparseMatrix->M]
              << std::endl;
}

void PrintLoadDistMatrix(LoadDistMatrix *loadDistMatrix)
{
    std::cout << "# rows : " << loadDistMatrix->M
              << " # columns : " << loadDistMatrix->N
              << " # nnz : " << loadDistMatrix->NNZ << std::endl;
    for (uint32_t rowIdx = 0; rowIdx < loadDistMatrix->M; ++rowIdx)
    {
        const auto sparseColIdxBegin = loadDistMatrix->ROW[rowIdx];
        const auto sparseColIdxEnd = loadDistMatrix->ROW[rowIdx + 1];
        for (uint32_t sparseColIdx = sparseColIdxBegin;
             sparseColIdx < sparseColIdxEnd; ++sparseColIdx)
        {
            const auto colIdx = loadDistMatrix->COL[sparseColIdx];
            const auto value = loadDistMatrix->Load[sparseColIdx];
            std::cout << "rowIdx : " << rowIdx
                      << " rowOffset : " << loadDistMatrix->ROW[rowIdx]
                      << " colIdx : " << colIdx << " load : " << value
                      << std::endl;
        }
    }

    std::cout << "rowOffset : " << loadDistMatrix->ROW[loadDistMatrix->M]
              << std::endl;
}

void LoadDistTest()
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
    Compute::Cuda::Sparse::GetLoadDist(loadDist, hostA, cudaA, cudaB, m,
                                       numMatrices, 0);

    CHECK_EQ(loadDist->ROW[loadDist->M], loadDist->NNZ);

    for (uint32_t i = 0; i < numMatrices; ++i)
        PrintLoadDistMatrix(loadDist + i);
}

void LoadDistTestFixed()
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
    Compute::Cuda::Sparse::GetLoadDist(loadDist, hostA, cudaA, cudaB, m,
                                       numMatrices, 0);

    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        CHECK_EQ(loadDist->NNZ, loadDist->ROW[loadDist->M]);
        PrintLoadDistMatrix(loadDist + i);
    }

    Util::MemoryManager::ClearHostMemoryPool();
    Util::MemoryManager::ClearCudaMemoryPool();
}

void SparseGemmTest()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> uniform(1, 10);
    std::normal_distribution<float> normal(0, 100);

    SparseMatrix *A, *B, *C = nullptr;
    SparseMatrix *cudaA, *cudaB, *cudaC = nullptr;
    const uint32_t numMatrices = 2;  // uniform(gen) % 5 + 1;
    const auto m = 10;               // m = uniform(gen);
    const auto n = 10;               // n = uniform(gen);
    const auto k = 10;               // k = uniform(gen);

    GenerateRandomSparseArray(&A, m, k, numMatrices);
    GenerateRandomSparseArray(&B, k, n, numMatrices);

    Compute::DeepAllocateSparseCuda(&cudaA, A, numMatrices, 0);
    Compute::DeepAllocateSparseCuda(&cudaB, B, numMatrices, 0);

    Compute::DeepCopyHostToDevice(cudaA, A, numMatrices, 0);
    Compute::DeepCopyHostToDevice(cudaB, B, numMatrices, 0);

    Compute::Cuda::Sparse::Gemm(&C, &cudaC, A, cudaA, cudaB, m, n, numMatrices,
                                0, true);

    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        CHECK_EQ(C->NNZ, C->ROW[C->M]);
        PrintSparseMatrix(C + i);
    }

    Compute::DeepFreeSparseHost(C, numMatrices);
    Compute::DeepFreeSparseCuda(cudaC, numMatrices, 0);

    Util::MemoryManager::ClearHostMemoryPool();
    Util::MemoryManager::ClearCudaMemoryPool();
}

}  // namespace Motutapu::Test