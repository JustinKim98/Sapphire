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
    for (uint32_t rowIdx = 0; rowIdx < sparseMatrix->M; ++rowIdx)
    {
        const auto sparseColIdxBegin = sparseMatrix->ROW[rowIdx];
        const auto sparseColIdxEnd = sparseMatrix->ROW[rowIdx + 1];
        for (uint32_t sparseColIdx = sparseColIdxBegin;
             sparseColIdx < sparseColIdxEnd; ++sparseColIdx)
        {
            const auto colIdx = sparseMatrix->COL[sparseColIdx];
            const auto value = sparseMatrix->V[sparseColIdx];
            std::cout << "rowIdx : " << rowIdx << " colIdx : " << colIdx
                      << " vaue : " << value << std::endl;
        }
    }
}

void PrintLoadDistMatrix(LoadDistMatrix *loadDistMatrix)
{
    for (uint32_t rowIdx = 0; rowIdx < loadDistMatrix->M; ++rowIdx)
    {
        const auto sparseColIdxBegin = loadDistMatrix->ROW[rowIdx];
        const auto sparseColIdxEnd = loadDistMatrix->ROW[rowIdx + 1];
        for (uint32_t sparseColIdx = sparseColIdxBegin;
             sparseColIdx < sparseColIdxEnd; ++sparseColIdx)
        {
            const auto colIdx = loadDistMatrix->COL[sparseColIdx];
            const auto value = loadDistMatrix->Load[sparseColIdx];
            std::cout << "rowIdx : " << rowIdx << " colIdx : " << colIdx
                      << " load : " << value << std::endl;
        }
    }
}

void LoadDistTest()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> uniform(1, 50);
    std::normal_distribution<float> normal(0, 100);

    SparseMatrix *hostA, *hostB;
    SparseMatrix *cudaA, *cudaB;
    const uint32_t numMatrices = uniform(gen) % 10 + 1;
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

    for (uint32_t i = 0; i < numMatrices; ++i)
        PrintLoadDistMatrix(loadDist + i);
}

void SparseGemmTest()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> uniform(1, 10);
    std::normal_distribution<float> normal(0, 100);

    SparseMatrix *A, *B, *C = nullptr;
    SparseMatrix *cudaA, *cudaB, *cudaC = nullptr;
    const uint32_t numMatrices = uniform(gen) % 5 + 1;
    const auto m = uniform(gen);
    const auto n = uniform(gen);
    const auto k = uniform(gen);

    GenerateRandomSparseArray(&A, m, k, numMatrices);
    GenerateRandomSparseArray(&B, k, n, numMatrices);

    Compute::DeepAllocateSparseCuda(&cudaA, A, numMatrices, 0);
    Compute::DeepAllocateSparseCuda(&cudaB, B, numMatrices, 0);

    Compute::DeepCopyHostToDevice(cudaA, A, numMatrices, 0);
    Compute::DeepCopyHostToDevice(cudaB, B, numMatrices, 0);

    Compute::Cuda::Sparse::Gemm(&C, &cudaC, A, cudaA, cudaB, m, n, numMatrices,
                                0, true);

    for (uint32_t i = 0; i < numMatrices; ++i)
        PrintSparseMatrix(C + i);

    Compute::DeepFreeSparseHost(C, numMatrices);
    Compute::DeepFreeSparseCuda(cudaC, numMatrices, 0);
}

}  // namespace Motutapu::Test