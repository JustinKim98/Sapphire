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
                      << " value : " << value << std::endl;
        }
    }
}

void SparseGemmTest()
{
    static_assert(sizeof(SparseMatrix) == sizeof(LoadDistMatrix) &&
                  sizeof(SparseMatrix) == 48);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> uniform(1, 50);
    std::normal_distribution<float> normal(0, 100);

    SparseMatrix *A, *B, *C = nullptr;
    SparseMatrix *cudaA, *cudaB, *cudaC = nullptr;
    const uint32_t numMatrices = uniform(gen) % 10 + 1;
    const auto m = uniform(gen);
    const auto n = uniform(gen);
    const auto k = uniform(gen);

    GenerateRandomSparseArray(&A, m, k, numMatrices);
    GenerateRandomSparseArray(&B, m, k, numMatrices);

    Compute::DeepAllocateSparseCuda(&cudaA, A, numMatrices, 0);
    Compute::DeepAllocateSparseCuda(&cudaB, B, numMatrices, 0);

    Compute::DeepCopyHostToDevice(cudaA, A, numMatrices, 0);
    Compute::DeepCopyHostToDevice(cudaB, B, numMatrices, 0);

    Compute::Cuda::Sparse::Gemm(&C, &cudaC, A, cudaA, cudaB, m, n, numMatrices,
                                0, true);

    for (uint32_t i = 0; i < numMatrices; ++i)
        PrintSparseMatrix(C + i);
}

}  // namespace Motutapu::Test