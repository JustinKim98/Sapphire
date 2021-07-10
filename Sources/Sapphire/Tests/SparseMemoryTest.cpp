// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Tests/SparseMemoryTest.hpp>
#include <list>
#include <iostream>

namespace Sapphire::Test
{
using namespace Sapphire::Compute;

void GenerateFixedSparseArray(SparseMatrix** sparseMatrixArray, uint32_t m,
                              uint32_t n, uint32_t numMatrices)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> uniform(0, n);
    std::normal_distribution<float> normal(0, 10);

    //! Array containing NNZ for each matrix
    auto* nnz = new uint32_t[numMatrices];
    //! Array containing NNZ for each row in the matrix

    for (uint32_t i = 0; i < numMatrices; ++i)
        nnz[i] = m * n;

    DeepAllocateSparseHost(sparseMatrixArray, m, n, nnz, numMatrices);
    SparseMatrix* sparse = *sparseMatrixArray;
    for (uint32_t matrixIdx = 0; matrixIdx < numMatrices; ++matrixIdx)
    {
        sparse[matrixIdx].NNZ = nnz[matrixIdx];
        sparse[matrixIdx].M = m;
        sparse[matrixIdx].N = n;
        uint32_t curNNZ = 0;
        for (uint32_t rowIdx = 0; rowIdx < sparse[matrixIdx].M; ++rowIdx)
        {
            sparse[matrixIdx].ROW[rowIdx] = curNNZ;

            uint32_t colIdx = 0;
            for (uint32_t sparseColIdx = curNNZ; sparseColIdx < curNNZ + n;
                 ++sparseColIdx)
            {
                sparse[matrixIdx].COL[sparseColIdx] = colIdx++;
                sparse[matrixIdx].V[sparseColIdx] = 1;  // normal(gen);
            }
            curNNZ += n;
        }
        sparse[matrixIdx].ROW[m] = curNNZ;
        CHECK_EQ(curNNZ, nnz[matrixIdx]);
    }

    delete[] nnz;
}

void GenerateRandomSparseArray(SparseMatrix** sparseMatrixArray, uint32_t m,
                               uint32_t n, uint32_t numMatrices)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> uniform(0, n);
    std::normal_distribution<float> normal(0, 100);

    //! Array containing NNZ for each matrix
    auto* nnz = new uint32_t[numMatrices];
    //! Array containing NNZ for each row in the matrix
    auto* nnzArray = new uint32_t[numMatrices * m];

    for (uint32_t rowOffset = 0; rowOffset < numMatrices * m; rowOffset += m)
    {
        uint32_t curNNZ = 0;
        for (uint32_t colOffset = rowOffset; colOffset < rowOffset + m;
             ++colOffset)
        {
            nnzArray[colOffset] = uniform(gen);
            curNNZ += nnzArray[colOffset];
        }
        nnz[rowOffset / m] = curNNZ;
    }

    DeepAllocateSparseHost(sparseMatrixArray, m, n, nnz, numMatrices);
    SparseMatrix* sparse = *sparseMatrixArray;
    for (uint32_t matrixIdx = 0; matrixIdx < numMatrices; ++matrixIdx)
    {
        sparse[matrixIdx].NNZ = nnz[matrixIdx];
        sparse[matrixIdx].M = m;
        sparse[matrixIdx].N = n;
        uint32_t curNNZ = 0;
        for (uint32_t rowIdx = 0; rowIdx < sparse[matrixIdx].M; ++rowIdx)
        {
            const auto nnzPerRow = nnzArray[m * matrixIdx + rowIdx];
            sparse[matrixIdx].ROW[rowIdx] = curNNZ;
            std::list<uint32_t> columnList;

            for (uint32_t i = 0; i < n; ++i)
                columnList.emplace_back(i);
            for (uint32_t i = 0; i < n - nnzPerRow; ++i)
            {
                auto itr = columnList.begin();
                auto size = columnList.size();
                std::uniform_int_distribution<uint32_t> dist(0, size - 1);
                auto offset = dist(gen);
                for (uint32_t j = 0; j < offset; ++j)
                    itr++;
                columnList.erase(itr);
            }

            auto itr = columnList.begin();
            for (uint32_t sparseColIdx = curNNZ;
                 sparseColIdx < curNNZ + nnzPerRow; ++sparseColIdx)
            {
                sparse[matrixIdx].COL[sparseColIdx] = *itr;
                sparse[matrixIdx].V[sparseColIdx] = normal(gen);
                itr++;
            }
            curNNZ += nnzPerRow;
        }
        sparse[matrixIdx].ROW[m] = curNNZ;
        CHECK_EQ(curNNZ, nnz[matrixIdx]);
    }

    delete[] nnz;
    delete[] nnzArray;
}

void SparseMemoryAllocationHost()
{
    static_assert(sizeof(SparseMatrix) == sizeof(LoadDistMatrix) &&
                  sizeof(SparseMatrix) == 48);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(1, 50);

    SparseMatrix* sparse;
    const uint32_t numMatrices = distrib(gen) % 10 + 1;
    const auto m = distrib(gen);
    const auto n = distrib(gen);
    auto* nnz = new uint32_t[numMatrices];

    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        nnz[i] = distrib(gen);
    }

    DeepAllocateSparseHost(&sparse, m, n, nnz, numMatrices);

    //! Verify if all memory regions are allocated correctly
    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        CHECK_EQ(sparse[i].M, m);
        CHECK_EQ(sparse[i].N, n);
        CHECK_EQ(sparse[i].NNZ, nnz[i]);

        for (uint32_t j = 0; j < sparse[i].NNZ; ++j)
        {
            sparse[i].COL[j] = 0;
            sparse[i].V[j] = 0.0f;
        }
        for (uint32_t j = 0; j < sparse[i].M + 1; ++j)
            sparse[i].ROW[j] = j;
    }

    DeepFreeSparseHost(sparse, numMatrices);

    Util::MemoryManager::ClearCudaMemoryPool();
    Util::MemoryManager::ClearHostMemoryPool();
    delete[] nnz;
}

void LoadDistMemoryAllocationHost()
{
    static_assert(sizeof(SparseMatrix) == sizeof(LoadDistMatrix) &&
                  sizeof(SparseMatrix) == 48);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> uniform(1, 50);
    std::normal_distribution<float> normal(0, 100);

    SparseMatrix* sparse;
    const uint32_t numMatrices = uniform(gen) % 10 + 1;
    const auto m = uniform(gen);
    const auto n = uniform(gen);

    GenerateRandomSparseArray(&sparse, m, n, numMatrices);

    LoadDistMatrix* loadDist;
    DeepAllocateLoadDistHost(&loadDist, sparse, numMatrices);

    //! Verify if all memory regions are correctly allocated
    for (uint32_t matrixIdx = 0; matrixIdx < numMatrices; ++matrixIdx)
    {
        CHECK_EQ(loadDist[matrixIdx].M, sparse[matrixIdx].M);
        CHECK_EQ(loadDist[matrixIdx].N, sparse[matrixIdx].N);
        CHECK_EQ(loadDist[matrixIdx].NNZ, sparse[matrixIdx].NNZ);

        for (uint32_t rowIdx = 0; rowIdx < loadDist[matrixIdx].M; ++rowIdx)
        {
            loadDist[matrixIdx].ROW[rowIdx] = sparse[matrixIdx].ROW[rowIdx];
            loadDist[matrixIdx].ROW[rowIdx + 1] =
                sparse[matrixIdx].ROW[rowIdx + 1];

            for (uint32_t k = loadDist[matrixIdx].ROW[rowIdx];
                 k < loadDist[matrixIdx].ROW[rowIdx + 1]; ++k)
            {
                loadDist[matrixIdx].COL[k] = uniform(gen);
                loadDist[matrixIdx].Load[k] = uniform(gen);
            }
        }
    }

    DeepFreeSparseHost(sparse, numMatrices);
    DeepFreeLoadDistHost(loadDist, numMatrices);
    Util::MemoryManager::ClearCudaMemoryPool();
    Util::MemoryManager::ClearHostMemoryPool();
}

void SparseMemoryAllocationDevice()
{
    static_assert(sizeof(SparseMatrix) == sizeof(LoadDistMatrix) &&
                  sizeof(SparseMatrix) == 48);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> uniformInt(1, 50);
    std::normal_distribution<float> normal(0, 10);

    SparseMatrix* hostSparseSrc;
    SparseMatrix* hostSparseDst;
    const uint32_t numMatrices = uniformInt(gen) % 10 + 1;
    const auto m = uniformInt(gen);
    const auto n = uniformInt(gen);
    auto* nnz = new uint32_t[numMatrices];

    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        nnz[i] = uniformInt(gen);
    }

    DeepAllocateSparseHost(&hostSparseSrc, m, n, nnz, numMatrices);
    DeepAllocateSparseHost(&hostSparseDst, m, n, nnz, numMatrices);

    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        CHECK_EQ(hostSparseSrc[i].M, m);
        CHECK_EQ(hostSparseSrc[i].N, n);
        CHECK_EQ(hostSparseSrc[i].NNZ, nnz[i]);

        for (uint32_t j = 0; j < hostSparseSrc[i].NNZ; ++j)
        {
            hostSparseSrc[i].COL[j] = uniformInt(gen);
            hostSparseSrc[i].V[j] = normal(gen);
        }
        for (uint32_t j = 0; j < hostSparseSrc[i].M + 1; ++j)
            hostSparseSrc[i].ROW[j] = j;
    }

    SparseMatrix* deviceSparse;
    DeepAllocateSparseCuda(&deviceSparse, hostSparseSrc, numMatrices, 0);
    DeepCopyHostToDevice(deviceSparse, hostSparseSrc, numMatrices, 0);
    DeepCopyDeviceToHost(hostSparseDst, deviceSparse, numMatrices, 0);

    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        CHECK_EQ(hostSparseSrc[i].M, hostSparseDst[i].M);
        CHECK_EQ(hostSparseSrc[i].N, hostSparseDst[i].N);
        CHECK_EQ(hostSparseSrc[i].NNZ, hostSparseDst[i].NNZ);

        for (uint32_t j = 0; j < hostSparseSrc[i].NNZ; ++j)
        {
            CHECK_EQ(hostSparseSrc[i].V[j], hostSparseDst[i].V[j]);
            CHECK_EQ(hostSparseSrc[i].COL[j], hostSparseDst[i].COL[j]);
        }

        for (uint32_t j = 0; j < hostSparseSrc[i].M + 1; ++j)
            CHECK_EQ(hostSparseSrc[i].ROW[j], hostSparseDst[i].ROW[j]);
    }

    DeepFreeSparseHost(hostSparseSrc, numMatrices);
    DeepFreeSparseHost(hostSparseDst, numMatrices);
    DeepFreeSparseCuda(deviceSparse, numMatrices, 0);

    CHECK_EQ(Util::MemoryManager::GetAllocatedByteSizeCuda(), 0);
    CHECK_EQ(Util::MemoryManager::GetAllocatedByteSizeHost(), 0);

    Util::MemoryManager::ClearCudaMemoryPool();
    Util::MemoryManager::ClearHostMemoryPool();
}

void SparseMemoryCopyDeviceToDevice()
{
    static_assert(sizeof(SparseMatrix) == sizeof(LoadDistMatrix) &&
                  sizeof(SparseMatrix) == 48);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> uniformInt(1, 50);
    std::normal_distribution<float> normal(0, 10);

    SparseMatrix* hostSparseSrc;
    SparseMatrix* hostSparseDst;
    const uint32_t numMatrices = uniformInt(gen) % 10 + 1;
    const auto m = uniformInt(gen);
    const auto n = uniformInt(gen);
    auto* nnz = new uint32_t[numMatrices];

    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        nnz[i] = uniformInt(gen);
    }

    DeepAllocateSparseHost(&hostSparseSrc, m, n, nnz, numMatrices);
    DeepAllocateSparseHost(&hostSparseDst, m, n, nnz, numMatrices);

    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        CHECK_EQ(hostSparseSrc[i].M, m);
        CHECK_EQ(hostSparseSrc[i].N, n);
        CHECK_EQ(hostSparseSrc[i].NNZ, nnz[i]);

        for (uint32_t j = 0; j < hostSparseSrc[i].NNZ; ++j)
        {
            hostSparseSrc[i].COL[j] = uniformInt(gen);
            hostSparseSrc[i].V[j] = normal(gen);
        }
        for (uint32_t j = 0; j < hostSparseSrc[i].M + 1; ++j)
            hostSparseSrc[i].ROW[j] = j;
    }

    SparseMatrix* deviceSparseSrc;
    SparseMatrix* deviceSparseDst;
    DeepAllocateSparseCuda(&deviceSparseSrc, hostSparseSrc, numMatrices, 0);
    DeepAllocateSparseCuda(&deviceSparseDst, hostSparseSrc, numMatrices, 0);

    DeepCopyHostToDevice(deviceSparseSrc, hostSparseSrc, numMatrices, 0);
    DeepCopyDeviceToDevice(deviceSparseDst, deviceSparseSrc, numMatrices, 0);
    DeepCopyDeviceToHost(hostSparseDst, deviceSparseDst, numMatrices, 0);

    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        CHECK_EQ(hostSparseSrc[i].M, hostSparseDst[i].M);
        CHECK_EQ(hostSparseSrc[i].N, hostSparseDst[i].N);
        CHECK_EQ(hostSparseSrc[i].NNZ, hostSparseDst[i].NNZ);

        for (uint32_t j = 0; j < hostSparseSrc[i].NNZ; ++j)
        {
            CHECK_EQ(hostSparseSrc[i].V[j], hostSparseDst[i].V[j]);
            CHECK_EQ(hostSparseSrc[i].COL[j], hostSparseDst[i].COL[j]);
        }

        for (uint32_t j = 0; j < hostSparseSrc[i].M + 1; ++j)
            CHECK_EQ(hostSparseSrc[i].ROW[j], hostSparseDst[i].ROW[j]);
    }

    DeepFreeSparseHost(hostSparseSrc, numMatrices);
    DeepFreeSparseHost(hostSparseDst, numMatrices);
    DeepFreeSparseCuda(deviceSparseSrc, numMatrices, 0);
    DeepFreeSparseCuda(deviceSparseDst, numMatrices, 0);

    CHECK_EQ(Util::MemoryManager::GetAllocatedByteSizeCuda(), 0);
    CHECK_EQ(Util::MemoryManager::GetAllocatedByteSizeHost(), 0);

    Util::MemoryManager::ClearCudaMemoryPool();
    Util::MemoryManager::ClearHostMemoryPool();
}

}  // namespace Sapphire::Test