// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/sparse/naive/SparseGemm.hpp>
#include <Sapphire/util/MemoryManager.hpp>
#include <Sapphire/util/Spinlock.hpp>
#include <atomic>
#include <vector>

namespace Sapphire::Compute::Sparse::Naive
{
bool Insert(uint32_t* tempIdxBuffer, float* tempValueBuffer,
            std::atomic<bool>* flagBuffer, uint32_t matrixIdx, uint32_t rowIdx,
            uint32_t colIdx, float value)
{
    const auto prime = MAX_NNZ_PER_ROW / 2;
    auto hash1 = [colIdx]() { return colIdx % prime; };
    auto hash2 = [colIdx]() { return prime - colIdx % prime; };
    auto hashOutput = hash1();
    auto offset = matrixIdx * rowIdx * MAX_NNZ_PER_ROW + colIdx;

    bool isFirst = false;
    Util::SpinLock::Lock(flagBuffer + offset);

    for (uint32_t i = 0; tempIdxBuffer[offset] != static_cast<uint32_t>(INF) ||
                         i < MAX_NNZ_PER_ROW;
         ++i)
    {
        hashOutput = (hash1() + i * hash2()) % MAX_NNZ_PER_ROW;
        offset = matrixIdx * rowIdx * MAX_NNZ_PER_ROW + hashOutput;
    }

    if (tempIdxBuffer[offset] == static_cast<uint32_t>(INF))
        isFirst = true;

    tempIdxBuffer[offset] = colIdx;
    tempValueBuffer[offset] += value;

    Util::SpinLock::Release(flagBuffer + offset);
    return isFirst;
}

void Sort(uint32_t* tempIdxBuffer, float* tempValueBuffer, size_t beginIdx,
          size_t endIdx)
{
    if (beginIdx == endIdx || endIdx == beginIdx + 1)
        return;

    const auto midIdx = (beginIdx + endIdx) / 2;
    Sort(tempIdxBuffer, tempValueBuffer, beginIdx, midIdx);
    Sort(tempIdxBuffer, tempValueBuffer, midIdx, endIdx);

    size_t left = 0, right = 0;
    std::vector<uint32_t> indices(endIdx - beginIdx);
    std::vector<float> values(endIdx - beginIdx);
    for (auto i = beginIdx; i < endIdx; ++i)
    {
        if (tempIdxBuffer[beginIdx + left] < tempIdxBuffer[midIdx + right])
        {
            indices[i] = tempIdxBuffer[beginIdx + left];
            values[i] = tempValueBuffer[beginIdx + left];
            left++;
        }
        else
        {
            indices[i] = tempIdxBuffer[midIdx + right];
            values[i] = tempValueBuffer[midIdx + right];
            right++;
        }
    }

    std::copy(indices.begin(), indices.end(), tempIdxBuffer + beginIdx);
    std::copy(values.begin(), values.end(), tempValueBuffer + beginIdx);
}

void Gemm(SparseMatrix** output, SparseMatrix* a, SparseMatrix* b, uint32_t m,
          uint32_t n, size_t numMatrices)
{
    uint32_t tempIdxBuffer[MAX_NNZ_PER_ROW * m * numMatrices];
    float tempValueBuffer[MAX_NNZ_PER_ROW * m * numMatrices];
    std::fill(tempIdxBuffer, tempIdxBuffer + MAX_NNZ_PER_ROW * m * numMatrices,
              static_cast<uint32_t>(INF));
    std::fill(tempValueBuffer,
              tempValueBuffer + MAX_NNZ_PER_ROW * m * numMatrices, 0);
    std::atomic<bool> flagBuffer[MAX_NNZ_PER_ROW * m * numMatrices];

    *output = static_cast<SparseMatrix*>(
        Util::MemoryManager::GetMemoryHost(sizeof(SparseMatrix) * numMatrices));

    for (uint32_t i = 0; i < m; ++i)
        (*output)[i].ROW = static_cast<uint32_t*>(
            Util::MemoryManager::GetMemoryHost(sizeof(uint32_t) * (m + 1)));

#pragma omp parallel for schedule(static) collapse(2) default(none) shared( \
    numMatrices, m, a, b, output, tempIdxBuffer, tempValueBuffer, flagBuffer)
    for (uint32_t matrixIdx = 0; matrixIdx < numMatrices; ++matrixIdx)
    {
        for (uint32_t rowIdx = 0; rowIdx < m; ++rowIdx)
        {
            uint32_t nnz = 0;
            auto* curMatrixA = a + matrixIdx;
            auto* curMatrixB = b + matrixIdx;
            auto* curMatrixOut = (*output) + matrixIdx;
            for (auto sparseColIdx = curMatrixA->ROW[rowIdx];
                 sparseColIdx < curMatrixA->ROW[rowIdx + 1]; ++sparseColIdx)
            {
                const auto colIdxA = curMatrixA->COL[sparseColIdx];
                const auto valueA = curMatrixA->V[sparseColIdx];
                for (auto sparseColIdxB = curMatrixB->ROW[colIdxA];
                     sparseColIdxB < curMatrixB->ROW[colIdxA + 1];
                     ++sparseColIdxB)
                {
                    const auto valueB = curMatrixB->V[sparseColIdxB];
                    const auto colIdxB = curMatrixB->COL[sparseColIdxB];
                    const auto valueOut = valueA * valueB;
                    nnz += static_cast<uint32_t>(
                        Insert(tempIdxBuffer, tempValueBuffer, flagBuffer,
                               matrixIdx, rowIdx, colIdxB, valueOut));
                }
            }

            const auto beginOffset = matrixIdx * rowIdx * MAX_NNZ_PER_ROW;
            const auto endOffset = matrixIdx * (rowIdx + 1) * MAX_NNZ_PER_ROW;
            Sort(tempIdxBuffer, tempValueBuffer, beginOffset, endOffset);
            curMatrixOut->COL = static_cast<uint32_t*>(
                Util::MemoryManager::GetMemoryHost(sizeof(uint32_t) * nnz));
            curMatrixOut->V = static_cast<float*>(
                Util::MemoryManager::GetMemoryHost(sizeof(uint32_t) * nnz));

            std::copy(tempIdxBuffer + beginOffset, tempIdxBuffer + endOffset,
                      curMatrixOut->COL);
            std::copy(tempValueBuffer + beginOffset,
                      tempValueBuffer + endOffset, curMatrixOut->V);
        }
    }
}

}  // namespace Sapphire::Compute::Sparse::Naive
