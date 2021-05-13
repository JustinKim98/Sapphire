// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/sparse/naive/SparseGemm.hpp>
#include <Sapphire/util/MemoryManager.hpp>
#include <Sapphire/util/Spinlock.hpp>
#include <atomic>

namespace Sapphire::Compute::Sparse::Naive
{
uint32_t Hash(uint32_t col, uint32_t numBuckets)
{
    return (12345 ^ col) % numBuckets;
}

void Insert(uint32_t* tempIdxBuffer, float* tempValueBuffer,
            std::atomic<bool>* flagBuffer, uint32_t matrixIdx, uint32_t rowIdx,
            uint32_t colIdx, float value)
{
    const auto offset =
        matrixIdx * rowIdx * MAX_NNZ_PER_ROW + Hash(colIdx, MAX_NNZ_PER_ROW);

    Util::SpinLock::Lock(flagBuffer + offset);

    tempIdxBuffer[offset] = colIdx;
    tempValueBuffer[offset] = value;

    Util::SpinLock::Release(flagBuffer + offset);
}

void Gemm(SparseMatrix** output, SparseMatrix* a, SparseMatrix* b, uint32_t m,
          uint32_t n, size_t numMatrices)
{
    uint32_t tempIdxBuffer[MAX_NNZ_PER_ROW * m * numMatrices];
    float tempValueBuffer[MAX_NNZ_PER_ROW * m * numMatrices];
    std::atomic<bool> flagBuffer[MAX_NNZ_PER_ROW * m * numMatrices];

    for (uint32_t matrixIdx = 0; matrixIdx < numMatrices; ++matrixIdx)
    {
        for (uint32_t rowIdx = 0; rowIdx < m; ++rowIdx)
        {
            auto* curMatrixA = a + matrixIdx;
            auto* curMatrixB = b + matrixIdx;
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
                    Insert(tempIdxBuffer, tempValueBuffer, flagBuffer,
                           matrixIdx, rowIdx, colIdxB, valueOut);
                }
            }
        }
    }
}
}  // namespace Sapphire::Compute::Sparse::Naive
