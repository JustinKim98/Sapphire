// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/sparse/naive/SparseGemm.hpp>
#include <Sapphire/util/ResourceManager.hpp>
#include <Sapphire/util/Spinlock.hpp>
#include <algorithm>
#include <atomic>
#include <vector>

namespace Sapphire::Compute::Sparse::Naive
{
uint32_t Hash1(uint32_t col, uint32_t numBuckets)
{
    return col % numBuckets;
}

void Insert(uint32_t* tempIdxBuffer, float* tempValueBuffer, uint32_t m,
            uint32_t matrixIdx, uint32_t rowIdx, uint32_t colIdx, float value,
            uint32_t* nnz)
{
    auto key = Hash1(colIdx, MAX_NNZ_PER_ROW_HOST);

    while (true)
    {
        auto offset = matrixIdx * m * MAX_NNZ_PER_ROW_HOST +
                      rowIdx * MAX_NNZ_PER_ROW_HOST + key;
        if (tempIdxBuffer[offset] == static_cast<uint32_t>(INF) ||
            tempIdxBuffer[offset] == colIdx)
        {
            if (tempIdxBuffer[offset] != colIdx)
                *nnz += 1;
            tempIdxBuffer[offset] = colIdx;
            tempValueBuffer[offset] += value;
            break;
        }
        key = (key + 1) % (MAX_NNZ_PER_ROW_HOST - 1);
    }
}

void Sort(uint32_t* tempIdxBuffer, float* tempValueBuffer,
          const size_t beginIdx,
          const size_t endIdx)
{
    uint32_t indices[MAX_NNZ_PER_ROW_HOST + 1];
    float values[MAX_NNZ_PER_ROW_HOST + 1];

    if (beginIdx >= endIdx || endIdx == beginIdx + 1)
        return;

    const auto midIdx = beginIdx + (endIdx - beginIdx) / 2;
    Sort(tempIdxBuffer, tempValueBuffer, beginIdx, midIdx);
    Sort(tempIdxBuffer, tempValueBuffer, midIdx, endIdx);

    size_t left = 0, right = 0;

    for (size_t vectorIdx = 0; vectorIdx < endIdx - beginIdx; ++vectorIdx)
    {
        if (right == endIdx - midIdx)
        {
            indices[vectorIdx] = tempIdxBuffer[beginIdx + left];
            values[vectorIdx] = tempValueBuffer[beginIdx + left];
            left++;
        }
        else if (left == midIdx - beginIdx)
        {
            indices[vectorIdx] = tempIdxBuffer[midIdx + right];
            values[vectorIdx] = tempValueBuffer[midIdx + right];
            right++;
        }
        else if (tempIdxBuffer[beginIdx + left] < tempIdxBuffer[midIdx + right])
        {
            indices[vectorIdx] = tempIdxBuffer[beginIdx + left];
            values[vectorIdx] = tempValueBuffer[beginIdx + left];
            left++;
        }
        else
        {
            indices[vectorIdx] = tempIdxBuffer[midIdx + right];
            values[vectorIdx] = tempValueBuffer[midIdx + right];
            right++;
        }
    }

    std::copy(indices, indices + (endIdx - beginIdx), tempIdxBuffer + beginIdx);
    std::copy(values, values + (endIdx - beginIdx), tempValueBuffer + beginIdx);
}

void Gemm(SparseMatrix** output, SparseMatrix* a, SparseMatrix* b, uint32_t m,
          uint32_t n, size_t numMatrices)
{
    auto* tempIdxBuffer =
        new uint32_t[MAX_NNZ_PER_ROW_HOST * (m + 1) * numMatrices];
    auto* tempValueBuffer =
        new float[MAX_NNZ_PER_ROW_HOST * (m + 1) * numMatrices];
    std::fill(tempIdxBuffer,
              tempIdxBuffer + MAX_NNZ_PER_ROW_HOST * (m + 1) * numMatrices,
              static_cast<uint32_t>(INF));
    std::fill(tempValueBuffer,
              tempValueBuffer + MAX_NNZ_PER_ROW_HOST * (m + 1) * numMatrices,
              0.0f);

    *output = static_cast<SparseMatrix*>(
        Util::ResourceManager::GetMemoryHost(sizeof(SparseMatrix) * numMatrices));

    for (uint32_t i = 0; i < numMatrices; ++i)
        (*output)[i].ROW = static_cast<uint32_t*>(
            Util::ResourceManager::GetMemoryHost(sizeof(uint32_t) * (m + 1)));

#pragma omp parallel for default(none)                                      \
    shared(numMatrices, m, n, a, b, output, tempIdxBuffer, tempValueBuffer) \
        schedule(static)
    for (long matrixIdx = 0; matrixIdx < static_cast<long>(numMatrices); ++
         matrixIdx)
    {
        auto* curMatrixA = a + matrixIdx;
        auto* curMatrixB = b + matrixIdx;
        auto* curMatrixOut = (*output) + matrixIdx;
        uint32_t matrixNNZ = 0;
        for (uint32_t rowIdx = 0; rowIdx < m; ++rowIdx)
        {
            curMatrixOut->ROW[rowIdx] = matrixNNZ;
            uint32_t rowNNZ = 0;
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
                    Insert(tempIdxBuffer, tempValueBuffer, m, matrixIdx, rowIdx,
                           colIdxB, valueOut, &rowNNZ);
                }
            }

            const auto beginOffset = matrixIdx * m * MAX_NNZ_PER_ROW_HOST +
                                     rowIdx * MAX_NNZ_PER_ROW_HOST;
            const auto endOffset = matrixIdx * m * MAX_NNZ_PER_ROW_HOST +
                                   (rowIdx + 1) * MAX_NNZ_PER_ROW_HOST;

            // std::sort(tempIdxBuffer + beginOffset, tempIdxBuffer +
            // endOffset);
            Sort(tempIdxBuffer, tempValueBuffer, beginOffset, endOffset);

            matrixNNZ += rowNNZ;
        }
        curMatrixOut->ROW[m] = matrixNNZ;

        curMatrixOut->COL = static_cast<uint32_t*>(
            Util::ResourceManager::GetMemoryHost(sizeof(uint32_t) * matrixNNZ));
        curMatrixOut->V = static_cast<float*>(
            Util::ResourceManager::GetMemoryHost(sizeof(float) * matrixNNZ));

        curMatrixOut->NNZ = matrixNNZ;
        curMatrixOut->M = m;
        curMatrixOut->N = n;

        for (size_t rowIdx = 0; rowIdx < m; ++rowIdx)
        {
            const auto rowNNZ =
                curMatrixOut->ROW[rowIdx + 1] - curMatrixOut->ROW[rowIdx];
            if (rowNNZ)
            {
                const auto copyOffset = matrixIdx * m * MAX_NNZ_PER_ROW_HOST +
                                        rowIdx * MAX_NNZ_PER_ROW_HOST;
                std::copy(tempIdxBuffer + copyOffset,
                          tempIdxBuffer + copyOffset + rowNNZ,
                          curMatrixOut->COL + curMatrixOut->ROW[rowIdx]);
                std::copy(tempValueBuffer + copyOffset,
                          tempValueBuffer + copyOffset + rowNNZ,
                          curMatrixOut->V + curMatrixOut->ROW[rowIdx]);
            }
        }
    }
    delete[] tempIdxBuffer;
    delete[] tempValueBuffer;
}
} // namespace Sapphire::Compute::Sparse::Naive
