// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/cuda/sparse/SparseGemm.cuh>

namespace Motutapu::Compute
{
__host__ void CalculateLoad(SparseMatrix* a, SparseMatrix* b,
                            SparseMatrix* loadDist, size_t numMatrices)
{
    const auto numLoops = 8;
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

    const auto blockDim = totalSize / MAX_THREAD_DIM_X;
    const auto firstLaunchSize = blockDim * threadDim * numLoops;

    if (firstLaunchSize > 0)
        CalculateLoadKernel<<<blockDim, threadDim>>>(a, b, loadDist,
                                                     firstLaunchSize);
    if (numMatrices > firstLaunchSize)
    {
        const SparseMatrix* offsetA = a + firstLaunchSize;
        const SparseMatrix* offsetB = b + firstLaunchSize;
        SparseMatrix* loadDistOffset = loadDist + firstLaunchSize;

        const auto secondLaunchSize = numMatrices - firstLaunchSize;

        CalculateLoadKernel<<<secondLaunchSize, threadDim>>>(
            offsetA, offsetB, loadDistOffset, secondLaunchSize);
    }
}

__global__ void CalculateLoadKernel(SparseMatrix* a, SparseMatrix* b,
                                    SparseMatrix* loadDist, size_t numMatrices)
{
    for (auto sparseMatrixIdx = gridIdx.x; sparseMatrixIdx < numMatrices;
         sparseMatrixIdx += gridDim.x)
    {
        for (auto rowIdx = blockIdx.x; rowIdx < M; rowIdx += blockDim.x)
        {
            loadDist->ROW[rowIdx] = a->ROW[rowIdx];
            for (auto sparseColIdx = a->ROW[rowIdx];
                 sparseColIdx < a->ROW[rowIdx + 1]; ++sparseColIdx)
            {
                const auto colIdx = a->COL[sparseColIdx];
                const auto numElemPerRowB = b->Row[colIdx + 1] - b->ROW[colIdx];
                loadDist->V[sparseColIdx] = numElemPerRowB;
                loadDist->COL[sparseColIdx] = colIdx;
            }
        }
    }
}

__host__ void CalculateRow(SparseMatrix* unmergedSparseMatrixRow,
                           SparseMatrix* a, SparseMatrix* b,
                           SparseMatrix* loadDist, uint32_t matrixNum)
{
    for (uint32_t matrixIdx = 0; matrixIdx < matrixNum; ++matrixIdx)
    {
        for (uint32_t rowIdx = 0; rowIdx < loadDist[matrixIdx].M + 1; ++rowIdx)
        {
            uint32_t nnzPerRow = 0;
            uint32_t maxSizePerRow = 0;
            for (uint32_t sparseColIdx = loadDist[matrixIdx].ROW[rowIdx];
                 sparseColIdx < loadDist[matrixIdx].ROW[rowIdx + 1];
                 ++sparseColIdx)
            {
                nnzPerRow += loadDist[matrixIdx].V[sparseColIdx];
                if (maxSizePerRow < loadDist[matrixIdx].V[sparseColIdx])
                    maxSizePerRow = loadDist[matrixIdx].V[sparseColIdx];
            }

            if (nnzPerRow > 0)
                CalculateRowKernelRow(nullptr, a + matrixIdx, b + matrixIdx,
                                      maxSizePerRow, rowIdx);
        }
    }
}

__global__ void CalculateRowKernelRow(SparseMatrix* unmergedSparseMatrixRow,
                                      SparseMatrix* a, SparseMatrix* b,
                                      uint32_t maxSizePerValue, uint32_t rowIdx)
{
    extern __shared__ float unmergedSparseRow[];

    const auto sparseColIdxABegin = a->ROW[rowIdx];
    const auto sparseColIdxAEnd = a->ROW[rowIdx + 1];

    const auto sparseColIdxA = sparseColIdxABegin + threadIdx.x;

    if (sparseColIdxA < sparseColIdxAEnd)
    {
        const auto colIdxA = a->COL[sparseColIdxA];
        const auto sparseColIdxBBegin = b->ROW[colIdxA];
        const auto sparseColIdxBEnd = b->ROW[colIdxA + 1];

        for (uint32_t sparseColIdxB = sparseColIdxBBegin;
             sparseColIdxB < sparseColIdxBEnd, ++sparseColIdxB)
        {
            //! todo : Think of more efficient way of calculating this position
            const auto unmergedSparseRowIdx = maxSizePerValue * threadIdx.x +
                                              sparseColIdxB -
                                              sparseColIdxBBegin;

            unmergedSparseRow[unmergedSparseRowIdx] =
                a->V[sparseColIdxA] * b->V[sparseColIdxB]
        }
    }

    //! todo : Merge and sort the unmerged SparseRow
}

}  // namespace Motutapu::Compute