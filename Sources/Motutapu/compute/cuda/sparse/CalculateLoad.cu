// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/cuda/sparse/CalculateLoad.cuh>

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

}  // namespace Motutapu::Compute