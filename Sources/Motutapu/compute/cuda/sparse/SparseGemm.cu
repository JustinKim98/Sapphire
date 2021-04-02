// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <stdint-gcc.h>
#include <Motutapu/compute/cuda/sparse/SparseGemm.cuh>

#define MAX_NNZ_PER_BLOCK_LARGE 1024
#define MAX_NNZ_PER_BLOCK_SMALL 512
#define GEMM_BLOCK_NUM 16

namespace Motutapu::Compute
{
__host__ void CalculateLoad(SparseMatrix* a, SparseMatrix* b,
                            LoadDistMatrix* loadDist, size_t numMatrices)
{
    const auto numLoops = 8;
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

    const auto blockDim = totalSize / MAX_THREAD_DIM_X;
    const auto firstLaunchSize = blockDim * threadDim * numLoops;

    if (firstLaunchSize > 0)
        CalculateLoadKernel<<<blockDim, threadDim>>>(loadDist, a, b);
    if (numMatrices > firstLaunchSize)
    {
        const SparseMatrix* offsetA = a + firstLaunchSize;
        const SparseMatrix* offsetB = b + firstLaunchSize;
        SparseMatrix* loadDistOffset = loadDist + firstLaunchSize;

        const auto secondLaunchSize = numMatrices - firstLaunchSize;

        CalculateLoadKernel<<<secondLaunchSize, threadDim>>>(loadDistOffset,
                                                             offsetA, offsetB);
    }
}

//! Todo : unify calculate Load kernel and Calculate Gemm
//! Should be executed using single block
__global__ void CalculateLoadKernel(LoadDistMatrix* loadDist, SparseMatrix* a,
                                    SparseMatrix* b)
{
    __shared__ uint32_t blockStartAddr[GEMM_BLOCK_NUM];
    //! Stores data in the format of (ROW, COL)
    //! Size must be larger than 2*(Number of required blocks + 1)

    uint32_t idx = 0;
    for (auto rowIdx = threadIdx.x; rowIdx < M; rowIdx += blockDim.x)
    {
        loadDist->ROW[rowIdx] = a->ROW[rowIdx];
        uint32_t rowLoad = 0;
        for (auto sparseColIdx = a->ROW[rowIdx];
             sparseColIdx < a->ROW[rowIdx + 1]; ++sparseColIdx)
        {
            const auto colIdx = a->COL[sparseColIdx];
            const auto numElemPerRowB = b->Row[colIdx + 1] - b->ROW[colIdx];
            loadDist->Load[sparseColIdx] = numElemPerRowB;
            loadDist->COL[sparseColIdx] = colIdx;

            if (sparseColIdx != a->ROW[rowIdx])
            {
                //! Load will stack as row advances
                loadDist->Load[sparseColIdx] +=
                    loadDist->Load[sparseColIdx - 1];
            }

            if (rowLoad + numElemPerRowB > MAX_NNZ_PER_BLOCK_LARGE)
            {
                blockStartAddr[idx] = rowIdx;
                blockStartAddr[idx + 1] = colIdx;
                idx += 2;
                rowLoad = 0;
            }
            rowLoad += numElemPerRowB;
        }
    }

    __syncthreads();

    if (threadIdx.x < idx / 2)
    {
        //! todo : Launch kernel Per block
        //! todo : How will we allocate the memory for output
        //! when we don't know how large it will be?
        //! Allocate the output with NNZ per ROW
        CalculateRowKernel<<<1, 32>>>(
            nullptr, nullptr, a, b, loadDist, blockStartAddr[threadIdx.x],
            blockStartAddr[threadIdx.x + 1], blockStartAddr[threadIdx.x + 2]);
    }
}

__global__ void CalculateRowKernel(float* cV, uint32_t* cCOL, SparseMatrix* a,
                                   SparseMatrix* b,
                                   LoadDistMatrix* stackedLoadDist,
                                   uint32_t rowIdx,
                                   uint32_t sparseColIndexBegin,
                                   uint32_t sparseColIndexEnd)
{
    //! Stores pair of computer value and pair of index
    __shared__ float tempValueArray[MAX_NNZ_PER_BLOCK_LARGE];
    __shared__ uint32_t tempIdxArray[MAX_NNZ_PER_BLOCK_LARGE];

    const auto sparseColIdxOffset = threadIdx.x;
    const auto sparseColIdxA = sparseColIdxBegin + sparseColIdxOffset;

    const auto colIdxA = a->COL[sparseColIdxA];
    const auto sparseColIdxBBegin = b->ROW[colIdxA];
    const auto sparseColIdxBEnd = b->ROW[colIdxA + 1];
    const auto nnz = stackedLoadDist->Load->V[sparseColIndexEnd - 1];

    if (sparseColIdxA < sparseColIdxAEnd)
    {
        for (uint32_t sparseColIdxB = sparseColIdxBBegin;
             sparseColIdxB < sparseColIdxBEnd, ++sparseColIdxB)
        {
            const auto unmergedSparseRowIdx =
                stackedLoadDist->Load[sparseColIdxA] + sparseColIdxB -
                sparseColIdxBBegin;

            tempValueArray[unmergedSparseRowIdx] =
                a->V[sparseColIdxA] * b->V[sparseColIdxB];
            tempIdxArray[unmergedSparseRowIdx] = b->COL[sparseColIdxB];
        }
    }

    Sort(tempValueArray, tempIdxArray, nnz);
    Merge(tempValueArray, tempIdxArray, nnz);

    const auto NNZPerThread = (nnz / threadDim.x > 0) ? nnz / threadDim.x : 1;

    for (uint32_t idx = NNZPerThread * threadIdx.x;
         idx < nnz && idx < NNZPerThread * (threadIdx.x + 1); ++idx)
    {
        const auto tempIdx = MAX_NNZ_PER_BLOCK_LARGE - nnz + idx;
        cV[idx] = tempValueArray[tempIdx];
        cCOL[idx] = tempValueArray[tempIdx];
    }
}

__host__ void CalculateGemm(SparseMatrix* c, const SparseMatrix* a,
                            const SparseMatrix* b, LoadDistMatrix* loadDist,
                            uint32_t matrixNum)
{
    for (uint32_t matrixIdx = 0; matrixIdx < matrixNum; ++matrixIdx)
    {
        for (uint32_t rowIdx = 0; rowIdx < loadDist[matrixIdx].M + 1; ++rowIdx)
        {
            uint32_t nnz = 0;
            uint32_t prevSparseColIdx = 0;
            uint32_t sparseColIdx = uint32_t sparseColIdx =
                loadDist[matrixIdx].ROW[rowIdx];
            for (; sparseColIdx < loadDist[matrixIdx].ROW[rowIdx + 1];
                 ++sparseColIdx)
            {
                if (nnz + loadDist[matrixIdx].Load[sparseColIdx] >=
                    MAX_NNZ_PER_BLOCK_LARGE)
                {
                    CalculateRowKernel<<<1, requiredThreads>>>(
                        c + matrixIdx, nullptr, 0, a + matrixIdx, b + matrixIdx,
                        loadDist + matrixIdx, rowIdx, prevSparseColIdx);
                    prevSparseColIdx = sparseColIdx;
                    nnz = 0;
                }

                nnz += loadDist[matrixIdx].Load[sparseColIdx];
                loadDist[matrixIdx].Load[sparseColIdx] = nnz;
            }

            if (nnz > 0 && nnz <= MAX_NNZ_PER_BLOCK_SMALL)
            {
            }
            else if (nnz > MAX_NNZ_PER_BLOCK_SMALL)
            {
                CalculateRowKernel<<<1, requiredThreads>>>(
                    c + matrixIdx, nullptr, 0, a + matrixIdx, b + matrixIx,
                    loadDist + matrixIdx, rowIdx, prevSparseColIdx);
                prevSparseColIdx = sparseColIdx;
            }
        }
    }
}

}  // namespace Motutapu::Compute