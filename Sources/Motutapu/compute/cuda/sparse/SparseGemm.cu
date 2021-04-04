// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/cuda/sparse/SparseGemm.cuh>
#include <cstdlib>

#define MAX_NNZ_PER_BLOCK_LARGE 1024
#define MAX_NNZ_PER_BLOCK_SMALL 512
#define GEMM_BLOCK_NUM 16

namespace Motutapu::Compute::Sparse
{
__host__ void Gemm(SparseMatrix* output, SparseMatrix* a, SparseMatrix* b,
                   LoadDistMatrix* loadDist, size_t numMatrices)
{
    const auto numLoops = 8;
    const auto threadDim = MAX_THREAD_DIM_X / numLoops;

    const auto blockDim = totalSize / MAX_THREAD_DIM_X;
    const auto firstLaunchSize = blockDim * threadDim * numLoops;

    if (firstLaunchSize > 0)
        GemmKernel<<<blockDim, threadDim>>>(loadDist, output, a, b);
    if (numMatrices > firstLaunchSize)
    {
        const SparseMatrix* offsetA = a + firstLaunchSize;
        const SparseMatrix* offsetB = b + firstLaunchSize;
        SparseMatrix* loadDistOffset = loadDist + firstLaunchSize;

        const auto secondLaunchSize = numMatrices - firstLaunchSize;

        GemmKernel<<<secondLaunchSize, threadDim>>>(loadDistOffset, output,
                                                    offsetA, offsetB);
    }
}

//! Todo : unify calculate Load kernel and Calculate Gemm
//! Should be executed using single block
__global__ void GemmKernel(LoadDistMatrix* loadDist, SparseMatrix* output,
                           SparseMatrix* a, SparseMatrix* b)
{
    __shared__ uint32_t* nnzTotal;

    uint32_t rowStart[GEMM_BLOCK_NUM];
    uint32_t colStart[GEMM_BLOCK_NUM];
    //! Size must be larger than Number of required blocks per row + 1

    const auto matrixIdx = blockIdx.x;
    const auto rowIdxBegin = threadIdx.x;
    const auto rowIdxStride = blockDim.x;

    SparseMatrix* curA = a + matrixIdx;
    SparseMatrix* curB = b + matrixIdx;
    SparseMatrix* curOutput = output + matrixIdx;
    LoadDistMatrix* curLoadDist = loadDist + matrixIdx;

    if (threadIdx.x == 0)
    {
        *nnzTotal = 0;
    }

    uint32_t idx = 0;
    for (auto rowIdxA = rowIdxBegin rowIdxA < M; rowIdxA += rowIdxStride)
    {
        auto sparseColIdxA = curA->ROW[rowIdxA];
        loadDist->ROW[rowIdxA] = curA->ROW[rowIdxA];
        rowStart[idx] = rowIdxA;
        colStart[idx] = sparseColIdxA;
        uint32_t curLoad = 0;
        uint32_t nnzPerRow = 0;
        for (; sparseColIdxA < curA->ROW[rowIdxA + 1]; ++sparseColIdxA)
        {
            const auto colIdxA = curA->COL[sparseColIdxA];
            const auto numElemPerRowB =
                curB->Row[colIdxA + 1] - curB->ROW[colIdxA];
            curLoadDist->Load[sparseColIdxA] = numElemPerRowB;
            curLoadDist->COL[sparseColIdxA] = colIdxA;

            if (sparseColIdxA != a->ROW[rowIdxA])
            {
                //! Load will stack as row advances
                curLoadDist->Load[sparseColIdxA] +=
                    curLoadDist->Load[sparseColIdxA - 1];
            }

            if (curLoad + numElemPerRowB > MAX_NNZ_PER_BLOCK_LARGE)
            {
                idx += 1;
                rowStart[idx] = rowIdxA;
                colStart[idx] = sparseColIdxA;
                curLoad = 0;
            }
            curLoad += numElemPerRowB;
            nnzPerRow += numElemPerRowB;
        }

        atomicAdd_block(nnzTotal, nnzPerRow);
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        output->NNZ = *nnzTotal;
        output->M = a->M;
        output->N = b->N;
        cudaMalloc(output->V, sizeof(float) * (*nnzTotal));
        cudaMalloc(output->COL, sizeof(uint32_t) * (*nnzTotal));
        cudaMalloc(output->ROW, sizeof(uint32_t) * (a->M + 1));
    }

    __syncthreads();

    idx = 0;
    for (auto rowIdxA = rowIdxBegin rowIdxA < M; rowIdxA += rowIdxStride)
    {
        CalculateRowKernel<<<1, 32>>>(curOut, curA, curB, curLoadDist, rowIdxA,
                                      colStart[idx], curStart[idx + 1]);
        idx += 1;
    }
}

__global__ void CalculateRowKernel(SparseMatrix* out, SparseMatrix* a,
                                   SparseMatrix* b, LoadDistMatrix* loadDist,
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
    const auto nnz = loadDist->Load->V[sparseColIndexEnd - 1];

    if (sparseColIdxA < sparseColIdxAEnd)
    {
        for (uint32_t sparseColIdxB = sparseColIdxBBegin;
             sparseColIdxB < sparseColIdxBEnd, ++sparseColIdxB)
        {
            const auto unmergedSparseRowIdx = loadDist->Load[sparseColIdxA] +
                                              sparseColIdxB -
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
        out->V[idx] = tempValueArray[tempIdx];
        out->COL[idx] = tempValueArray[tempIdx];
    }
}

[[maybe_unused]] __host__ void CalculateGemm(SparseMatrix* c,
                                             const SparseMatrix* a,
                                             const SparseMatrix* b,
                                             LoadDistMatrix* loadDist,
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
                        nullptr, 0, a + matrixIdx, b + matrixIdx,
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
                    nullptr, 0, a + matrixIdx, b + matrixIx,
                    loadDist + matrixIdx, rowIdx, prevSparseColIdx);
                prevSparseColIdx = sparseColIdx;
            }
        }
    }
}

}  // namespace Motutapu::Compute::Sparse