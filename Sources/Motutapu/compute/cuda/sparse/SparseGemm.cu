// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <stdint-gcc.h>
#include <Motutapu/compute/cuda/sparse/SparseGemm.cuh>
#include <cstdlib>

#define MAX_NNZ_PER_BLOCK_LARGE 1024
#define MAX_NNZ_PER_BLOCK_SMALL 512
#define GEMM_BLOCK_NUM 16
#define MAX_BLOCK_DIM 1024

namespace Motutapu::Compute::Sparse
{
__host__ void Gemm(SparseMatrix* output, SparseMatrix* a, SparseMatrix* b,
                   LoadDistMatrix* loadDist, size_t numMatrices)
{
    auto* nnzArray =
        static_cast<uint32_t*>(malloc(sizeof(uint32_t) * numMatrices));
    CallLoadDist(a, b, loadDist, nnzArray, numMatrices);
    AllocateOutput(output, a, b, numMatrices, nnzArray);
}

__host__ void CallLoadDist(SparseMatrix* a, SparseMatrix* b,
                           LoadDistMatrix* loadDist, uint32_t* nnzArray,
                           size_t numMatrices)
{
    const auto numLoops = 8;
    const auto M = a[0].M;
    const auto threadDim = M / numLoops;
    uint32_t* deviceNNZArray = nullptr;
    cudaMalloc((void**)&deviceNNZArray, sizeof(uint32_t) * numMatrices);

    const auto blockDim =
        (numMatrices > MAX_GRID_DIM) ? numMatrices - MAX_GRID_DIM : numMatrices;

    if (blockDim > 0)
        LoadDistKernel<<<blockDim, threadDim>>>(loadDist, a, b, deviceNNZArray);
    if (numMatrices > blockDim)
    {
        const SparseMatrix* offsetA = a + blockDim;
        const SparseMatrix* offsetB = b + blockDim;
        LoadDistMatrix* loadDistOffset = loadDist + blockDim;

        const auto secondBlockDim = numMatrices - blockDim;
        LoadDistKernel<<<secondBlockDim, threadDim>>>(loadDistOffset, offsetA,
                                                      offsetB, deviceNNZArray);
    }
    cudaMemcpy(nnzArray, deviceNNZArray, sizeof(uint32_t) * numMatrices,
               cudaMemcpyDeviceToHost);
    cudaFree(deviceNNZArray);
}

__host__ void AllocateOutput(SparseMatrix* output, SparseMatrix* a,
                             SparseMatrix* b, size_t numMatrices,
                             const uint32_t* nnzArray)
{
    for (uint32_t matrixIdx = 0; matrixIdx < numMatrices; ++matrixIdx)
    {
        SparseMatrix* curOutput = output + matrixIdx;
        curOutput->M = a->M;
        curOutput->N = b->N;
        curOutput->NNZ = nnzArray[matrixIdx];
        cudaFree(curOutput->V);
        cudaFree(curOutput->ROW);
        cudaFree(curOutput->COL);
        cudaMalloc((void**)curOutput->V, sizeof(float) * curOutput->NNZ);
        cudaMalloc((void**)curOutput->COL, sizeof(float) * curOutput->NNZ);
        cudaMalloc((void**)curOutput->ROW,
                   sizeof(uint32_t) * (curOutput->M + 1));
    }
}

//! Todo : unify calculate Load kernel and Calculate Gemm
//! Should be executed using single block
__global__ void LoadDistKernel(LoadDistMatrix* loadDist, SparseMatrix* a,
                               SparseMatrix* b, uint32_t* nnzArray)
{
    __shared__ uint32_t* nnzPerMatrix;

    uint32_t rowStart[GEMM_BLOCK_NUM];
    uint32_t colStart[GEMM_BLOCK_NUM];
    //! ByteSize must be larger than Number of required blocks per row + 1

    const auto matrixIdx = blockIdx.x;
    const auto rowIdxBegin = threadIdx.x;
    const auto rowIdxStride = blockDim.x;

    SparseMatrix* curA = a + matrixIdx;
    SparseMatrix* curB = b + matrixIdx;
    LoadDistMatrix* curLoadDist = loadDist + matrixIdx;

    if (threadIdx.x == 0)
        *nnzPerMatrix = 0;

    uint32_t idx = 0;
    for (auto rowIdxA = rowIdxBegin; rowIdxA < a[matrixIdx].M;
         rowIdxA += rowIdxStride)
    {
        auto sparseColIdxA = curA->ROW[rowIdxA];
        curLoadDist->ROW[rowIdxA] = curA->ROW[rowIdxA];
        rowStart[idx] = rowIdxA;
        colStart[idx] = sparseColIdxA;
        uint32_t curLoad = 0;
        uint32_t nnzPerRow = 0;
        for (; sparseColIdxA < curA->ROW[rowIdxA + 1]; ++sparseColIdxA)
        {
            const auto colIdxA = curA->COL[sparseColIdxA];
            const auto numElemPerRowB =
                curB->ROW[colIdxA + 1] - curB->ROW[colIdxA];
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
        atomicAdd_block(nnzPerMatrix, nnzPerRow);
    }

    curLoadDist->NNZ = *nnzPerMatrix;
    nnzArray[matrixIdx] = *nnzPerMatrix;
}

__global__ void CalculateRowKernel(SparseMatrix* out, SparseMatrix* a,
                                   SparseMatrix* b, LoadDistMatrix* loadDist,
                                   uint32_t rowIdx, uint32_t sparseColIdxBegin,
                                   uint32_t sparseColIdxEnd)
{
    //! Stores pair of computer value and pair of index
    __shared__ float tempValueArray[MAX_NNZ_PER_BLOCK_LARGE];
    __shared__ uint32_t tempIdxArray[MAX_NNZ_PER_BLOCK_LARGE];

    const auto M = out[0].M;
    SparseMatrix* curOut = out + blockIdx.x / M;
    SparseMatrix* curA = a + blockIdx.x / M;
    SparseMatrix* curB = b + blockIdx.x / M;
    LoadDistMatrix* curLoadDist = loadDist + blockIdx.x / M;

    const auto sparseColIdxOffset = threadIdx.x;
    const auto sparseColIdxA = sparseColIdxBegin + sparseColIdxOffset;

    const auto colIdxA = curA->COL[sparseColIdxA];
    const auto sparseColIdxBBegin = curB->ROW[colIdxA];
    const auto sparseColIdxBEnd = curB->ROW[colIdxA + 1];
    const auto nnz = curLoadDist->Load[sparseColIdxEnd - 1];

    if (sparseColIdxA < sparseColIdxEnd)
    {
        for (uint32_t sparseColIdxB = sparseColIdxBBegin;
             sparseColIdxB < sparseColIdxBEnd; ++sparseColIdxB)
        {
            const auto unmergedSparseRowIdx = loadDist->Load[sparseColIdxA] +
                                              sparseColIdxB -
                                              sparseColIdxBBegin;

            tempValueArray[unmergedSparseRowIdx] =
                curA->V[sparseColIdxA] * curB->V[sparseColIdxB];
            tempIdxArray[unmergedSparseRowIdx] = curB->COL[sparseColIdxB];
        }
    }

    uint32_t mergedNumElements = 0;
    Sort(tempValueArray, tempIdxArray, nnz);
    Merge(tempValueArray, tempIdxArray, nnz, &mergedNumElements);

    const auto stride = (mergedNumElements / blockDim.x > 0)
                            ? mergedNumElements / blockDim.x
                            : 1;

    for (uint32_t idx = stride * threadIdx.x;
         idx < nnz && idx < stride * (threadIdx.x + 1); ++idx)
    {
        //! TODO : How are we going to store the result if row is distributed?
        const auto tempIdx = MAX_NNZ_PER_BLOCK_LARGE - nnz + idx;
        curOut->V[idx] = tempValueArray[tempIdx];
        curOut->COL[idx] = tempIdxArray[tempIdx];
    }
}

// todo : Check if this algorithm works by manipulating on python
__device__ void Sort(float* tempValArray, uint32_t* tempIdxArray,
                     uint32_t arraySize)
{
    const auto id = threadIdx.x;

    // todo : initialize array with largest negative number

    if (id > arraySize / 2)
        return;

    for (uint32_t level = 0; level < log2(arraySize); ++level)
    {
        for (auto stride = __double2uint_rz(pow(2, level)); stride > 0;
             stride /= 2)
        {
            bool dir =
                (id / stride) % 2 == 0;  // true if downward, false if upward

            if (dir)
            {
                const auto idx =
                    (id / (stride * 2)) * stride * 4 + id % (stride * 2);
                float tempVal = tempValArray[idx];
                uint32_t tempIdx = tempIdxArray[idx];

                tempValArray[idx] = tempValArray[idx + stride];
                tempIdxArray[idx] = tempIdxArray[idx + stride];

                tempValArray[idx + stride] = tempVal;
                tempIdxArray[idx + stride] = tempIdx;
            }
            else
            {
                const auto idx = (id / (stride * 2)) * stride * 4 +
                                 (stride * 4 - id % (stride * 2));
                float tempVal = tempValArray[idx];
                uint32_t tempIdx = tempIdxArray[idx];

                tempValArray[idx] = tempValArray[idx - stride];
                tempIdxArray[idx] = tempIdxArray[idx - stride];

                tempValArray[idx - stride] = tempVal;
                tempIdxArray[idx - stride] = tempIdx;
            }
        }
    }
}
}  // namespace Motutapu::Compute::Sparse