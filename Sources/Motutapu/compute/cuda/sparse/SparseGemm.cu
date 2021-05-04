// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <stdint-gcc.h>
#include <Motutapu/compute/Sparse.hpp>
#include <Motutapu/compute/cuda/sparse/SparseGemm.cuh>

#include <cstdlib>

#define MAX_NNZ_PER_BLOCK_LARGE 1024
#define MAX_NNZ_PER_BLOCK_SMALL 512
#define GEMM_BLOCK_NUM 16
#define MAX_BLOCK_DIM 1024
#define INF (~0)

namespace Motutapu::Compute::Cuda::Sparse
{
__device__ uint32_t Hash(uint32_t col, uint32_t numBuckets)
{
    return col % numBuckets;
}

__host__ void Gemm(SparseMatrix** hostOutput, SparseMatrix** cudaOutput,
                   SparseMatrix* hostA, SparseMatrix* cudaA,
                   SparseMatrix* cudaB, uint32_t m, uint32_t n,
                   size_t numMatrices, int deviceId, bool copyResultToHost)
{
    LoadDistMatrix *hostLoadDist, *cudaLoadDist;
    auto* nnzArray =
        static_cast<uint32_t*>(malloc(sizeof(uint32_t) * numMatrices));
    DeepAllocateLoadDistHost(&hostLoadDist, hostA, numMatrices);
    DeepAllocateLoadDistCuda(&cudaLoadDist, hostLoadDist, numMatrices,
                             deviceId);
    CallLoadDist(cudaA, cudaB, cudaLoadDist, m, nnzArray, numMatrices);
    DeepAllocateSparseHost(hostOutput, m, n, nnzArray, numMatrices);
    DeepAllocateSparseCuda(cudaOutput, *hostOutput, numMatrices, deviceId);
    CalculateRowKernel<<<numMatrices, 64>>>(*cudaOutput, cudaA, cudaB,
                                            cudaLoadDist);

    if (copyResultToHost)
        DeepCopyDeviceToHost(*hostOutput, *cudaOutput, numMatrices, deviceId);

    free(nnzArray);
    DeepFreeLoadDistCuda(cudaLoadDist, numMatrices, deviceId);
    DeepFreeLoadDistHost(hostLoadDist, numMatrices);
}

__host__ void CallLoadDist(SparseMatrix* a, SparseMatrix* b,
                           LoadDistMatrix* loadDist, uint32_t M,
                           uint32_t* nnzArray, size_t numMatrices)
{
    const auto numLoops = 8;
    const auto threadDim = M / numLoops;
    uint32_t* deviceNNZArray = nullptr;
    cudaMalloc((void**)&deviceNNZArray, sizeof(uint32_t) * numMatrices);

    const auto blockDim =
        (numMatrices > MAX_GRID_DIM) ? numMatrices - MAX_GRID_DIM : numMatrices;

    if (blockDim > 0)
        LoadDistKernel<<<blockDim, threadDim>>>(loadDist, a, b, deviceNNZArray);
    if (numMatrices > blockDim)
    {
        SparseMatrix* offsetA = a + blockDim;
        SparseMatrix* offsetB = b + blockDim;
        LoadDistMatrix* loadDistOffset = loadDist + blockDim;

        const auto secondBlockDim = numMatrices - blockDim;
        LoadDistKernel<<<secondBlockDim, threadDim>>>(loadDistOffset, offsetA,
                                                      offsetB, deviceNNZArray);
    }
    cudaMemcpy(nnzArray, deviceNNZArray, sizeof(uint32_t) * numMatrices,
               cudaMemcpyDeviceToHost);
    cudaFree(deviceNNZArray);
}

__host__ void AllocateOutput(SparseMatrix* output, uint32_t m, uint32_t n,
                             size_t numMatrices, const uint32_t* nnzArray)
{
    for (uint32_t matrixIdx = 0; matrixIdx < numMatrices; ++matrixIdx)
    {
        SparseMatrix* curOutput = output + matrixIdx;
        curOutput->M = m;
        curOutput->N = n;
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
    __shared__ uint32_t nnzPerMatrix;
    //! ByteSize must be larger than Number of required blocks per row + 1

    const auto matrixIdx = blockIdx.x;
    const auto rowIdxBegin = threadIdx.x;
    const auto rowIdxStride = blockDim.x;

    SparseMatrix* curA = a + matrixIdx;
    SparseMatrix* curB = b + matrixIdx;
    LoadDistMatrix* curLoadDist = loadDist + matrixIdx;

    if (threadIdx.x == 0)
        nnzPerMatrix = 0;

    __syncthreads();

    for (auto rowIdxA = rowIdxBegin; rowIdxA < a[matrixIdx].M;
         rowIdxA += rowIdxStride)
    {
        auto sparseColIdxA = curA->ROW[rowIdxA];
        curLoadDist->ROW[rowIdxA] = curA->ROW[rowIdxA];
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
            nnzPerRow += numElemPerRowB;
        }
        atomicAdd_block(&nnzPerMatrix, nnzPerRow);
    }

    curLoadDist->NNZ = nnzPerMatrix;
    nnzArray[matrixIdx] = nnzPerMatrix;
}

__global__ void CalculateRowKernel(SparseMatrix* out, SparseMatrix* a,
                                   SparseMatrix* b, LoadDistMatrix* loadDist)
{
    //! Stores pair of computer value and pair of index
    __shared__ float tempValueArray[MAX_NNZ_PER_BLOCK_LARGE];
    __shared__ uint32_t tempIdxArray[MAX_NNZ_PER_BLOCK_LARGE];

    InitIndexArray(tempIdxArray, MAX_NNZ_PER_BLOCK_LARGE);

    const auto M = out[0].M;
    SparseMatrix* curOut = out + blockIdx.x / M;
    SparseMatrix* curA = a + blockIdx.x / M;
    SparseMatrix* curB = b + blockIdx.x / M;
    LoadDistMatrix* curLoadDist = loadDist + blockIdx.x / M;

    const auto curRowIdx = blockIdx.x % M;
    const auto sparseColIdxOffset = threadIdx.x;
    const auto sparseColIdxA = sparseColIdxOffset;

    const auto colIdxA = curA->COL[sparseColIdxA];
    const auto sparseColIdxBBegin = curB->ROW[colIdxA];
    const auto sparseColIdxBEnd = curB->ROW[colIdxA + 1];
    const auto nnz = curLoadDist->Load[curLoadDist->ROW[curRowIdx - 1]];

    if (sparseColIdxA < curA->ROW[curRowIdx])
    {
        for (uint32_t sparseColIdxB = sparseColIdxBBegin;
             sparseColIdxB < sparseColIdxBEnd; ++sparseColIdxB)
        {
            InsertHash(tempValueArray, tempIdxArray,
                       curA->V[sparseColIdxA] * curB->V[sparseColIdxB],
                       curB->COL[sparseColIdxB], MAX_NNZ_PER_BLOCK_LARGE);
        }
    }

    Sort(tempValueArray, tempIdxArray, MAX_NNZ_PER_BLOCK_LARGE);

    for (uint32_t idx = threadIdx.x; idx < nnz; idx += blockDim.x)
    {
        //! TODO : How are we going to store the result if row is distributed?
        curOut->V[idx] = tempValueArray[idx];
        curOut->COL[idx] = tempIdxArray[idx];
    }
}

__device__ void Sort(float* tempValArray, uint32_t* tempIdxArray,
                     uint32_t arraySize)
{
    const uint32_t maxLevel =
        __double2uint_rz(log2(__uint2float_rz(arraySize)));
    for (uint32_t level = 0; level < maxLevel; ++level)
    {
        const auto phase = __double2uint_rz(pow(2, level));
        for (auto stride = phase; stride > 0; stride /= 2)
        {
            for (uint32_t id = threadIdx.x; id <= arraySize / 2;
                 id += blockDim.x)
            {
                const auto sizePerBlock = stride * 2;
                const auto sizePerBlockPair = 2 * sizePerBlock;
                const bool direction = id / phase == 0;

                if ((id / stride) % 2 == 0)
                {
                    const auto idx = (id / sizePerBlock) * sizePerBlockPair +
                                     id % sizePerBlock;
                    const auto targetIdx = idx + stride;

                    if ((direction &&
                         tempIdxArray[idx] > tempIdxArray[targetIdx]) ||
                        (!direction &&
                         tempIdxArray[idx] < tempIdxArray[targetIdx]))
                    {
                        Swap(tempValArray + idx, tempValArray + targetIdx);
                        Swap(tempIdxArray + idx, tempIdxArray + targetIdx);
                    }
                }
                else
                {
                    const auto idx = ((arraySize / 2 - id) / sizePerBlock) *
                                         sizePerBlockPair +
                                     (arraySize / 2 - id) % sizePerBlock;
                    const auto targetIdx = idx + stride;

                    if ((direction &&
                         tempIdxArray[arraySize - idx] <
                             tempIdxArray[arraySize - targetIdx]) ||
                        (!direction && tempIdxArray[arraySize - idx] >
                                           tempIdxArray[arraySize - targetIdx]))
                    {
                        Swap(tempValArray + (arraySize - idx),
                             tempValArray + (arraySize - targetIdx));
                        Swap(tempIdxArray + (arraySize - idx),
                             tempIdxArray + (arraySize - targetIdx));
                    }
                }
            }
        }
        __syncthreads();
    }
}

__device__ void InsertHash(float* valueArray, uint32_t* idxArray, float value,
                           uint32_t index, uint32_t arraySize)
{
    auto key = Hash(index, arraySize);

    while (idxArray[key] != index && idxArray[key] != INF)
    {
        key = Hash(index + 1, arraySize);
    }
    if (idxArray[key] == index)
        atomicAdd_block(valueArray + key, value);
    else if (idxArray[key] == INF)
    {
        idxArray[key] = index;
        atomicExch_block(valueArray + key, value);
    }
}

__device__ void InitIndexArray(uint32_t* idxArray, uint32_t arraySize)
{
    const auto id = threadIdx.x;
    for (uint32_t idx = id; idx < arraySize; idx += blockDim.x)
    {
        idxArray[idx] = INF;
    }
}
}  // namespace Motutapu::Compute::Cuda::Sparse