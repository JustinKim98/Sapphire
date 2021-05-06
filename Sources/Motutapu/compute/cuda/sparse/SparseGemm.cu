// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <stdint-gcc.h>
#include <Motutapu/compute/Sparse.hpp>
#include <Motutapu/compute/cuda/sparse/SparseGemm.cuh>
#include <Motutapu/util/MemoryManager.hpp>

#include <cstdlib>

#define MAX_NNZ_PER_BLOCK 512
#define THREADS_PER_BLOCK 16
#define INF (~0)

namespace Motutapu::Compute::Cuda::Sparse
{
__device__ uint32_t Hash(uint32_t col, uint32_t numBuckets)
{
    return col % numBuckets;
}

__host__ void GetLoadDist(LoadDistMatrix* hostLoadDist, SparseMatrix* hostA,
                          SparseMatrix* cudaA, SparseMatrix* cudaB, uint32_t m,
                          size_t numMatrices, int deviceId)
{
    LoadDistMatrix* cudaLoadDist;
    auto* nnzArray =
        static_cast<uint32_t*>(malloc(sizeof(uint32_t) * numMatrices));
    DeepAllocateLoadDistCuda(&cudaLoadDist, hostLoadDist, numMatrices,
                             deviceId);
    CallLoadDist(cudaA, cudaB, cudaLoadDist, m, nnzArray, numMatrices,
                 deviceId);
    DeepCopyDeviceToHost(hostLoadDist, cudaLoadDist, numMatrices, deviceId);

    free(nnzArray);
    DeepFreeLoadDistCuda(cudaLoadDist, numMatrices, deviceId);
}

__host__ void Gemm(SparseMatrix** hostOutput, SparseMatrix** cudaOutput,
                   SparseMatrix* hostA, SparseMatrix* cudaA,
                   SparseMatrix* cudaB, uint32_t m, uint32_t n,
                   size_t numMatrices, int deviceId, bool copyResultToHost)
{
    LoadDistMatrix *hostLoadDist, *cudaLoadDist;
    auto* nnzArray = static_cast<uint32_t*>(
        Util::MemoryManager::GetMemoryHost(sizeof(uint32_t) * numMatrices));
    DeepAllocateLoadDistHost(&hostLoadDist, hostA, numMatrices);
    DeepAllocateLoadDistCuda(&cudaLoadDist, hostLoadDist, numMatrices,
                             deviceId);
    CallLoadDist(cudaA, cudaB, cudaLoadDist, m, nnzArray, numMatrices, 0);
    DeepAllocateSparseHost(hostOutput, m, n, nnzArray, numMatrices);
    DeepAllocateSparseCuda(cudaOutput, *hostOutput, numMatrices, deviceId);
    auto* tempValueArray =
        static_cast<float*>(Util::MemoryManager::GetMemoryCuda(
            sizeof(float) * numMatrices * m * MAX_NNZ_PER_BLOCK, deviceId));

    auto* tempIdxArray =
        static_cast<uint32_t*>(Util::MemoryManager::GetMemoryCuda(
            sizeof(uint32_t) * numMatrices * m * MAX_NNZ_PER_BLOCK, deviceId));

    Calculate<<<numMatrices * m, THREADS_PER_BLOCK>>>(
        *cudaOutput, cudaA, cudaB, cudaLoadDist, tempIdxArray, tempValueArray,
        m, numMatrices);
    const auto blockDim =
        (numMatrices % 32 == 0) ? numMatrices / 32 : numMatrices / 32 + 1;
    StackRowKernel<<<blockDim, THREADS_PER_BLOCK>>>(*cudaOutput, numMatrices);
    StoreOutput<<<numMatrices * m, THREADS_PER_BLOCK>>>(
        *cudaOutput, tempIdxArray, tempValueArray, m, numMatrices);

    if (copyResultToHost)
        DeepCopyDeviceToHost(*hostOutput, *cudaOutput, numMatrices, deviceId);

    Util::MemoryManager::DeReferenceCuda(tempValueArray, deviceId);
    Util::MemoryManager::DeReferenceCuda(tempIdxArray, deviceId);
    Util::MemoryManager::DeReferenceHost(nnzArray);
    DeepFreeLoadDistCuda(cudaLoadDist, numMatrices, deviceId);
    DeepFreeLoadDistHost(hostLoadDist, numMatrices);
}

__host__ void CallLoadDist(SparseMatrix* a, SparseMatrix* b,
                           LoadDistMatrix* loadDist, uint32_t M,
                           uint32_t* nnzArray, size_t numMatrices, int deviceId)
{
    const auto threadDim = 32;
    auto* deviceNNZArray =
        static_cast<uint32_t*>(Util::MemoryManager::GetMemoryCuda(
            sizeof(uint32_t) * numMatrices, deviceId));

    const auto blockDim =
        (numMatrices > MAX_GRID_DIM) ? numMatrices - MAX_GRID_DIM : numMatrices;

    if (blockDim > 0)
        LoadDistKernel<<<blockDim, threadDim>>>(loadDist, a, b, deviceNNZArray);
    //    if (numMatrices > blockDim)
    //    {
    //        SparseMatrix* offsetA = a + blockDim;
    //        SparseMatrix* offsetB = b + blockDim;
    //        LoadDistMatrix* loadDistOffset = loadDist + blockDim;
    //
    //        const auto secondBlockDim = numMatrices - blockDim;
    //        LoadDistKernel<<<secondBlockDim, threadDim>>>(loadDistOffset,
    //        offsetA,
    //                                                      offsetB,
    //                                                      deviceNNZArray);
    //    }
    cudaMemcpy(nnzArray, deviceNNZArray, sizeof(uint32_t) * numMatrices,
               cudaMemcpyDeviceToHost);
    Util::MemoryManager::DeReferenceCuda(deviceNNZArray, deviceId);
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

    const auto m = a[matrixIdx].M;
    for (auto rowIdxA = rowIdxBegin; rowIdxA < m; rowIdxA += rowIdxStride)
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

            if (sparseColIdxA != curA->ROW[rowIdxA])
            {
                //! Load will stack as row advances
                curLoadDist->Load[sparseColIdxA] +=
                    curLoadDist->Load[sparseColIdxA - 1];
            }
            nnzPerRow += numElemPerRowB;
        }
        atomicAdd_block(&nnzPerMatrix, nnzPerRow);
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        curLoadDist->ROW[m] = curA->ROW[m];
        nnzArray[matrixIdx] = nnzPerMatrix;
    }
}

__global__ void Calculate(SparseMatrix* out, SparseMatrix* a, SparseMatrix* b,
                          LoadDistMatrix* loadDist, uint32_t* idxArray,
                          float* valArray, uint32_t m, uint32_t numMatrices)
{
    //! Stores pair of computer value and pair of index
    __shared__ float tempValueArray[MAX_NNZ_PER_BLOCK];
    __shared__ uint32_t tempIdxArray[MAX_NNZ_PER_BLOCK];
    __shared__ uint32_t nnz;

    InitIdxArray(tempIdxArray, MAX_NNZ_PER_BLOCK);
    if (threadIdx.x == 0)
        nnz = 0;

    const auto curRowIdx = blockIdx.x % m;
    const auto matrixOffset = blockIdx.x / m;

    if (curRowIdx == 0)
        out->ROW[m] = 0;

    SparseMatrix* curOut = out + matrixOffset;
    SparseMatrix* curA = a + matrixOffset;
    SparseMatrix* curB = b + matrixOffset;
    LoadDistMatrix* curLoadDist = loadDist + matrixOffset;

    const auto sparseColIdxOffset = threadIdx.x;
    const auto sparseColIdxA = curA->ROW[curRowIdx] + sparseColIdxOffset;

    if (sparseColIdxA < curA->ROW[curRowIdx + 1] &&
        curLoadDist->ROW[curRowIdx] < curLoadDist->ROW[curRowIdx + 1])
    {
        const auto colIdxA = curA->COL[sparseColIdxA];
        const auto sparseColIdxBBegin = curB->ROW[colIdxA];
        const auto sparseColIdxBEnd = curB->ROW[colIdxA + 1];

        if (sparseColIdxA < curA->ROW[curRowIdx + 1])
        {
            for (uint32_t sparseColIdxB = sparseColIdxBBegin;
                 sparseColIdxB < sparseColIdxBEnd; ++sparseColIdxB)
            {
                InsertHash(tempValueArray, tempIdxArray, &nnz,
                           curA->V[sparseColIdxA] * curB->V[sparseColIdxB],
                           curB->COL[sparseColIdxB], MAX_NNZ_PER_BLOCK);
            }
        }
    }

    __syncthreads();

    Sort(tempValueArray, tempIdxArray, MAX_NNZ_PER_BLOCK);

    __syncthreads();

    if (threadIdx.x == 0 && curRowIdx < m)
        curOut->ROW[curRowIdx] = nnz;

    const auto curOffset = MAX_NNZ_PER_BLOCK * (m * matrixOffset + curRowIdx);
    for (uint32_t idx = threadIdx.x; idx < nnz; idx += blockDim.x)
    {
        valArray[curOffset + idx] = tempValueArray[idx];
        idxArray[curOffset + idx] = tempIdxArray[idx];
    }
}

__global__ void StackRowKernel(SparseMatrix* out, uint32_t numMatrices)
{
    const auto matrixIdx = blockDim.x * blockIdx.x + threadIdx.x;

    if (matrixIdx < numMatrices)
    {
        SparseMatrix* curMatrix = out + matrixIdx;
        const auto m = curMatrix->M;
        uint32_t stackedOffset = 0;
        for (uint32_t rowIdx = 0; rowIdx < m; ++rowIdx)
        {
            const auto temp = curMatrix->ROW[rowIdx];
            curMatrix->ROW[rowIdx] = stackedOffset;
            stackedOffset += temp;
        }
        curMatrix->ROW[m] = stackedOffset;
        curMatrix->NNZ = stackedOffset;
    }
}

__global__ void StoreOutput(SparseMatrix* out, const uint32_t* idxArray,
                            const float* valArray, uint32_t M,
                            uint32_t numMatrices)
{
    const auto curRowIdx = blockIdx.x % M;
    const auto matrixOffset = blockIdx.x / M;

    SparseMatrix* curOut = out + matrixOffset;
    const auto arrayOffset = MAX_NNZ_PER_BLOCK * (M * matrixOffset + curRowIdx);
    const auto nnz = curOut->ROW[curRowIdx + 1] - curOut->ROW[curRowIdx];
    const auto colOffset = curOut->ROW[curRowIdx];
    for (auto i = threadIdx.x; i < nnz; i += blockDim.x)
    {
        curOut->COL[colOffset + i] = idxArray[arrayOffset + i];
        curOut->V[colOffset + i] = valArray[arrayOffset + i];
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
            for (uint32_t id = threadIdx.x; id < arraySize / 2;
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

__device__ void InsertHash(float* valueArray, uint32_t* idxArray, uint32_t* nnz,
                           float value, uint32_t index, uint32_t arraySize)
{
    auto key = Hash(index, arraySize);

    while (idxArray[key] != index && idxArray[key] != INF)
    {
        key = Hash(index + 1, arraySize);
    }

    if (atomicCAS_block(idxArray + key, INF, index) == INF)
    {
        atomicExch_block(valueArray + key, value);
        atomicAdd_block(nnz, 1);
    }
    else
    {
        atomicAdd_block(valueArray + key, value);
    }
}

__device__ void InitIdxArray(uint32_t* idxArray, uint32_t arraySize)
{
    const auto id = threadIdx.x;
    for (uint32_t idx = id; idx < arraySize; idx += blockDim.x)
    {
        idxArray[idx] = INF;
    }
}
}  // namespace Motutapu::Compute::Cuda::Sparse