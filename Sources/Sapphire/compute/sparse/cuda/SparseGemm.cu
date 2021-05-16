// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/cudaUtil/Memory.hpp>
#include <Sapphire/compute/sparse/Sparse.hpp>
#include <Sapphire/compute/sparse/cuda/SparseGemm.cuh>
#include <Sapphire/util/MemoryManager.hpp>
#include <cstdlib>

#define THREADS_PER_BLOCK 32

namespace Sapphire::Compute::Sparse::Cuda
{
using namespace Util;

__device__ uint32_t Hash1(uint32_t col, uint32_t numBuckets)
{
    return col % (numBuckets / 2);
}

__device__ uint32_t Hash2(uint32_t col, uint32_t numBuckets)
{
    return (numBuckets / 2) - col % (numBuckets / 2);
}

__host__ void GetLoadDist(LoadDistMatrix* hostLoadDist, SparseMatrix* cudaA,
                          SparseMatrix* cudaB, uint32_t m, size_t numMatrices,
                          int deviceId)
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
    auto* tempValueArray =
        static_cast<float*>(Util::MemoryManager::GetMemoryCuda(
            sizeof(float) * numMatrices * m * MAX_NNZ_PER_ROW, deviceId));

    auto* tempIdxArray =
        static_cast<uint32_t*>(Util::MemoryManager::GetMemoryCuda(
            sizeof(uint32_t) * numMatrices * m * MAX_NNZ_PER_ROW, deviceId));

    *cudaOutput = static_cast<SparseMatrix*>(MemoryManager::GetMemoryCuda(
        sizeof(SparseMatrix) * numMatrices, deviceId));
    auto* outputBuffer = static_cast<SparseMatrix*>(
        MemoryManager::GetMemoryHost(sizeof(SparseMatrix) * numMatrices));

    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        outputBuffer[i].ROW = static_cast<uint32_t*>(
            MemoryManager::GetMemoryCuda((m + 1) * sizeof(uint32_t), deviceId));
    }

    Compute::Cuda::CopyHostToDevice(*cudaOutput, outputBuffer,
                                    sizeof(SparseMatrix) * numMatrices);

    GemmKernel<<<numMatrices * m, THREADS_PER_BLOCK>>>(
        *cudaOutput, cudaA, cudaB, tempIdxArray, tempValueArray, m);
    const auto blockDim = (numMatrices % THREADS_PER_BLOCK == 0)
                              ? numMatrices / THREADS_PER_BLOCK
                              : numMatrices / THREADS_PER_BLOCK + 1;
    StackRowKernel<<<blockDim, THREADS_PER_BLOCK>>>(*cudaOutput, m,
                                                    numMatrices);

    for (uint32_t i = 0; i < numMatrices; ++i)
    {
        uint32_t nnz;
        Compute::Cuda::CopyDeviceToHost(&nnz, outputBuffer[i].ROW + m,
                                        sizeof(uint32_t));

        (outputBuffer + i)->V = static_cast<float*>(
            MemoryManager::GetMemoryCuda(sizeof(float) * nnz, deviceId));
        (outputBuffer + i)->COL = static_cast<uint32_t*>(
            MemoryManager::GetMemoryCuda(sizeof(uint32_t) * nnz, deviceId));

        (outputBuffer + i)->M = m;
        (outputBuffer + i)->N = n;
        (outputBuffer + i)->NNZ = nnz;
    }

    Compute::Cuda::CopyHostToDevice(*cudaOutput, outputBuffer,
                                    sizeof(SparseMatrix) * numMatrices);

    StoreOutput<<<numMatrices * m, THREADS_PER_BLOCK>>>(
        *cudaOutput, tempIdxArray, tempValueArray, m, numMatrices);

    if (copyResultToHost)
    {
        *hostOutput = static_cast<SparseMatrix*>(
            MemoryManager::GetMemoryHost(sizeof(SparseMatrix) * numMatrices));
        for (uint32_t i = 0; i < numMatrices; ++i)
        {
            (*hostOutput + i)->ROW = static_cast<uint32_t*>(
                MemoryManager::GetMemoryHost(sizeof(uint32_t) * (m + 1)));
            Compute::Cuda::CopyDeviceToHost((*hostOutput + i)->ROW,
                                            (outputBuffer + i)->ROW,
                                            sizeof(uint32_t) * (m + 1));

            const auto nnz = (*hostOutput + i)->ROW[m];
            (*hostOutput + i)->V = static_cast<float*>(
                MemoryManager::GetMemoryHost(sizeof(float) * nnz));
            (*hostOutput + i)->COL = static_cast<uint32_t*>(
                MemoryManager::GetMemoryHost(sizeof(float) * nnz));

            Compute::Cuda::CopyDeviceToHost((*hostOutput + i)->V,
                                            (outputBuffer + i)->V,
                                            sizeof(float) * nnz);
            Compute::Cuda::CopyDeviceToHost((*hostOutput + i)->COL,
                                            (outputBuffer + i)->COL,
                                            sizeof(uint32_t) * nnz);

            (*hostOutput + i)->M = m;
            (*hostOutput + i)->N = n;
            (*hostOutput + i)->NNZ = nnz;
        }
    }

    MemoryManager::DeReferenceHost(outputBuffer);
    Util::MemoryManager::DeReferenceCuda(tempValueArray, deviceId);
    Util::MemoryManager::DeReferenceCuda(tempIdxArray, deviceId);
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

    cudaMemcpy(nnzArray, deviceNNZArray, sizeof(uint32_t) * numMatrices,
               cudaMemcpyDeviceToHost);
    Util::MemoryManager::DeReferenceCuda(deviceNNZArray, deviceId);
}

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
#if __CUDA_ARCH__ < 600
        atomicAdd(&nnzPerMatrix, nnzPerRow);
#else
        atomicAdd_block(&nnzPerMatrix, nnzPerRow);
#endif
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        curLoadDist->ROW[m] = curA->ROW[m];
        nnzArray[matrixIdx] = nnzPerMatrix;
    }
}

__global__ void GemmKernel(SparseMatrix* out, SparseMatrix* a, SparseMatrix* b,
                           uint32_t* idxArray, float* valArray, uint32_t m)
{
    //! Stores pair of computer value and pair of index
    __shared__ float tempValueArray[MAX_NNZ_PER_ROW];
    __shared__ uint32_t tempIdxArray[MAX_NNZ_PER_ROW];
    __shared__ uint32_t nnz;

    InitIdxArray(tempIdxArray, MAX_NNZ_PER_ROW);
    if (threadIdx.x == 0)
        nnz = 0;

    const auto curRowIdx = blockIdx.x % m;
    const auto matrixOffset = blockIdx.x / m;

    if (curRowIdx == 0)
        out->ROW[m] = 0;

    SparseMatrix* curOut = out + matrixOffset;
    SparseMatrix* curA = a + matrixOffset;
    SparseMatrix* curB = b + matrixOffset;

    const auto rowNNZa = curA->ROW[curRowIdx + 1] - curA->ROW[curRowIdx];
    for (uint32_t sparseColIdxOffset = threadIdx.x;
         sparseColIdxOffset < rowNNZa; sparseColIdxOffset += blockDim.x)
    {
        const auto sparseColIdxA = curA->ROW[curRowIdx] + sparseColIdxOffset;

        const auto colIdxA = curA->COL[sparseColIdxA];
        const auto sparseColIdxBBegin = curB->ROW[colIdxA];
        const auto sparseColIdxBEnd = curB->ROW[colIdxA + 1];

        for (uint32_t sparseColIdxB = sparseColIdxBBegin;
             sparseColIdxB < sparseColIdxBEnd; ++sparseColIdxB)
        {
            InsertHash(tempValueArray, tempIdxArray, &nnz,
                       curA->V[sparseColIdxA] * curB->V[sparseColIdxB],
                       curB->COL[sparseColIdxB], MAX_NNZ_PER_ROW);
        }
    }
    __syncthreads();

    Sort(tempValueArray, tempIdxArray, MAX_NNZ_PER_ROW);

    __syncthreads();

    if (threadIdx.x == 0 && curRowIdx < m)
        curOut->ROW[curRowIdx] = nnz;

    const auto curOffset = MAX_NNZ_PER_ROW * (m * matrixOffset + curRowIdx);
    for (uint32_t idx = threadIdx.x; idx < nnz; idx += blockDim.x)
    {
        valArray[curOffset + idx] = tempValueArray[idx];
        idxArray[curOffset + idx] = tempIdxArray[idx];
    }
}

__global__ void StackRowKernel(SparseMatrix* out, uint32_t m,
                               uint32_t numMatrices)
{
    const auto matrixIdx = blockDim.x * blockIdx.x + threadIdx.x;

    if (matrixIdx < numMatrices)
    {
        SparseMatrix* curMatrix = out + matrixIdx;
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
    const auto arrayOffset = MAX_NNZ_PER_ROW * (M * matrixOffset + curRowIdx);
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
        for (uint32_t stride = phase; stride > 0; stride /= 2)
        {
            for (uint32_t id = threadIdx.x; id < arraySize / 2;
                 id += blockDim.x)
            {
                const auto sizePerBlock = stride * 2;
                const auto sizePerBlockPair = 2 * sizePerBlock;
                const bool direction = (id / phase) % 2 == 0;

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
                    auto idx = ((arraySize / 2 - id) / sizePerBlock) *
                                   sizePerBlockPair +
                               (arraySize / 2 - id) % sizePerBlock;
                    auto targetIdx = idx + stride;

                    idx = arraySize - idx;
                    targetIdx = arraySize - targetIdx;

                    if ((direction &&
                         tempIdxArray[idx] < tempIdxArray[targetIdx]) ||
                        (!direction &&
                         tempIdxArray[idx] > tempIdxArray[targetIdx]))
                    {
                        Swap(tempValArray + idx, tempValArray + targetIdx);
                        Swap(tempIdxArray + idx, tempIdxArray + targetIdx);
                    }
                }
            }
            __syncthreads();
        }
    }
}

__device__ void InsertHash(float* valueArray, uint32_t* idxArray, uint32_t* nnz,
                           float value, uint32_t index, uint32_t arraySize)
{
    auto key = Hash1(index, arraySize);

    uint32_t i = 1;
    while ((idxArray[key] != index && idxArray[key] != INF) ||
           idxArray[key] == DELETED_MARKER)
    {
        key =
            (Hash1(index, arraySize) + i * Hash2(index, arraySize)) % arraySize;
        i++;
    }

#if __CUDA_ARCH__ < 600
    if (atomicCAS(idxArray + key, INF, index) == INF)
    {
        atomicExch(valueArray + key, value);
        atomicAdd(nnz, 1);
    }
    else
    {
        atomicAdd(valueArray + key, value);
        if (valueArray[key] == 0.0f)
        {
            atomicExch(valueArray + key, DELETED_MARKER);
            atomicSub(nnz, 1);
        }
    }
#else
    if (atomicCAS_block(idxArray + key, INF, index) == INF)
    {
        atomicExch_block(valueArray + key, value);
        atomicAdd_block(nnz, 1);
    }
    else
    {
        atomicAdd_block(valueArray + key, value);
        if (valueArray[key] == 0.0f)
        {
            atomicExch_block(valueArray + key, DELETED_MARKER);
            atomicSub_block(nnz, 1);
        }
    }
#endif
}

__device__ void InitIdxArray(uint32_t* idxArray, uint32_t arraySize)
{
    const auto id = threadIdx.x;
    for (uint32_t idx = id; idx < arraySize; idx += blockDim.x)
    {
        idxArray[idx] = INF;
    }
}
}  // namespace Sapphire::Compute::Sparse::Cuda