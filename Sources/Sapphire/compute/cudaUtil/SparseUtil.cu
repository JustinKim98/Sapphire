// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/cudaUtil/SparseUtil.cuh>
#include <Sapphire/compute/sparse/SparseMatrix.hpp>

namespace Sapphire::Compute::Cuda
{
__device__ uint32_t Hash1(uint32_t col, uint32_t numBuckets)
{
    return col % (numBuckets / 2);
}

__device__ uint32_t Hash2(uint32_t col, uint32_t numBuckets)
{
    return (numBuckets / 2) - col % (numBuckets / 2);
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
}  // namespace Sapphire::Compute::Cuda