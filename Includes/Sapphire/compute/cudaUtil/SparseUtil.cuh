//
// Created by jwkim98 on 2021/05/17.
//

#ifndef SAPPHIRE_SPARSEUTIL_CUH
#define SAPPHIRE_SPARSEUTIL_CUH

#include <Sapphire/compute/cudaUtil/CudaParams.cuh>
namespace Sapphire::Compute::Cuda
{
__device__ void InsertHash(float* valueArray, uint32_t* idxArray, uint32_t* nnz,
                           float value, uint32_t index, uint32_t arraySize);

//! Sorts the array in increasing order using bitonic esc sort algorithm
//! \param tempValArray : array of values. Its size must be power of 2
//! \param tempIdxArray : Array of indices. Its size must be identical to 2
//! \param arraySize : Size of the array. Must be power of 2
__device__ void Sort(float* tempValArray, uint32_t* tempIdxArray,
                     uint32_t arraySize);

template <typename T>
__device__ void Swap(T* a, T* b)
{
    auto temp = *a;
    *a = *b;
    *b = temp;
}
}  // namespace Sapphire::Compute::Cuda

#endif  // SAPPHIRE_SPARSEUTIL_CUH
