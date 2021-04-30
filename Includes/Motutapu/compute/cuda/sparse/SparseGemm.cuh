// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_COMPUTE_CALCULATE_LOAD_CUH
#define MOTUTAPU_COMPUTE_CALCULATE_LOAD_CUH

#include <Motutapu/compute/Sparse.hpp>
#include <Motutapu/compute/cuda/Memory.cuh>

namespace Motutapu::Compute::Sparse
{
//! Calculates Gemm by launching LoadDistKernel on the GPU
//! \param output : Array of output sparse matrices. output must be shallow
//! allocated.
//! \param a : Array of sparse matrix for operand a. Must be dense
//! allocated
//! \param b : Array of sparse matrix for operand b. Must be dense
//! allocated
//! \param loadDist : array of load distribution matrix. Must be dense
//! allocated
//! \param numMatrices : number of matrices to compute Gemm
__host__ void Gemm(SparseMatrix* output, SparseMatrix* a, SparseMatrix* b,
                   LoadDistMatrix* loadDist, size_t numMatrices);

__host__ void CallLoadDist(SparseMatrix* a, SparseMatrix* b,
                           LoadDistMatrix* loadDist, uint32_t* nnzArray,
                           size_t numMatrices);

__host__ void AllocateOutput(SparseMatrix* output, SparseMatrix* a,
                             SparseMatrix* b, size_t numMatrices,
                             const uint32_t* nnzArray);

//! Each block works for each matrix
//! Assigns number of calculation for each element
//! -- Constraints --
//! All matrices in the same array must have same shape
//! All matrix arrays must be pre-allocated
//! All matrix arrays must be in same size
//! \param loadDist : Array of load distribution matrix
//! \param a : Array of input sparse matrices for operand A.
//! \param b : Array of Input sparse matrices for operand B.
//! \param nnzArray : Array of non-zeros for each matrix. Its size must be
//! identical to size of matrix arrays.
__global__ void LoadDistKernel(LoadDistMatrix* loadDist, SparseMatrix* a,
                               SparseMatrix* b, uint32_t* nnzArray);

//! Kernel for calculating sparse matrix
//! Each block is responsible for one row
//! Each thread will compute multiplications corresponding to one value in A's
//! row
//! -- Constraints --
//! All matrices in the same array must have same shape.
//! All matrix arrays must be pre-allocated.
//! All matrix arrays must be in same size
//! loadDist matrix should have same size and shape with matrix A
//! \param out : Array of output sparse matrices
//! \param a : Array of input sparse matrices for operand a.
//! \param b : Array of input sparse matrices for operand b.
//! \param loadDist : Array of load distribution matrices.
//! \param sparseColIdxBegin : Start index of computation for matrix A.
//! \param sparseColIdxEnd : Last index + 1 of computation for matrix A.
__global__ void CalculateRowKernel(SparseMatrix* out, SparseMatrix* a,
                                   SparseMatrix* b, LoadDistMatrix* loadDist);

//! Sorts the array in increasing order using bitonic esc sort algorithm
//! \param tempValArray : array of values. Its size must be power of 2
//! \param tempIdxArray : Array of indices. Its size must be identical to 2
//! \param arraySize : Size of the array. Must be power of 2
__device__ void Sort(float* tempValArray, uint32_t* tempIdxArray,
                     uint32_t arraySize);

//! Merges the array sorted by Sort function.

__device__ void Merge(float* tempValArray, uint32_t* tempIdxArray,
                      uint32_t* numMergedElements, uint32_t numElements);

__device__ void InsertHash(float* valueArray, uint32_t* idxArray, float value,
                           uint32_t index, uint32_t arraySize);

__device__ void InitIndexArray(uint32_t* idxArray, uint32_t arraySize);

template <typename T>
__device__ void Swap(T* a, T* b)
{
    auto temp = *a;
    *a = *b;
    *b = temp;
}

__device__ uint32_t Hash(uint32_t col, uint32_t numBuckets)
{
    return col % numBuckets;
}

}  // namespace Motutapu::Compute::Sparse

#endif  // MOTUTAPU_CALCULATELOAD_CUH
