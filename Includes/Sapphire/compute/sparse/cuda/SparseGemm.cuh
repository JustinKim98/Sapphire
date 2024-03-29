// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Sapphire_COMPUTE_CALCULATE_LOAD_CUH
#define Sapphire_COMPUTE_CALCULATE_LOAD_CUH

#include <Sapphire/compute/cudaUtil/CudaParams.cuh>
#include <Sapphire/compute/cudaUtil/Memory.hpp>
#include <Sapphire/compute/sparse/Sparse.hpp>

namespace Sapphire::Compute::Sparse::Cuda
{
__host__ void GetLoadDist(LoadDistMatrix* hostLoadDist, SparseMatrix* cudaA,
                          SparseMatrix* cudaB, uint32_t m, size_t numMatrices,
                          int deviceId);

//! Calculates Gemm by launching LoadDistKernel on the GPU
//! \param hostOutput : Array of output sparse matrices on the host memory
//! Required memory is automatically allocated
//! It's caller's responsibility to free the allocated memory
//! \param cudaOutput : Array of output sparse matrices on the device Memory
//! Required memory is automatically allocated
//! It's caller's responsibility to free the allocated memory
//! \param cudaA : Array of sparse matrix for operand a on device memory.
//! Must be dense allocated
//! \param hostB : Array of sparse matrix for operand a on host memory.
//! Must be dense allocated
//! \param cudaB : Array of sparse matrix for operand b on device memory.
//! Must be dense allocated
//! \param m : Expected number of rows for output matrix
//! \param n : Expected number of columns for output matrix
//! \param numMatrices : number of matrices to compute Gemm
//! \param DeviceId : Device to perform the computation
//! \param copyResultToHost : If true, copies the result to host output.
__host__ void Gemm(SparseMatrix** hostOutput, SparseMatrix** cudaOutput,
                   SparseMatrix* cudaA, SparseMatrix* cudaB, uint32_t m,
                   uint32_t n, size_t numMatrices, int deviceId,
                   bool copyResultToHost);

__host__ void CallLoadDist(SparseMatrix* a, SparseMatrix* b,
                           LoadDistMatrix* loadDist, uint32_t M,
                           uint32_t* nnzArray, size_t numMatrices,
                           int deviceId);

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
__global__ void GemmKernel(SparseMatrix* out, SparseMatrix* a, SparseMatrix* b,
                           uint32_t* idxArray, float* valArray, uint32_t m);

__global__ void StackRowKernel(SparseMatrix* out, uint32_t m,
                               uint32_t numMatrices);

__global__ void StoreOutput(SparseMatrix* out, const uint32_t* idxArray,
                            const float* valArray, uint32_t M,
                            uint32_t numMatrices);

//! Sorts the array in increasing order using bitonic esc sort algorithm
//! \param tempValArray : array of values. Its size must be power of 2
//! \param tempIdxArray : Array of indices. Its size must be identical to 2
//! \param arraySize : Size of the array. Must be power of 2
__device__ void Sort(float* tempValArray, uint32_t* tempIdxArray,
                     uint32_t arraySize);

//! Merges the array sorted by Sort function.

__device__ void Merge(float* tempValArray, uint32_t* tempIdxArray,
                      uint32_t* numMergedElements, uint32_t numElements);

__device__ void InsertHash(float* valueArray, uint32_t* idxArray, uint32_t* nnz,
                           float value, uint32_t index, uint32_t arraySize);

template <typename T>
__device__ void Swap(T* a, T* b)
{
    auto temp = *a;
    *a = *b;
    *b = temp;
}

}  // namespace Sapphire::Compute::Sparse::Cuda

#endif  // Sapphire_CALCULATELOAD_CUH
