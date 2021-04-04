// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_COMPUTE_CALCULATELOAD_CUH
#define MOTUTAPU_COMPUTE_CALCULATELOAD_CUH

#include <Motutapu/compute/cuda/Memory.cuh>
#include <Motutapu/compute/cuda/sparse/Sparse.hpp>

namespace Motutapu::Compute::Sparse
{
//! Calculates Gemm by launching GemmKernel on the GPU
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

//! Each block works for each matrix
//! Assigns number of calculation for each element

__global__ void GemmKernel(LoadDistMatrix* loadDist, SparseMatrix* output,
                           SparseMatrix* a, SparseMatrix* b);

//! Launches sparse matrix multiplication kernel
//! Each matrix is called simultaneously with streams
//! \param c : input&output c
//! \param a : input a
//! \param b : input b
//! \param loadDist : matrix containing load distribution. This function will
//! change load distribution into stacked load distribution
//! \param matrixNum : number of matrices in a batch \param nnzPerBlock : number
//! of maximum non zeros per block
//! \param nnzPerBlock : Maximum number of non zeros allowed for each block.
//! This parameter also denotes size of (value, colIdx) tuples allocated for
//! each block. This parameter must be power of 2
__host__ void CalculateGemm(SparseMatrix* c, const SparseMatrix* a,
                            const SparseMatrix* b, LoadDistMatrix* loadDist,
                            uint32_t matrixNum);

//! Kernel for calculating sparse matrix
//! Each block is responsible for one row
//! Each thread will compute multiplications corresponding to one value in A's
//! row

__global__ void CalculateRowKernel(SparseMatrix* out, SparseMatrix* a,
                                   SparseMatrix* b, LoadDistMatrix* loadDist,
                                   uint32_t rowIdx,
                                   uint32_t sparseColIndexBegin,
                                   uint32_t sparseColIndexEnd);

__device__ void Sort(float* tempValueColIdxPair, uint32_t* tempIdxArray,
                     uint32_t numElements);

__device__ void Merge(float* tempValueColIdxPair, uint32_t* tempIdxArray,
                      uint32_t numElements);

}  // namespace Motutapu::Compute::Sparse

#endif  // MOTUTAPU_CALCULATELOAD_CUH
