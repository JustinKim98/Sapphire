// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_COMPUTE_CALCULATELOAD_CUH
#define MOTUTAPU_COMPUTE_CALCULATELOAD_CUH

#include <Motutapu/compute/Sparse.hpp>
#include <Motutapu/compute/cuda/Memory.cuh>

namespace Motutapu::Compute
{
__host__ void CalculateLoad(SparseMatrix* a, SparseMatrix* b,
                            LoadDistMatrix* loadDist, size_t numMatrices);

//! Each block works for each matrix
//! Assigns number of calculation for each element
__global__ void CalculateLoadKernel(LoadDistMatrix* loadDist,
                                    SparseMatrix* a, SparseMatrix* b,
                                    size_t numMatrices);

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
__global__ void CalculateRowKernel(float* cV, uint32_t* cCOL, SparseMatrix* a,
                                   SparseMatrix* b,
                                   LoadDistMatrix* stackedLoadDist,
                                   uint32_t rowIdx,
                                   uint32_t sparseColIndexBegin,
                                   uint32_t sparseColIndexEnd, uint32_t nnz);

__device__ void Sort(float* tempValueColIdxPair, uint32_t* tempIdxArray,
                     uint32_t numElements);

__device__ void Merge(float* tempValueColIdxPair, uint32_t* tempIdxArray,
                      uint32_t numElements);

}  // namespace Motutapu::Compute

#endif  // MOTUTAPU_CALCULATELOAD_CUH
