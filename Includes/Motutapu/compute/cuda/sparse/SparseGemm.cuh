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
                            SparseMatrix* loadDist, size_t numMatrices);

//! Each block works for each matrix
//! Assigns number of calculation for each element
__global__ void CalculateLoadKernel(SparseMatrix* a, SparseMatrix* b,
                                    SparseMatrix* loadDist, size_t numMatrices);

//! Launches sparse matrix multiplication kernel
//! Each matrix is called simultaneously with streams
__host__ void CalculateRow(SparseMatrix* unmergedSparseMatrixRow,
                           SparseMatrix* a, SparseMatrix* b,
                           SparseMatrix* loadDist, uint32_t matrixNum);

//! Kernel for calculating sparse matrix
//! Each block is responsible for one row
//! Each thread will compute multiplications corresponding to one value in A's
//! row
__global__ void CalculateRowKernelRow(SparseMatrix* unmergedSparseMatrixRow,
                                      SparseMatrix* a, SparseMatrix* b,
                                      uint32_t maxSizePerValue,
                                      uint32_t rowIdx);

}  // namespace Motutapu::Compute

#endif  // MOTUTAPU_CALCULATELOAD_CUH
