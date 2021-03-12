// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_COMPUTE_MATRIXFORMAT_HPP
#define MOTUTAPU_COMPUTE_MATRIXFORMAT_HPP

#include <Motutapu/compute/SparseMatrix.hpp>
#include <cstdlib>

namespace Motutapu::Compute
{
void DeepAllocateSparseHost(SparseMatrix* target, size_t m, size_t nnz,
                            size_t numMatrices);

void DeepFreeSparseHost(SparseMatrix* target);

void DeepAllocateSparseCuda(SparseMatrix* cudaTarget, SparseMatrix* hostTarget,
                            size_t numMatrices, int deviceId);

void DeepFreeSparseCuda(SparseMatrix* cudaTarget, int deviceId);

void ConvertDenseToSparseHost(SparseMatrix* dst, float* src, size_t numRows,
                              size_t numCols, size_t numMatrices);

void ConvertDenseToSparseCuda(SparseMatrix* dst, float* src);

void ConvertSparseToDenseHost(float* dst, SparseMatrix* src);

void ConvertSparseToDenseCuda(float* dst, SparseMatrix* src);
}  // namespace Motutapu::Compute

#endif  // MOTUTAPU_MATRIXFORMAT_HPP
