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
void DeepAllocateSparseHost(SparseMatrix** targetPtr, uint32_t m, uint32_t nnz,
                            uint32_t numMatrices);

void DeepFreeSparseHost(SparseMatrix* target);

void DeepAllocateSparseCuda(SparseMatrix** cudaTargetPtr,
                            SparseMatrix* hostTarget, uint32_t numMatrices,
                            int deviceId);

void DeepFreeSparseCuda(SparseMatrix* cudaTarget, int deviceId);

void ConvertDenseToSparseHost(SparseMatrix* dst, float* src, size_t numRows,
                              size_t numCols, size_t numMatrices);

void ConvertDenseToSparseCuda(SparseMatrix* dst, float* src, size_t numRows,
                              size_t numCols, size_t numMatrices);

void ConvertSparseToDenseHost(float* dst, SparseMatrix* src, size_t numRows,
                              size_t numCols, size_t numMatrices);

void ConvertSparseToDenseCuda(float* dst, SparseMatrix* src, size_t numRows,
                              size_t numCols, size_t numMatrices);

//! Copies sparse matrix from device to host
void CopySparseDeviceToHost(SparseMatrix* dst, SparseMatrix* src,
                            size_t numMatrices);
//! Copies sparse matrix from host to device
void CopySparseHostToDevice(SparseMatrix* dst, SparseMatrix* src,
                            size_t numMatrices);

void LaunchSparseGemm(SparseMatrix* A, SparseMatrix* B, uint32_t numMatrices,
                      uint32_t numRows, uint32_t numCols, int deviceId);

void AllocateGemm(SparseMatrix* Out, SparseMatrix* A, SparseMatrix* B);

}  // namespace Motutapu::Compute

#endif  // MOTUTAPU_MATRIXFORMAT_HPP
