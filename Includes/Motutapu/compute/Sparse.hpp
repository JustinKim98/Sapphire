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
//! Deep allocates sparse matrix on host
//! \param targetPtr : ptr to allocate sparse matrix
//! \param m : array of number of rows
//! \param nnz : array of number of non-zeros
//! \param numMatrices : number of matrice
void DeepAllocateSparseHost(SparseMatrix** targetPtr, const uint32_t m[],
                            const uint32_t nnz[], uint32_t numMatrices);

//! Deep allocates load distribution matrix on host
//! \param targetPtr : ptr to allocate Load distribution matrix
//! \param m : array of number of rows
//! \param nnz : array of number of non-zeros
//! \param numMatrices : number of matrices
void DeepAllocateLoadDistMatrixHost(LoadDistMatrix** targetPtr,
                                    const uint32_t m[], const uint32_t nnz[],
                                    uint32_t numMatrices);

//! Frees sparse matrix on the host
//! \param target : ptr to the target sparse matrix array to free
//! \param numMatrices : number of matrices
void DeepFreeSparseHost(SparseMatrix* target, uint32_t numMatrices);

void DeepAllocateSparseCuda(SparseMatrix** cudaTargetPtr,
                            SparseMatrix* hostPtr, uint32_t numMatrices,
                            int deviceId);

void DeepAllocateLoadDistCuda(LoadDistMatrix** cudaTargetPtr,
                              LoadDistMatrix* hostPtr, uint32_t numMatrices,
                              int deviceId);

void DeepFreeSparseCuda(SparseMatrix* cudaTarget, int deviceId);

//! Copies sparse matrix on the GPU
//! \param destArray : Destination array on the Gpu
//! \param srcArray : Source array on the Gpu
//! \param size : Number of sparse matrices
void DeepCopySparseMatrixOnGpu(SparseMatrix* destArray, SparseMatrix* srcArray,
                               uint32_t size);

//! Copies sparse matrix on the GPU
//! \param destArray : Destination array on the Gpu
//! \param srcArray : Source array on the Gpu
//! \param size : Number of sparse matrices
void DeepCopySparseMatrixOnGpu(LoadDistMatrix* destArray,
                               LoadDistMatrix* srcArray, uint32_t size);

//! Deep copies host matrix to GPU from Host
//! \param deviceArray : Target device array to copy
//! \param hostArray : Source device array to copy from
//! \param size : Number of sparse matrices
void DeepCopyHostToGpu(SparseMatrix* deviceArray, SparseMatrix* hostArray,
                       uint32_t size);

//! Deep copies host matrix to GPU from Host
//! \param deviceArray : Target device array to copy
//! \param hostArray : Source device array to copy from
//! \param size : Number of sparse matrices
void DeepCopyHostToGpu(LoadDistMatrix* deviceArray, LoadDistMatrix* hostArray,
                       uint32_t size);

//! Deep copies host matrix to Host from Gpu
//! \param hostArray : Target device array to copy
//! \param deviceArray : Source device array to copy from
//! \param size : Number of sparse matrices
void DeepCopyGpuToHost(SparseMatrix* hostArray, SparseMatrix* deviceArray,
                       uint32_t size);

//! Deep copies host matrix to Host from Gpu
//! \param hostArray : Target device array to copy
//! \param deviceArray : Source device array to copy from
//! \param size : Number of sparse matrices
void DeepCopyGpuToHost(LoadDistMatrix* hostArray, LoadDistMatrix* deviceArray,
                       uint32_t size);

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
