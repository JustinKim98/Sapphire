// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_COMPUTE_MATRIXFORMAT_HPP
#define MOTUTAPU_COMPUTE_MATRIXFORMAT_HPP

#include <Motutapu/compute/SparseMatrix.hpp>
#include <cstdlib>

namespace Motutapu::Compute::Cuda::Sparse
{
//! Deep allocates sparse matrix on host
//! \param sparseMatrixArray : ptr to allocate sparse matrix
//! \param m : array of number of rows
//! \param nnz : array of number of non-zeros
//! \param numMatrices : number of matrices
void DeepAllocateSparseHost(SparseMatrix** sparseMatrixArray, const uint32_t m[],
                            const uint32_t nnz[], uint32_t numMatrices);

//! Deep allocates load distribution matrix on host
//! \param loadDistArray : ptr to allocate Load distribution matrix
//! \param m : array of number of rows
//! \param nnz : array of number of non-zeros
//! \param numMatrices : number of matrices
void DeepAllocateLoadDistMatrixHost(LoadDistMatrix** loadDistArray,
                                    SparseMatrix* sparseArray,
                                    uint32_t numMatrices);

//! Frees sparse matrix on the host
//! \param target : ptr to the target sparse matrix array to free
//! \param numMatrices : number of matrices
void DeepFreeSparseHost(SparseMatrix* target, uint32_t numMatrices);

//! Allocates sparse matrix array on the GPU
//! \param targetPtr : ptr to allocate the sparse matrix array
//! \param hostPtr : ptr to the previously allocated host
//! \param numMatrices : number of matrices
//! \param deviceId : ID of the device to allocate
void DeepAllocateSparseCuda(SparseMatrix** targetPtr, SparseMatrix* hostPtr,
                            uint32_t numMatrices, int deviceId);

//! Allocates load distribution matrix array on the GPU
//! \param targetPtr : ptr to the allocation target
//! \param hostPtr : ptr to the previously allocated host
//! \param numMatrices : number of matrices
//! \param deviceId : ID of the device to allocate
void DeepAllocateLoadDistCuda(LoadDistMatrix** targetPtr,
                              LoadDistMatrix* hostPtr, uint32_t numMatrices,
                              int deviceId);

//! Frees sparse matrix allocated on the GPU
//! \param targetPtr : Array of targetPtr sparse matrix to free
//! \param deviceId : ID of the device that owns the targetPtr
void DeepFreeSparseCuda(SparseMatrix* targetPtr, uint32_t numMatrices,
                        int deviceId);

//! Frees load distribution matrix allocated on the GPU
//! \param targetPtr : Array of targetPtr sparse matrix to free
//! \param deviceId : ID of the device that owns the targetPtr
void DeepFreeLoadDistCuda(LoadDistMatrix* targetPtr, uint32_t numMatrices,
                          int deviceId);

//! Copies sparse matrix on the GPU
//! All pointers given as parameters must be allocated with size of
//! numMatrices*sizeof(SparseMatrix) bytes
//! \param gpuDstArray : Destination array on
//! the GPU
//! \param gpuSrcArray : Source array on the GPU
//! \param hostDstArray :
//! Destination array on the Host
//! \param hostSrcArray : Source array on the Host
//! \param numMatrices : Number of sparse matrices
//! \param deviceId : ID of the device that owns both gpuDstArray and
//! gpuSrcArray
void DeepCopyGpuToGpu(SparseMatrix* gpuDstArray, SparseMatrix* gpuSrcArray,
                      uint32_t numMatrices, int deviceId);

//! Copies load distribution matrix on the GPU
//! All pointers given as parameters must be allocated with size of
//! numMatrices*sizeof(SparseMatrix) bytes
//! \param gpuDstArray : Destination array on
//! the GPU
//! \param gpuSrcArray : Source array on the GPU
//! \param hostDstArray :
//! Destination array on the Host
//! \param hostSrcArray : Source array on the Host
//! \param numMatrices : Number of sparse matrices to copy
//! \param deviceId : ID of the device that owns both gpuDstArray and
//! gpuSrcArray
void DeepCopyGpuToGpu(LoadDistMatrix* gpuDstArray, LoadDistMatrix* gpuSrcArray,
                      uint32_t numMatrices, int deviceId);

//! Deep copies host matrix to GPU from Host
//! \param dstArray : Target device array to copy
//! \param srcArray : Source device array to copy from
//! \param numMatrices : Number of sparse matrices to copy
void DeepCopyHostToGpu(SparseMatrix* gpuDstArray, SparseMatrix* hostSrcArray,
                       uint32_t numMatrices, int deviceId);

//! Deep copies host matrix to GPU from Host
//! \param gpuDstArray : Target device array to copy
//! \param hostSrcArray : Source device array to copy from
//! \param numMatrices : Number of sparse matrices
void DeepCopyHostToGpu(LoadDistMatrix* gpuDstArray,
                       LoadDistMatrix* hostSrcArray, uint32_t numMatrices,
                       int deviceId);

//! Deep copies host matrix to Host from Gpu
//! \param hostDstArray : Destination array on the Host
//! \param gpuSrcArray : Source array on the GPU
//! \param hostSrcArray : Source array on the Host
//! \param numMatrices : Number of sparse matrices to copy
void DeepCopyGpuToHost(SparseMatrix* hostDstArray, SparseMatrix* gpuSrcArray,
                       uint32_t numMatrices);

//! Deep copies host matrix to Host from Gpu
//! \param hostArray : Target device array to copy
//! \param deviceArray : Source device array to copy from
//! \param size : Number of sparse matrices
void DeepCopyGpuToHost(LoadDistMatrix* hostDstArray,
                       LoadDistMatrix* gpuSrcArray, uint32_t numMatrices);

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

void LaunchSparseGemm(SparseMatrix* C, SparseMatrix* A, SparseMatrix* B,
                      uint32_t numMatrices, uint32_t numRows, uint32_t numCols,
                      int deviceId);

void AllocateGemm(SparseMatrix* Out, SparseMatrix* A, SparseMatrix* B);

}  // namespace Motutapu::Compute::Cuda::Sparse

#endif  // MOTUTAPU_MATRIXFORMAT_HPP
