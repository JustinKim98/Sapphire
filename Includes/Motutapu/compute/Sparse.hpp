// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_COMPUTE_MATRIXFORMAT_HPP
#define MOTUTAPU_COMPUTE_MATRIXFORMAT_HPP

#include "../../../../../../../../usr/include/c++/9/cstdlib"
#include "SparseMatrix.hpp"

namespace Motutapu::Compute::Sparse
{
//! Deep allocates sparse matrix on host
//! \param sparseMatrixArray : ptr to allocate sparse matrix
//! \param m : number of rows
//! \param n : number of columns
//! \param nnz : array of number of non-zeros
//! \param numMatrices : number of matrices
void DeepAllocateSparseHost(SparseMatrix** sparseMatrixArray, const uint32_t m,
                            const uint32_t n, const uint32_t nnz[],
                            uint32_t numMatrices);

//! Deep allocates load distribution matrix on host
//! \param loadDistArray : ptr to allocate Load distribution matrix array
//! \param sparseArray : ptr to the allocated sparse matrix array
//! \param numMatrices : number of matrices
void DeepAllocateLoadDistHost(LoadDistMatrix** loadDistArray,
                              SparseMatrix* sparseArray, uint32_t numMatrices);

//! Frees sparse matrix array on the host
//! \param sparseMatrixArray : ptr to the sparseMatrixArray sparse matrix array
//! to free
//! \param numMatrices : number of matrices
void DeepFreeSparseHost(SparseMatrix* sparseMatrixArray, uint32_t numMatrices);

//! Frees Load distribution matrix array on the host
//! \param loadDistArray : ptr to the loadDistArray load distribution matrix
//! array to free
//! \param numMatrices : number of matrices
void DeepFreeLoadDistHost(LoadDistMatrix* loadDistArray, uint32_t numMatrices);

//! Allocates sparse matrix array on the GPU
//! \param deviceSparseMatrixArray : ptr to allocate the sparse matrix array
//! \param hostSparseMatrixArray : ptr to the previously allocated host
//! \param numMatrices : number of matrices
//! \param deviceId : ID of the device to allocate
void DeepAllocateSparseCuda(SparseMatrix** deviceSparseMatrixArray,
                            SparseMatrix* hostSparseMatrixArray,
                            uint32_t numMatrices, int deviceId);

//! Allocates load distribution matrix array on the GPU
//! \param deviceLoadDistArray : ptr to the allocation target load distribution
//! matrix array
//! \param hostLoadDistArray : ptr to the previously allocated host
//! load distribution matrix array
//! \param numMatrices : number of matrices
//! \param deviceId : ID of the device to allocate
void DeepAllocateLoadDistCuda(LoadDistMatrix** deviceLoadDistArray,
                              LoadDistMatrix* hostLoadDistArray,
                              uint32_t numMatrices, int deviceId);

//! Frees sparse matrix allocated on the GPU
//! \param sparseMatrixArray : Array of sparseMatrixArray sparse matrix to free
//! on the device
//! \param deviceId : ID of the device that owns the
//! sparseMatrixArray
void DeepFreeSparseCuda(SparseMatrix* sparseMatrixArray, uint32_t numMatrices,
                        int deviceId);

//! Frees load distribution matrix allocated on the GPU
//! \param loadDistArray : Array of loadDistArray load distribution matrix to
//! free on the device
//! \param deviceId : ID of the device that owns the
//! loadDistArray
void DeepFreeLoadDistCuda(LoadDistMatrix* loadDistArray, uint32_t numMatrices,
                          int deviceId);

//! Copies sparse matrix on the GPU
//! All pointers given as parameters must be allocated with size of
//! numMatrices*sizeof(SparseMatrix) bytes
//! \param deviceDstArray : Destination array on
//! the GPU
//! \param deviceSrcArray : Source array on the GPU
//! \param hostDstArray :
//! Destination array on the Host
//! \param hostSrcArray : Source array on the Host
//! \param numMatrices : Number of sparse matrices
//! \param deviceId : ID of the device that owns both deviceDstArray and
//! deviceSrcArray
void DeepCopyGpuToGpu(SparseMatrix* deviceDstArray,
                      SparseMatrix* deviceSrcArray, uint32_t numMatrices,
                      int deviceId);

//! Copies load distribution matrix on the GPU
//! All pointers given as parameters must be allocated with size of
//! numMatrices*sizeof(SparseMatrix) bytes
//! \param deviceDstArray : destination device array
//! \param deviceSrcArray : source device array to copy from
//! \param numMatrices : number of sparse matrices to copy
//! \param deviceId : ID of the device that owns both deviceDstArray and
//! deviceSrcArray
void DeepCopyGpuToGpu(LoadDistMatrix* deviceDstArray,
                      LoadDistMatrix* deviceSrcArray, uint32_t numMatrices,
                      int deviceId);

//! Deep copies host matrix to GPU from Host
//! \param deviceDstArray : destination device array
//! \param hostSrcArray : source host array
//! \param numMatrices : Number of sparse matrices to copy
void DeepCopyHostToGpu(SparseMatrix* deviceDstArray, SparseMatrix* hostSrcArray,
                       uint32_t numMatrices, int deviceId);

//! Deep copies host matrix to GPU from Host
//! \param deviceDstArray : destination device array
//! \param hostSrcArray : source host array
//! \param numMatrices : number of sparse matrices
void DeepCopyHostToGpu(LoadDistMatrix* deviceDstArray,
                       LoadDistMatrix* hostSrcArray, uint32_t numMatrices,
                       int deviceId);

//! Deep copies host matrix to Host from Gpu
//! \param hostDstArray : destination host array
//! \param deviceSrcArray : source device array
//! \param hostSrcArray : source hot array
//! \param numMatrices : number of sparse matrices to copy
void DeepCopyGpuToHost(SparseMatrix* hostDstArray, SparseMatrix* deviceSrcArray,
                       uint32_t numMatrices, int deviceId);

//! Deep copies host matrix to Host from Gpu
//! \param hostDstArray : destination host array
//! \param deviceArray : source device array
//! \param numMatrices : Number of sparse matrices
void DeepCopyGpuToHost(LoadDistMatrix* hostDstArray,
                       LoadDistMatrix* deviceSrcArray, uint32_t numMatrices);

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

}  // namespace Motutapu::Compute::Sparse

#endif  // MOTUTAPU_MATRIXFORMAT_HPP
