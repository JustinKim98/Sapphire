// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Sapphire_COMPUTE_MATRIXFORMAT_HPP
#define Sapphire_COMPUTE_MATRIXFORMAT_HPP

#include <Sapphire/compute/sparse/SparseMatrix.hpp>
#include <cstdlib>

namespace Sapphire::Compute
{
//! Deep allocates sparse matrix on host
//! \param sparseMatrixArray : ptr to allocate sparse matrix
//! \param m : number of rows
//! \param n : number of columns
//! \param nnz : array of number of non-zeros
//! \param numMatrices : number of matrices
void DeepAllocateSparseHost(SparseMatrix** sparseMatrixArray, uint32_t m,
                            uint32_t n, const uint32_t nnz[],
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

//! Allocates sparse matrix array on the Device
//! \param deviceSparseArray : ptr to allocate the sparse matrix array
//! \param hostSparseMatrixArray : ptr to the previously allocated host
//! \param numMatrices : number of matrices
//! \param deviceId : ID of the device to allocate
void DeepAllocateSparseCuda(SparseMatrix** deviceSparseArray,
                            SparseMatrix* hostSparseMatrixArray,
                            uint32_t numMatrices, int deviceId);

//! Allocates load distribution matrix array on the Device
//! \param deviceLoadDistArray : ptr to the allocation target load distribution
//! matrix array
//! \param hostLoadDistArray : ptr to the previously allocated host
//! load distribution matrix array
//! \param numMatrices : number of matrices
//! \param deviceId : ID of the device to allocate
void DeepAllocateLoadDistCuda(LoadDistMatrix** deviceLoadDistArray,
                              LoadDistMatrix* hostLoadDistArray,
                              uint32_t numMatrices, int deviceId);

//! Frees sparse matrix allocated on the Device
//! \param deviceSparseArray : Array of deviceSparseArray sparse matrix to free
//! on the device
//! \param deviceId : ID of the device that owns the
//! deviceSparseArray
void DeepFreeSparseCuda(SparseMatrix* deviceSparseArray, uint32_t numMatrices,
                        int deviceId);

//! Frees load distribution matrix allocated on the Device
//! \param deviceLoadDistArray : Array of deviceLoadDistArray load distribution
//! matrix to free on the device \param deviceId : ID of the device that owns
//! the deviceLoadDistArray
void DeepFreeLoadDistCuda(LoadDistMatrix* deviceLoadDistArray,
                          uint32_t numMatrices, int deviceId);

//! Deep copies sparse matrix to Device from Host
//! All pointers given as parameters must be allocated with size of
//! numMatrices*sizeof(SparseMatrix) bytes
//! \param deviceDstArray : Destination array on
//! the Device
//! \param deviceSrcArray : Source array on the Device
//! \param hostDstArray :
//! Destination array on the Host
//! \param hostSrcArray : Source array on the Host
//! \param numMatrices : Number of sparse matrices
//! \param deviceId : ID of the device that owns both deviceDstArray and
//! deviceSrcArray
void DeepCopyDeviceToDevice(SparseMatrix* deviceDstArray,
                            SparseMatrix* deviceSrcArray, uint32_t numMatrices,
                            int deviceId);

//! Deep copies load distribution matrix to Device from Host
//! All pointers given as parameters must be allocated with size of
//! numMatrices*sizeof(SparseMatrix) bytes
//! \param deviceDstArray : destination device array
//! \param deviceSrcArray : source device array to copy from
//! \param numMatrices : number of sparse matrices to copy
//! \param deviceId : ID of the device that owns both deviceDstArray and
//! deviceSrcArray
void DeepCopyDeviceToDevice(LoadDistMatrix* deviceDstArray,
                            LoadDistMatrix* deviceSrcArray,
                            uint32_t numMatrices, int deviceId);

//! Deep copies sparse matrix to Device from Host
//! \param deviceDstArray : destination device array
//! \param hostSrcArray : source host array
//! \param numMatrices : Number of sparse matrices to copy
void DeepCopyHostToDevice(SparseMatrix* deviceDstArray,
                          SparseMatrix* hostSrcArray, uint32_t numMatrices,
                          int deviceId);

//! Deep copies load distribution matrix to Device from Host
//! \param deviceDstArray : destination device array
//! \param hostSrcArray : source host array
//! \param numMatrices : number of sparse matrices
void DeepCopyHostToDevice(LoadDistMatrix* deviceDstArray,
                          LoadDistMatrix* hostSrcArray, uint32_t numMatrices,
                          int deviceId);

//! Deep copies sparse matrix to Host from Device
//! \param hostDstArray : destination host array
//! \param deviceSrcArray : source device array
//! \param hostSrcArray : source hot array
//! \param numMatrices : number of sparse matrices to copy
void DeepCopyDeviceToHost(SparseMatrix* hostDstArray,
                          SparseMatrix* deviceSrcArray, uint32_t numMatrices,
                          int deviceId);

//! Deep copies load distribution matrix to Host from Device
//! \param hostDstArray : destination host array
//! \param deviceArray : source device array
//! \param numMatrices : Number of sparse matrices
void DeepCopyDeviceToHost(LoadDistMatrix* hostDstArray,
                          LoadDistMatrix* deviceSrcArray, uint32_t numMatrices,
                          int deviceId);

//! Deep copies sparse matrix to Host from Host
//! \param hostDstArray : destination host array
//! \param hostSrcArray : source host array
//! \param numMatrices : number of sparse matrices to copy
void DeepCopyHostToHost(SparseMatrix* hostDstArray, SparseMatrix* hostSrcArray,
                        uint32_t numMatrices);

//! Deep copies host matrix to Host from Host
//! \param hostDstArray : destination host array
//! \param deviceArray : source host array
//! \param numMatrices : Number of sparse matrices
void DeepCopyHostToHost(LoadDistMatrix* hostDstArray,
                        LoadDistMatrix* hostSrcArray, uint32_t numMatrices);

void CreateSparseMatrixWithDenseMatrix(SparseMatrix** dst, const float* src,
                                       uint32_t m, uint32_t n, uint32_t paddedN,
                                       uint32_t numMatrices);

void ConvertSparseMatrixToDenseMatrix(float* dst, const SparseMatrix* src,
                                      uint32_t m, uint32_t n, uint32_t paddedN,
                                      uint32_t numMatrices);

}  // namespace Sapphire::Compute

#endif  // Sapphire_MATRIXFORMAT_HPP
