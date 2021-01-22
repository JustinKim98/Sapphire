// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_CUDA_SPARSEMATRIXMANAGE_DECL_CUH
#define MOTUTAPU_CUDA_SPARSEMATRIXMANAGE_DECL_CUH

#include <cstdint>
#include <Motutapu/compute/cuda/CudaParams.hpp>
#include <Motutapu/util/SparseMatrixDecl.hpp>


namespace Motutapu::Cuda::Sparse
{
template <typename T>
__device__ void DeepAllocateSparseMatrix(SparseMatrix<T>* dest);

template <typename T>
__device__ void DeepFreeSparseMatrix(SparseMatrix<T>* target);

template <typename T>
__device__ void ShallowFreeSparseMatrix(SparseMatrix<T>* target);

template <typename T>
__device__ void DeepCopySparseMatrix(SparseMatrix<T>* dest,
                                     SparseMatrix<T>* src, uint32_t rowOffset);

//! Frees sparse matrix
//! \tparam T : Data type of the sparse matrix
//! \param targetArray : target array to free
//! \param size : Number of sparse matrices
template <typename T>
__global__ void DeepFreeKernel(SparseMatrix<T>* targetArray, uint32_t size);

//! Allocates sparse matrix on the GPU
//! \tparam T : Data type of the sparse matrix (Must include numCols and NNZ)
//! \param targetArray : target array to deep allocate
//! \param size : Number of sparse matrices
template <typename T>
__global__ void DeepAllocateKernel(SparseMatrix<T>* targetArray, uint32_t size);

//! Copies sparse matrix on the GPU
//! \tparam T : Data type of the sparse matrix
//! \param destArray : Destination array on the Gpu
//! \param srcArray : Source array on the Gpu
//! \param size : Number of sparse matrices
template <typename T>
__global__ void DeepCopySparseMatrixOnGpu(SparseMatrix<T>* destArray,
                                          SparseMatrix<T>* srcArray,
                                          uint32_t size);
//! Deep copies host matrix to GPU from Host
//! \tparam T : Data type of the sparse matrix
//! \param deviceArray : Target device array to copy
//! \param hostArray : Source device array to copy from
//! \param size : Number of sparse matrices
template <typename T>
__host__ void DeepCopyHostToGpu(SparseMatrix<T>* deviceArray,
                                SparseMatrix<T>* hostArray, uint32_t size);

//! Deep copies host matrix to Host from Gpu
//! \tparam T : Data type of the sparse matrix
//! \param deviceArray : Target device array to copy
//! \param hostArray : Source device array to copy from
//! \param size : Number of sparse matrices
template <typename T>
__host__ void DeepCopyGpuToHost(SparseMatrix<T>* deviceArray,
                                SparseMatrix<T>* hostArray, uint32_t size);

//! Shallow Allocates Sparse matrix on the GPU 
//! \tparam T : Data type of the sparse matrix
//! \param targetArray : The target array to allocate sparse matrix (Device)
//! \param size : Number of sparse matrices
template <typename T>
__host__ void ShallowAllocateGpu(SparseMatrix<T>* targetArray, uint32_t size);

//! Deeply allocates Sparse matrix on the GPU
//! After copying size data
//! Must be called after Shallow allocation
//! \tparam T : Data type of the sparse matrix
//! \param targetArray : The target array to allocate sparse matrix (Device)
//! \param hostRefArray : The source to copy size from
//! \param size : The size of the target and host array
template <typename T>
__host__ void DeepAllocateGpu(SparseMatrix<T>* targetArray,
                              SparseMatrix<T>* hostRefArray, uint32_t size);
}
#endif
