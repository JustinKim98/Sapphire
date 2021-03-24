// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_COMPUTE_SPARSEMATRIX_HPP
#define MOTUTAPU_COMPUTE_SPARSEMATRIX_HPP

#include <Motutapu/compute/Sparse.hpp>

namespace Motutapu::Compute
{
__host__ void DeepAllocateSparseMatrix(SparseMatrix* cudaTarget, size_t m,
                                       size_t nnz);

__host__ void ShallowAllocateSparseMatrix(SparseMatrix* target);

__host__ void DeepFreeSparseMatrix(SparseMatrix* target);

__host__ void ShallowFreeSparseMatrix(SparseMatrix* target);

__host__ void DeepCopySparseMatrix(SparseMatrix* dest, SparseMatrix* src,
                                   uint32_t rowOffset);

__host__ void ConvertDenseToSparse(SparseMatrix* dst, float* src,
                                   uint32_t numRows, uint32_t numCols,
                                   uint32_t numMatrices);

__host__ void ConvertSparseToDense(float* dst, SparseMatrix* src,
                                   uint32_t numMatrices);

__global__ void ConvertDenseToSparseKernel(SparseMatrix* dst, float* src,
                                           uint32_t numRows, uint32_t numCols,
                                           uint32_t numMatrices);

__global__ void ConvertSparseToDenseKernel(float* dst, SparseMatrix* src,
                                           uint32_t numMatrices);

//! Frees sparse matrix
//! \param targetArray : target array to free
//! \param size : Number of sparse matrices
__global__ void DeepFreeKernel(SparseMatrix* targetArray, uint32_t size);

//! Allocates sparse matrix on the GPU
//! \param targetArray : target array to deep allocate
//! \param size : Number of sparse matrices
__global__ void DeepAllocateKernel(SparseMatrix* targetArray, uint32_t size);

//! Copies sparse matrix on the GPU
//! \param destArray : Destination array on the Gpu
//! \param srcArray : Source array on the Gpu
//! \param size : Number of sparse matrices
__global__ void DeepCopySparseMatrixOnGpu(SparseMatrix* destArray,
                                          SparseMatrix* srcArray,
                                          uint32_t size);
//! Deep copies host matrix to GPU from Host
//! \param deviceArray : Target device array to copy
//! \param hostArray : Source device array to copy from
//! \param size : Number of sparse matrices
__host__ void DeepCopyHostToGpu(SparseMatrix* deviceArray,
                                SparseMatrix* hostArray, uint32_t size);

//! Deep copies host matrix to Host from Gpu
//! \param deviceArray : Target device array to copy
//! \param hostArray : Source device array to copy from
//! \param size : Number of sparse matrices
__host__ void DeepCopyGpuToHost(SparseMatrix* deviceArray,
                                SparseMatrix* hostArray, uint32_t size);

//! Shallow Allocates Sparse matrix on the GPU
//! \param targetArray : The target array to allocate sparse matrix (Device)
//! \param size : Number of sparse matrices
__host__ void ShallowAllocateGpu(SparseMatrix* targetArray, uint32_t size);

//! Deeply allocates Sparse matrix on the GPU
//! After copying size data
//! Must be called after Shallow allocation
//! \param targetArray : The target array to allocate sparse matrix (Device)
//! \param hostRefArray : The source to copy size from
//! \param size : The size of the target and host array
__host__ void DeepAllocateGpu(SparseMatrix* targetArray,
                              SparseMatrix* hostRefArray, uint32_t size);

}  // namespace Motutapu::Compute

#endif  // MOTUTAPU_SPARSEMATRIX_HPP
