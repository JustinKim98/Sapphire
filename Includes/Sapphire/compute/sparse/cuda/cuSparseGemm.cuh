// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_COMPUTE_SPARSE_CUSPARSEGEMM_CUH
#define SAPPHIRE_COMPUTE_SPARSE_CUSPARSEGEMM_CUH

#include <Sapphire/compute/cudaUtil/CudaParams.cuh>
#include <Sapphire/compute/sparse/Sparse.hpp>

namespace Sapphire::Compute::Sparse::Cuda
{
size_t cuSparseGemm(SparseMatrix** hostOutput, SparseMatrix** cudaOutput,
                  SparseMatrix* cudaA, SparseMatrix* cudaB, uint32_t m,
                  uint32_t n, size_t numMatrices, int deviceId,
                  bool copyResultToHost);
}
#endif
