// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_COMPUTE_SPARSE_NAIVE_SPARSEGEMM_HPP
#define SAPPHIRE_COMPUTE_SPARSE_NAIVE SPARSEGEMM_HPP

#include <Sapphire/compute/sparse/SparseMatrix.hpp>
#include <cstdlib>

namespace Sapphire::Compute::Sparse::Naive
{
void Gemm(SparseMatrix** output, SparseMatrix* a, SparseMatrix* b, uint32_t m,
          uint32_t n, size_t numMatrices);
}

#endif  // SAPPHIRE_SPARSEGEMM_HPP
