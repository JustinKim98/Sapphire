// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Sapphire_SPARSE_GEMM_TEST_HPP
#define Sapphire_SPARSE_GEMM_TEST_HPP

#include <cstdlib>

namespace Sapphire::Test
{
void LoadDistTestFixed(bool printVerbose);

void LoadDistTest(bool printVerbose);

long SparseGemmTestComplex(bool printVerbose, size_t minimumNumMatrices);

long SparseGemmTestSimple(bool printVerbose);

void NestedPerformanceTest(size_t m, size_t n, size_t k, size_t numMatrices,
                           float sparsity);

}  // namespace Sapphire::Test

#endif  // Sapphire_SPARSEGEMMTEST_HPP
