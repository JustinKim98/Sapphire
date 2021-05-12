// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Sapphire_SPARSE_GEMM_TEST_HPP
#define Sapphire_SPARSE_GEMM_TEST_HPP

namespace Sapphire::Test
{
void LoadDistTestFixed(bool printVerbose);

void LoadDistTest(bool printVerbose);

void SparseGemmTest(bool printVerbose);
}  // namespace Sapphire::Test

#endif  // Sapphire_SPARSEGEMMTEST_HPP
