// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_SPARSE_GEMM_TEST_HPP
#define MOTUTAPU_SPARSE_GEMM_TEST_HPP

namespace Motutapu::Test
{
void LoadDistTestFixed(bool printVerbose);

void LoadDistTest(bool printVerbose);

void SparseGemmTest(bool printVerbose);
}  // namespace Motutapu::Test

#endif  // MOTUTAPU_SPARSEGEMMTEST_HPP
