// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_TEST_TESTUTIL_HPP
#define SAPPHIRE_TEST_TESTUTIL_HPP
#include <cstdlib>
namespace Sapphire::Test
{
void InitRandomDenseMatrix(float* matrixPtr, const size_t m, const size_t n,
                               const size_t paddedN, const size_t numMatrices,
                               const float sparsity);
}

#endif  // SAPPHIRE_TESTUTIL_HPP
