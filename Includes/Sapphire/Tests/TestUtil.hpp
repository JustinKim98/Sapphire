// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_TEST_TEST_UTIL_HPP
#define SAPPHIRE_TEST_TEST_UTIL_HPP

#include <Sapphire/tensor/Shape.hpp>
#include <limits>

namespace Sapphire::Test
{
void InitIntegerDenseMatrix(float* matrixPtr, const size_t m, const size_t n,
                            const size_t paddedN, const size_t numMatrices,
                            const float sparsity);

void InitRandomDenseMatrix(float* matrixPtr, const size_t m, const size_t n,
                           const size_t paddedN, const size_t numMatrices,
                           const float sparsity);

Shape CreateRandomShape(int dim, int maxDim = 10);

void CheckNoneZeroEquality(const float* ptrA, const float* ptrB, unsigned size,
                           bool print,
                           float equalThreshold = std::numeric_limits<
                               float>::epsilon());
} // namespace Sapphire::Test

#endif  // SAPPHIRE_TESTUTIL_HPP
