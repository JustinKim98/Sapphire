// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Tests/TestUtil.hpp>
#include <random>

namespace Sapphire::Test
{
void InitFixedDenseMatrix(float* matrixPtr, const size_t m, const size_t n,
                          const size_t paddedN, const size_t numMatrices,
                          const float sparsity)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::normal_distribution<float> normal(0, 10);

#pragma omp parallel for default(none) collapse(3) shared( \
    numMatrices, m, n, paddedN, uniform, gen, sparsity, matrixPtr, normal)
    for (size_t matrixIdx = 0; matrixIdx < numMatrices; ++matrixIdx)
        for (size_t rowIdx = 0; rowIdx < m; ++rowIdx)
            for (size_t colIdx = 0; colIdx < n; ++colIdx)
            {
                if (uniform(gen) > sparsity)
                    matrixPtr[matrixIdx * m * paddedN + rowIdx * paddedN +
                              colIdx] = 1.0f;
                else
                    matrixPtr[matrixIdx * m * paddedN + rowIdx * paddedN +
                              colIdx] = 0.0f;
            }
}

void InitRandomDenseMatrix(float* matrixPtr, const size_t m, const size_t n,
                           const size_t paddedN, const size_t numMatrices,
                           const float sparsity)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::normal_distribution<float> normal(0, 10);

#pragma omp parallel for default(none) collapse(3) shared( \
    numMatrices, m, n, paddedN, uniform, gen, sparsity, matrixPtr, normal)
    for (size_t matrixIdx = 0; matrixIdx < numMatrices; ++matrixIdx)
        for (size_t rowIdx = 0; rowIdx < m; ++rowIdx)
            for (size_t colIdx = 0; colIdx < n; ++colIdx)
            {
                if (uniform(gen) > sparsity)
                    matrixPtr[matrixIdx * m * paddedN + rowIdx * paddedN +
                              colIdx] = normal(gen);
                else
                    matrixPtr[matrixIdx * m * paddedN + rowIdx * paddedN +
                              colIdx] = 0.0f;
            }
}
}  // namespace Sapphire::Test