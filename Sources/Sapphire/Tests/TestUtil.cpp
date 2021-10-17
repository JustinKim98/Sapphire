// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Tests/TestUtil.hpp>
#include <Sapphire/util/Shape.hpp>
#include <random>
#include <doctest.h>
#include <iostream>
#include <algorithm>

namespace Sapphire::Test
{
void InitIntegerDenseMatrix(float* matrixPtr, const size_t m, const size_t n,
                            const size_t paddedN, const size_t numMatrices,
                            const float sparsity)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    std::uniform_int_distribution<int> uniform(-30, 30);
    std::normal_distribution<float> normal(0, 10);

    for (long matrixIdx = 0; matrixIdx < static_cast<long>(numMatrices); ++
         matrixIdx)
        for (long rowIdx = 0; rowIdx < static_cast<long>(m); ++rowIdx)
            for (long colIdx = 0; colIdx < static_cast<long>(n); ++colIdx)
            {
                if (prob(gen) > static_cast<double>(sparsity))
                    matrixPtr[matrixIdx * m * paddedN + rowIdx * paddedN +
                              colIdx] = static_cast<float>(uniform(gen));
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

    for (long matrixIdx = 0; matrixIdx < static_cast<long>(numMatrices); ++
         matrixIdx)
        for (long rowIdx = 0; rowIdx < static_cast<long>(m); ++rowIdx)
            for (long colIdx = 0; colIdx < static_cast<long>(n); ++colIdx)
            {
                if (uniform(gen) > static_cast<double>(sparsity))
                    matrixPtr[matrixIdx * m * paddedN + rowIdx * paddedN +
                              colIdx] = normal(gen);
                else
                    matrixPtr[matrixIdx * m * paddedN + rowIdx * paddedN +
                              colIdx] = 0.0f;
            }
}

Shape CreateRandomShape(int dim, int maxDim)
{
    if (dim <= 0)
        throw std::invalid_argument("Dimension must be greater than zero");

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, maxDim);
    std::vector<int> shapeVector(dim);
    for (int i = 0; i < dim; ++i)
    {
        shapeVector.at(i) = distrib(gen) % maxDim + 1;
    }
    return Shape(shapeVector);
}

Shape ShuffleShape(int dim, Shape shape)
{
    if (dim <= 0)
        throw std::invalid_argument("Dimension must be greater than zero");

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(1, shape.Dim());
    const std::size_t newDim = dist(gen);

    std::vector<int> shapeVector = shape.GetShapeVector();
    std::shuffle(shapeVector.begin(), shapeVector.end(), rd);
    while (shapeVector.size() != newDim)
    {
        std::shuffle(shapeVector.begin(), shapeVector.end(), rd);
        shapeVector.front() = shapeVector.back() * shapeVector.front();
        shapeVector.pop_back();
    }
    return Shape(shapeVector);
}


void CheckNoneZero(const float* ptr, unsigned size,
                   bool print)
{
    for (unsigned int i = 0; i < size; ++i)
    {
        if (print)
            std::cout << "ptrA: " << ptr[i] << std::endl;
        auto pass = ptr[i] > 0 || ptr[i] < 0;
        CHECK(pass);
    }
}
} // namespace Sapphire::Test
