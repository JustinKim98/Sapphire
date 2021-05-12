// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/dense/naive/NaiveInitialize.hpp>
#include <random>

namespace Motutapu::Compute::Naive
{
void Normal(float* data, float mean, float sd, const Shape& shape,
            size_t paddedCols, size_t batchSize)
{
    const auto totalSize = shape.Size() * batchSize;
    const auto cols = shape.At(shape.Dim() - 1);
    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::normal_distribution<float> dist(mean, sd);

#pragma omp parallel for collapse(2) schedule(static)
    for (unsigned int i = 0; i < totalSize / cols; ++i)
        for (size_t j = 0; j < cols; ++j)
            data[paddedCols * i + j] = dist(gen);
}

void Uniform(float* data, float min, float max, const Shape& shape,
             size_t paddedCols, size_t batchSize)
{
    const auto totalSize = shape.Size() * batchSize;
    const auto cols = shape.At(shape.Dim() - 1);
    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::uniform_real_distribution<float> dist(min, max);

#pragma omp parallel for collapse(2) schedule(static)
    for (unsigned int i = 0; i < totalSize / cols; ++i)
        for (size_t j = 0; j < cols; ++j)
            data[paddedCols * i + j] = dist(gen);
}

void Scalar(float* data, float value, const Shape& shape, size_t paddedCols,
            size_t batchSize)
{
    const auto totalSize = shape.Size() * batchSize;
    const auto cols = shape.At(shape.Dim() - 1);

#pragma omp parallel for collapse(2) schedule(static)
    for (unsigned int i = 0; i < totalSize / cols; ++i)
        for (size_t j = 0; j < cols; ++j)
            data[paddedCols * i + j] = value;
}
}  // namespace Motutapu::Compute::Naive