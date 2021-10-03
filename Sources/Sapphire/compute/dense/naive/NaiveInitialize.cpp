// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/dense/naive/NaiveInitialize.hpp>
#include <random>

namespace Sapphire::Compute::Dense::Naive
{
void Normal(float* data, float mean, float sd, const Shape& shape,
            std::size_t paddedCols)
{
    const auto totalSize = shape.Size();
    const auto cols = shape.At(shape.Dim() - 1);
    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::normal_distribution<float> dist(mean, sd);

    for (long i = 0; i < static_cast<long>(totalSize / cols); ++i)
        for (int j = 0; j < cols; ++j)
            data[paddedCols * i + j] = dist(gen);
}

void Uniform(float* data, float min, float max, const Shape& shape,
             std::size_t paddedCols)
{
    const auto totalSize = shape.Size();
    const auto cols = shape.At(shape.Dim() - 1);
    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::uniform_real_distribution<float> dist(min, max);

    for (long i = 0; i < static_cast<long>(totalSize / cols); ++i)
        for (int j = 0; j < cols; ++j)
            data[paddedCols * i + j] = dist(gen);
}

void Scalar(float* data, float value, const Shape& shape,
            std::size_t paddedCols)
{
    const auto totalSize = shape.Size();
    const auto cols = shape.At(shape.Dim() - 1);

    for (long i = 0; i < static_cast<long>(totalSize / cols); ++i)
        for (int j = 0; j < cols; ++j)
            data[paddedCols * i + j] = value;
}
} // namespace Sapphire::Compute::Dense::Naive
