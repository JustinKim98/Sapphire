// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/dense/naive/NaiveInitialize.hpp>
#include <random>

namespace Sapphire::Compute::Dense::Naive
{
void Normal(float* data, float mean, float sd, const Shape& shape)
{
    const auto totalSize = shape.Size();
    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::normal_distribution dist(mean, sd);

    for (int i = 0; i < totalSize; ++i)
        data[i] = dist(gen);
}

void Uniform(float* data, float min, float max, const Shape& shape)
{
    const auto totalSize = shape.Size();
    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::uniform_real_distribution dist(min, max);

    for (int i = 0; i < totalSize; ++i)
        data[i] = dist(gen);
}

void Scalar(float* data, float value, const Shape& shape)
{
    const auto totalSize = shape.Size();

    for (long i = 0; i < totalSize; ++i)
        data[i] = value;
}
} // namespace Sapphire::Compute::Dense::Naive
