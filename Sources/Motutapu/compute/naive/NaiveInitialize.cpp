// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/naive/NaiveInitialize.hpp>
#include <random>

namespace Motutapu::Compute::Naive
{
void Normal(float* data, float mean, float sd, unsigned int size)
{
    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::normal_distribution<> dist(mean, sd);
    for (unsigned int i = 0; i < size; ++i)
    {
        data[i] = dist(gen);
    }
}

void Uniform(float* data, float min, float max, unsigned int size)
{
    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::uniform_real_distribution<> dist(min, max);
    for (unsigned int i = 0; i < size; ++i)
    {
        data[i] = dist(gen);
    }
}

void Scalar(float* data, float value, unsigned int size)
{
    for (unsigned int i = 0; i < size; ++i)
    {
        data[i] = value;
    }
}
}  // namespace Motutapu::Compute::Naive