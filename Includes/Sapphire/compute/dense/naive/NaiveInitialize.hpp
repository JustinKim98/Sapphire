// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_COMPUTE_DENSE_NAIVE_INITIALIZE_HPP
#define SAPPHIRE_COMPUTE_DENSE_NAIVE_INITIALIZE_HPP
#include <Sapphire/util/Shape.hpp>


namespace Sapphire::Compute::Dense::Naive
{
void Normal(float* data, float mean, float sd, const Shape& shape);

void Uniform(float* data, float min, float max, const Shape& shape);

void Scalar(float* data, float value, const Shape& shape);
} // namespace Sapphire::Compute::Naive

#endif  // Sapphire_NAIVEINITIALIZE_HPP
