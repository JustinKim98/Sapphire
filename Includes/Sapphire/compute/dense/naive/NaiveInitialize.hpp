// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Sapphire_NAIVEINITIALIZE_HPP
#define Sapphire_NAIVEINITIALIZE_HPP
#include "Sapphire/tensor/Shape.hpp"
#include "cstdlib"

namespace Sapphire::Compute::Dense::Naive
{
void Normal(float* data, float mean, float sd, const Shape& shape,
            size_t paddedCols, size_t batchSize);

void Uniform(float* data, float min, float max, const Shape& shape,
             size_t paddedCols, size_t batchSize);

void Scalar(float* data, float value, const Shape& shape, size_t paddedCols,
            size_t batchSize);
}  // namespace Sapphire::Compute::Naive

#endif  // Sapphire_NAIVEINITIALIZE_HPP
