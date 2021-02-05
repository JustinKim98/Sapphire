// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_NAIVEINITIALIZE_HPP
#define MOTUTAPU_NAIVEINITIALIZE_HPP
#include <Motutapu/tensor/Shape.hpp>
#include <cstdlib>

namespace Motutapu::Compute::Naive
{
void Normal(float* data, float mean, float sd, const Shape& shape,
            size_t paddedCols, size_t batchSize);

void Uniform(float* data, float min, float max, const Shape& shape,
             size_t paddedCols, size_t batchSize);

void Scalar(float* data, float value, const Shape& shape, size_t paddedCols,
            size_t batchSize);
}  // namespace Motutapu::Compute::Naive

#endif  // MOTUTAPU_NAIVEINITIALIZE_HPP
