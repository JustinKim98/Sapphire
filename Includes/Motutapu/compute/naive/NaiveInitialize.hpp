// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_NAIVEINITIALIZE_HPP
#define MOTUTAPU_NAIVEINITIALIZE_HPP

namespace Motutapu::Compute::Naive
{
void Normal(float* data, float mean, float sd, unsigned int size);

void Uniform(float* data, float min, float max, unsigned int size);

void Scalar(float* data, float value, unsigned int size);
}  // namespace Motutapu::Compute

#endif  // MOTUTAPU_NAIVEINITIALIZE_HPP
