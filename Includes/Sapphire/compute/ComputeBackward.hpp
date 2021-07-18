// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_COMPUTE_COMPUTE_BACKWARD_HPP
#define SAPPHIRE_COMPUTE_COMPUTE_BACKWARD_HPP
#include <Sapphire/tensor/TensorData.hpp>
#include <algorithm>

namespace Sapphire::Compute
{
using namespace TensorUtil;

void Conv2DBackward(TensorData& dx, TensorData& dFilter, const TensorData& dy,
                    const TensorData& x, const TensorData& filter,
                    int strideRow, int strideCol, int dilationRow,
                    int dilationCol, int rowPadding, int columnPadding);

void MaxPool2DBackward(TensorData& dy, TensorData& dx, const TensorData& x,
                       const TensorData& y, int windowHeight, int windowWidth,
                       int strideRow, int strideCol, int rowPadding,
                       int columnPadding);

void AvgPool2DBackward(TensorData& dy, TensorData& dx, const TensorData& x,
                       const TensorData& y, int windowHeight, int windowWidth,
                       int strideRow, int strideCol, int rowPadding,
                       int columnPadding);

void DotBackward(TensorData& da, TensorData& db, const TensorData& dy,
                 const TensorData& a, const TensorData& b);

void PowBackward(TensorData& dx, const TensorData& dy, const TensorData& x,
                 float factor);

void cosBackward(TensorData& dx, const TensorData& dy, const TensorData& x);

void sinBackward(TensorData& dx, const TensorData& dy, const TensorData& x);

void tanBackward(TensorData& dx, const TensorData& dy, const TensorData& x);

void coshBackward(TensorData& dx, const TensorData& dy, const TensorData& x);

void sinhBackward(TensorData& dx, const TensorData& dy, const TensorData& x);

void tanhBackward(TensorData& dx, const TensorData& dy, const TensorData& x);

void logBackward(TensorData& dx, const TensorData& dy, const TensorData& x);

void log10Backward(TensorData& dx, const TensorData& dy, const TensorData& x);

void ReLUBackward(TensorData& dx, const TensorData& dy);

void LeakyReluBackward(TensorData& dx, const TensorData& dy, float a);

void InverseBackward(TensorData& dx, const TensorData& dy, const TensorData& x);

void MeanBackward(TensorData& dx, const TensorData& dy, const TensorData& x);

void MeanBackward(TensorData& dx, const TensorData& dy, const TensorData& x,
                   int dim);

void SoftMaxBackward(TensorData& dx, const TensorData& dy, const TensorData& x);

} // namespace Sapphire::Compute

#endif
