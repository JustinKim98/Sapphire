// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_COMPUTE_COMPUTE_BACKWARD_HPP
#define SAPPHIRE_COMPUTE_COMPUTE_BACKWARD_HPP
#include <Sapphire/tensor/TensorData.hpp>

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

void logBackward(TensorData& dx, const TensorData& dy, const TensorData& x);

void log10Backward(TensorData& dx, const TensorData& dy, const TensorData& x);

void InverseBackward(TensorData& dx, const TensorData& dy, const TensorData& x);

void MeanBackward(TensorData& dx, const TensorData& dy, const TensorData& x);

void MeanBackward(TensorData& dx, const TensorData& dy, const TensorData& x,
                   int dim);


} // namespace Sapphire::Compute

#endif
