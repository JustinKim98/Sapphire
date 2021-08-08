// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_COMPUTE_CONVOLUTION_OPS_HPP
#define SAPPHIRE_COMPUTE_CONVOLUTION_OPS_HPP
#include <Sapphire/tensor/TensorData.hpp>

namespace Sapphire::Compute
{
using namespace TensorUtil;

//! x, y, filter must have shape of C,H,W with Same batch size N (Data aligned in NCHW format)
void Conv2DForward(TensorData& y, const TensorData& x, const TensorData& filter,
                   int strideRow, int strideCol, int dilationRow,
                   int dilationCol, int rowPadding, int columnPadding);

void MaxPool2DForward(TensorData& y, const TensorData& x, int windowHeight,
                      int windowWidth, int strideRow, int strideCol,
                      int rowPadding, int columnPadding);

void AvgPool2DForward(TensorData& y, const TensorData& x, int windowHeight,
                      int windowWidth, int strideRow, int strideCol,
                      int rowPadding, int columnPadding);

void Conv2DBackward(TensorData& dx, TensorData& dFilter, const TensorData& dy,
                    const TensorData& x, const TensorData& filter,
                    int strideRow, int strideCol, int dilationRow,
                    int dilationCol, int rowPadding, int columnPadding);

void MaxPool2DBackward(TensorData& dx, const TensorData& dy, const TensorData& x,
                       const TensorData& y, int windowHeight, int windowWidth,
                       int strideRow, int strideCol, int rowPadding,
                       int columnPadding);

void AvgPool2DBackward(TensorData& dx, const TensorData& dy, const TensorData& x,
                       const TensorData& y, int windowHeight, int windowWidth,
                       int strideRow, int strideCol, int rowPadding,
                       int columnPadding);
}

#endif