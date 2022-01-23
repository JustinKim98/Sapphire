// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties

#ifndef SAPPHIRE_COMPUTE_CONVOLUTION_HPP
#define SAPPHIRE_COMPUTE_CONVOLUTION_HPP

#include <Sapphire/tensor/TensorData.hpp>

namespace Sapphire::Compute::Dense::Naive
{
using namespace TensorUtil;

void Im2Col(TensorData& inputMatrix, const TensorData& filter,
            const TensorData& input, int strideRow, int strideCol,
            int rowPadding, int colPadding, int dilationRow, int dilationCol,
            float pad = 0.0f);

void Col2Im(TensorData& input, const TensorData& inputMatrix,
            const TensorData& filter, int strideCol, int strideRow,
            int rowPadding, int colPadding, int dilationRow, int dilationCol);

void Conv2D(TensorData& y, const TensorData& x, const TensorData& filter,
            int strideRow, int strideCol, int rowPadding, int colPadding,
            int dilationRow, int dilationCol, DeviceInfo device);

void Conv2DBackward(TensorData& dx, TensorData& dFilter, const TensorData& dy,
                    const TensorData& x, const TensorData& filter,
                    int strideRow,
                    int strideCol, int rowPadding, int colPadding,
                    int dilationRow, int dilationCol, DeviceInfo device);
}

#endif
