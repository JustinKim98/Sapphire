// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_COMPUTE_DENSE_NAIVE_POOL_HPP
#define SAPPHIRE_COMPUTE_DENSE_NAIVE_POOL_HPP

#include <Sapphire/tensor/TensorData.hpp>
#include <utility>

namespace Sapphire::Compute::Dense::Naive
{
//! Performs Max pooling
void MaxPool2D(TensorUtil::TensorData& y, const TensorUtil::TensorData& x,
               std::pair<int, int> filterSize, std::pair<int, int> stride,
               std::pair<int, int> padding, std::pair<int, int> dilation);

//! Performs back propagation of max pooling
void MaxPool2DBackward(TensorUtil::TensorData& dx,
                       const TensorUtil::TensorData& x,
                       const TensorUtil::TensorData& dy,
                       std::pair<int, int> filterSize,
                       std::pair<int, int> stride, std::pair<int, int> padding,
                       std::pair<int, int> dilation);
}

#endif
