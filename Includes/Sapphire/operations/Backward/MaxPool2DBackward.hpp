// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_BACKPROP_MAX_POOL_2D_BACKWARD_HPP
#define SAPPHIRE_BACKPROP_MAX_POOL_2D_BACKWARD_HPP

#include <Sapphire/operations/Backward/BackPropWrapper.hpp>

namespace Sapphire::BackProp
{
using namespace TensorUtil;

class MaxPool2DBackProp : public BackPropWrapper
{
public:
    MaxPool2DBackProp(TensorData dx, TensorData dy, TensorData x,
                      TensorData y,
                      std::pair<int, int> windowSize,
                      std::pair<int, int> stride,
                      std::pair<int, int> padSize);

    ~MaxPool2DBackProp() override = default;

private:
    void m_runBackProp() override;

    std::pair<int, int> m_windowSize, m_stride, m_padSize;
};
}

#endif
