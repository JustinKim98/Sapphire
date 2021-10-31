// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_BACKPROP_CONV2DBACKWARD_HPP
#define SAPPHIRE_BACKPROP_CONV2DBACKWARD_HPP

#include <Sapphire/operations/Backward/BackPropWrapper.hpp>

namespace Sapphire::BackProp
{
using namespace TensorUtil;

class Conv2DBackProp : public BackPropWrapper
{
public:
    Conv2DBackProp(TensorUtil::TensorData dx, TensorUtil::TensorData dy,
                   TensorUtil::TensorData filter, TensorUtil::TensorData bias,
                   TensorUtil::TensorData x,
                   std::pair<int, int> stride, std::pair<int, int> dilation,
                   std::pair<int, int> padding,
                   Optimizer::Optimizer* optimizer);

    Conv2DBackProp(TensorUtil::TensorData dx, TensorUtil::TensorData dy,
                   TensorUtil::TensorData filter, TensorUtil::TensorData x,
                   std::pair<int, int> stride, std::pair<int, int> dilation,
                   std::pair<int, int> padding,
                   Optimizer::Optimizer* optimizer);

    ~Conv2DBackProp() override = default;

private:
    void m_runBackProp() override;

    std::pair<int, int> m_stride, m_dilation, m_padding;
    unsigned int m_batchSize;
    bool m_hasBias;
};
}

#endif
