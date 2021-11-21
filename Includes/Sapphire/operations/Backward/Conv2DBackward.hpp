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
    Conv2DBackProp(std::string name, TensorData dx, TensorData dy,
                   TensorData filter, TensorData bias,
                   TensorData x,
                   std::pair<int, int> stride, std::pair<int, int> dilation,
                   std::pair<int, int> padding);

    Conv2DBackProp(std::string name, TensorData dx, TensorData dy,
                   TensorData filter, TensorData x,
                   std::pair<int, int> stride, std::pair<int, int> dilation,
                   std::pair<int, int> padding);

    Conv2DBackProp(const Conv2DBackProp& conv2DBackProp) = default;
    Conv2DBackProp(Conv2DBackProp&& conv2DBackProp) noexcept = default;
    Conv2DBackProp& operator=(const Conv2DBackProp& conv2DBackProp) = delete;
    Conv2DBackProp& operator=(Conv2DBackProp&& conv2DBackProp) noexcept = delete
    ;

    ~Conv2DBackProp() override = default;

private:
    void m_runBackProp() override;

    std::pair<int, int> m_stride, m_dilation, m_padding;
    unsigned int m_batchSize;
    bool m_hasBias;
};
}

#endif
