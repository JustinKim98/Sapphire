// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_BACKPROP_SOFTMAX_BACKWARD_HPP
#define SAPPHIRE_BACKPROP_SOFTMAX_BACKWARD_HPP

#include <Sapphire/operations/Backward/BackPropWrapper.hpp>

namespace Sapphire::BackProp
{
class SoftMaxBackward : public BackPropWrapper
{
public:
    SoftMaxBackward(std::string name, TensorUtil::TensorData dx,
                    TensorUtil::TensorData dy,
                    TensorUtil::TensorData y);

private:
    void m_runBackProp() override;
};
}
#endif
