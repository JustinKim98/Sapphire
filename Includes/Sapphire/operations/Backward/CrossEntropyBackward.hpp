// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_BACKPROP_CROSS_ENTROPY_BACKWARD_HPP
#define SAPPHIRE_BACKPROP_CROSS_ENTROPY_BACKWARD_HPP

#include <Sapphire/operations/Backward/BackPropWrapper.hpp>

namespace Sapphire::BackProp
{
class CrossEntropyBackward : public BackPropWrapper
{
public:
    CrossEntropyBackward(std::string name, TensorUtil::TensorData dx,
                         TensorUtil::TensorData x,
                         TensorUtil::TensorData label);

private:
    void m_runBackProp() override;
};
}

#endif
