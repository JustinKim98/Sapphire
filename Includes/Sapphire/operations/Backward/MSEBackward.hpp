// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Sapphire_BACKPROP_MSEBACKPROP_HPP
#define Sapphire_BACKPROP_MSEBACKPROP_HPP

#include <Sapphire/operations/Backward/BackPropWrapper.hpp>

namespace Sapphire::BackProp
{
class MSEBackward : public BackPropWrapper
{
 public:
    MSEBackward(const TensorUtil::TensorData& x, TensorUtil::TensorData dx,
                const TensorUtil::TensorData& label, TensorUtil::TensorData dy);

    bool InvokeBackProp(const TensorUtil::TensorData& input) override;
};
}  // namespace Sapphire::BackProp

#endif  // Sapphire_MSEBACKPROP_HPP
