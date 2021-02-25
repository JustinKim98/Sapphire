// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_BACKPROP_MSEBACKPROP_HPP
#define MOTUTAPU_BACKPROP_MSEBACKPROP_HPP

#include <Motutapu/operations/Backward/BackPropWrapper.hpp>

namespace Motutapu::BackProp
{
class MSEBackProp : public BackPropWrapper
{
 public:
    explicit MSEBackProp(const TensorUtil::TensorData& x,
                         TensorUtil::TensorData dx,
                         const TensorUtil::TensorData& label,
                         TensorUtil::TensorData dy);
    bool InvokeBackProp(const TensorUtil::TensorData& input) override;
};
}  // namespace Motutapu::BackProp

#endif  // MOTUTAPU_MSEBACKPROP_HPP
