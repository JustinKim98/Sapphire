// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_BACKWARD_MATHBACKWARD_DECL_HPP
#define MOTUTAPU_BACKWARD_MATHBACKWARD_DECL_HPP

#include <Motutapu/operations/Backward/BackPropWrapper.hpp>

namespace Motutapu::BackProp
{
class MulBackProp : public BackPropWrapper
{
 public:
    explicit MulBackProp(TensorUtil::TensorData a, TensorUtil::TensorData da,
                         TensorUtil::TensorData b, TensorUtil::TensorData db,
                         TensorUtil::TensorData dy);

    bool InvokeBackProp(const TensorUtil::TensorData& input) override;
};

class AddBackProp : public BackPropWrapper
{
 public:
    explicit AddBackProp(TensorUtil::TensorData da, TensorUtil::TensorData db,
                         TensorUtil::TensorData dy);

    bool InvokeBackProp(const TensorUtil::TensorData& input) override;
};

}  // namespace Motutapu::BackProp

#endif
