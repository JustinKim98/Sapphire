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
    explicit MulBackProp(TensorUtil::TensorData A,
                         TensorUtil::TensorData gradientA,
                         TensorUtil::TensorData B,
                         TensorUtil::TensorData gradientB,
                         TensorUtil::TensorData gradientIn);

    bool InvokeBackProp(const TensorUtil::TensorData& input) override;

 private:
    void m_backProp();
};


}  // namespace Motutapu::BackProp

#endif
