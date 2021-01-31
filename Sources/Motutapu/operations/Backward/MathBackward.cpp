// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/Model.hpp>
#include <Motutapu/operations/Backward/MathBackward.hpp>
#include <iostream>

namespace Motutapu::BackProp
{
void MulBackProp::Backward(std::vector<TensorUtil::TensorData>& outputs,
                           const TensorUtil::TensorData& input) const
{
    std::cout << "MulBackProp invoked" << std::endl;
}

void AddBackProp::Backward(std::vector<TensorUtil::TensorData>& outputs,
                           const TensorUtil::TensorData& input) const
{
    std::cout << "AddBackProp invoked" << std::endl;
}

void AddBackPropInplace::Backward(std::vector<TensorUtil::TensorData>& outputs,
                                  const TensorUtil::TensorData& input) const

{
    std::cout << "AddBackProp IsInplace invoked" << std::endl;
}
}  // namespace Motutapu::BackProp