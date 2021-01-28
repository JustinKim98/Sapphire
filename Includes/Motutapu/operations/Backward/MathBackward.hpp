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
    explicit MulBackProp(unsigned int tensorKeyA, unsigned int tensorKeyB)
        : BackPropWrapper({ tensorKeyA, tensorKeyB }, false)
    {
    }

    void Backward(std::vector<Util::TensorData>& outputs,
                  const Util::TensorData& input) const override;
};

class AddBackProp : public BackPropWrapper
{
 public:
    explicit AddBackProp(unsigned int tensorKeyA, unsigned int tensorKeyB)
        : BackPropWrapper({ tensorKeyA, tensorKeyB }, false)
    {
    }

    void Backward(std::vector<Util::TensorData>& outputs,
                  const Util::TensorData& input) const override;
};

class AddBackPropInplace : public BackPropWrapper
{
 public:
    explicit AddBackPropInplace(unsigned int tensorKeyA)
        : BackPropWrapper({ tensorKeyA }, true)
    {
    }

    void Backward(std::vector<Util::TensorData>& outputs,
                  const Util::TensorData& input) const override;
};

}  // namespace Motutapu::BackProp

#endif
