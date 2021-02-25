// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_LINEAR_HPP
#define MOTUTAPU_LINEAR_HPP

#include <Motutapu/tensor/Tensor.hpp>
#include <ostream>

namespace Motutapu::NN
{
class Linear
{
 public:
    Linear(unsigned int inputFeatureSize, unsigned int outputFeatureSize,
           const Device& device, bool bias = true, bool isSparse = false);

    Tensor operator()(const Tensor& tensor) const;

 private:
    int m_unitKey = -1;
    unsigned int m_outputs;
    Type m_type = Type::Dense;
    bool m_bias;
};
}  // namespace Motutapu::NN

#endif  // MOTUTAPU_LINEAR_HPP
