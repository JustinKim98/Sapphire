// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_NN_CONV2D_HPP
#define SAPPHIRE_NN_CONV2D_HPP

#include <Sapphire/operations/Initializers/Initialize.hpp>
#include <Sapphire/operations/Unit.hpp>
#include <Sapphire/operations/optimizers/Optimizer.hpp>
#include <Sapphire/tensor/Tensor.hpp>
#include <utility>

namespace Sapphire::NN
{
class Conv2D : public Unit
{
 public:
    Conv2D(int yChannels, int xChannels, std::pair<int, int> inputSize,
           std::pair<int, int> filterSize, std::pair<int, int> stride,
           std::pair<int, int> padSize, std::pair<int, int> dilation,
           Optimizer::Optimizer* optimizer, bool useBias);

    ~Conv2D() override = default;

    Conv2D(const Conv2D& conv2D) = default;
    Conv2D(Conv2D&& conv2D) = default;
    Conv2D& operator=(const Conv2D& conv2D) = default;
    Conv2D& operator=(Conv2D&& conv2D) noexcept = default;

    Tensor operator()(Tensor& tensor, Tensor& filter, Tensor& bias);
    Tensor operator()(Tensor& tensor, Tensor& filter);

 private:
    [[nodiscard]] int m_registerOutputTensor(
        const TensorUtil::TensorDescriptor& xDesc) const;

    void m_checkArguments(
        std::vector<TensorUtil::TensorDescriptor*> arguments) const override;

    std::pair<int, int> m_inputSize, m_filterSize, m_stride, m_padSize,
        m_dilation;
    int m_yChannels = -1;
    int m_xChannels = -1;
    bool m_useBias = false;
    bool m_isSparse = false;
    int m_yRows = -1;
    int m_yCols = -1;
};
};  // namespace Sapphire::NN

#endif
