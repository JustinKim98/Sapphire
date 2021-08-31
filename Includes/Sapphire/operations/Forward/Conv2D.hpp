// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_NN_CONV2D_HPP
#define SAPPHIRE_NN_CONV2D_HPP

#include <Sapphire/tensor/Tensor.hpp>
#include <Sapphire/operations/optimizers/Optimizer.hpp>
#include <Sapphire/util/SharedPtr.hpp>
#include <Sapphire/operations/Unit.hpp>
#include <Sapphire/operations/Initializers/Initialize.hpp>
#include <utility>

namespace Sapphire::NN
{
class Conv2D : public Unit
{
public:
    Conv2D(int inChannels, int outChannels, std::pair<int, int> inputSize,
           std::pair<int, int> kernelSize,
           std::pair<int, int> stride, std::pair<int, int> padSize,
           std::pair<int, int> dilation, bool useBias,
           Util::SharedPtr<Optimizer::Optimizer> optimizer,
           std::unique_ptr<Initialize::Initializer> kernelInitializer,
           std::unique_ptr<Initialize::Initializer> biasInitializer,
           CudaDevice device, bool isSparse = false);
    ~Conv2D() override = default;

    Conv2D(const Conv2D& conv2D) = default;
    Conv2D(Conv2D&& conv2D) = default;
    Conv2D& operator=(const Conv2D& conv2D) = default;
    Conv2D& operator=(Conv2D&& conv2D) noexcept = default;

    Tensor operator()(Tensor& tensor);

private:
    [[nodiscard]] int m_registerOutputTensor(
        const TensorUtil::TensorDescriptor& xDesc) const;

    [[nodiscard]] bool m_checkArguments(
        std::vector<TensorUtil::TensorDescriptor> arguments) override;

    int m_inputChannels, m_outputChannels;
    std::pair<int, int> m_inputSize, m_kernelSize, m_stride, m_padSize,
                        m_dilation;
    bool m_useBias;

    CudaDevice m_device;
    bool m_isSparse;
    int m_yRows, m_yCols;

    Util::SharedPtr<Optimizer::Optimizer> m_optimizer;
};
};

#endif
