// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/operations/Backward/Conv2DBackward.hpp>
#include <Sapphire/compute/ConvolutionOps.hpp>

namespace Sapphire::BackProp
{
const int dxIdx = 0;
const int dyIdx = 0;
const int kernelIdx = 0;
const int biasIdx = 1;
const int xIdx = 0;

Conv2DBackProp::Conv2DBackProp(
    TensorUtil::TensorData dx, TensorUtil::TensorData dy,
    TensorUtil::TensorData filter, TensorUtil::TensorData bias,
    TensorUtil::TensorData x,
    std::pair<int, int> stride, std::pair<int, int> dilation,
    std::pair<int, int> padding,
    Util::SharedPtr<Optimizer::Optimizer> optimizer)
    : BackPropWrapper({ std::move(dx) }, { std::move(dy) },
                      { std::move(filter), std::move(bias) }, { std::move(x) },
                      {},
                      std::move(optimizer)),
      m_stride(stride),
      m_dilation(dilation),
      m_padding(padding),
      m_hasBias(true)
{
}

Conv2DBackProp::Conv2DBackProp(
    TensorUtil::TensorData dx, TensorUtil::TensorData dy,
    TensorUtil::TensorData filter,
    TensorUtil::TensorData x,
    std::pair<int, int> stride,
    std::pair<int, int> dilation, std::pair<int, int> padding,
    Util::SharedPtr<Optimizer::Optimizer> optimizer)
    : BackPropWrapper({ std::move(dx) }, { std::move(dy) },
                      { std::move(filter) }, { std::move(x) },
                      {}, std::move(optimizer)),
      m_stride(stride),
      m_dilation(dilation),
      m_padding(padding),
      m_hasBias(false)
{
}

void Conv2DBackProp::m_runBackProp()
{
    auto kernel = m_trainableData[kernelIdx];
    auto dx = m_dxVector[dxIdx];
    auto dy = m_dyVector[dyIdx];
    auto x = m_constants[xIdx];

    const auto [strideRow, strideCol] = m_stride;
    const auto [dilationRow, dilationCol] = m_dilation;
    const auto [rowPadding, colPadding] = m_padding;

    TensorUtil::TensorData dKernel(kernel.GetShape(), kernel.GetType(),
                                   kernel.GetCudaDevice());

    Compute::Conv2DBackward(dx, dKernel, dy, x, kernel, strideRow, strideCol,
                            rowPadding, colPadding, dilationRow, dilationCol);
    m_optimizer->operator()(kernel, dKernel);

    if (m_hasBias)
    {
        auto bias = m_trainableData[biasIdx];
        m_optimizer->operator()(bias, dy);
    }
}
}
