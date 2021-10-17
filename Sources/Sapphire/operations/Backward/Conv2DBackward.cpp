// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cassert>
#include <Sapphire/operations/Backward/Conv2DBackward.hpp>
#include <Sapphire/compute/ConvolutionOps.hpp>
#include <Sapphire/compute/BasicOps.hpp>

namespace Sapphire::BackProp
{
constexpr int dxIdx = 0;
constexpr int dyIdx = 0;
constexpr int kernelIdx = 0;
constexpr int biasIdx = 1;
constexpr int xIdx = 0;

Conv2DBackProp::Conv2DBackProp(
    TensorUtil::TensorData dx, TensorUtil::TensorData dy,
    TensorUtil::TensorData filter, TensorUtil::TensorData bias,
    TensorUtil::TensorData x,
    std::pair<int, int> stride, std::pair<int, int> dilation,
    std::pair<int, int> padding,
    Optimizer::Optimizer* optimizer)
    : BackPropWrapper({ std::move(dx) }, { std::move(dy) },
                      { std::move(filter), std::move(bias) }, { std::move(x) },
                      {},
                      optimizer),
      m_stride(std::move(stride)),
      m_dilation(std::move(dilation)),
      m_padding(std::move(padding)),
      m_hasBias(true)
{
}

Conv2DBackProp::Conv2DBackProp(
    TensorUtil::TensorData dx, TensorUtil::TensorData dy,
    TensorUtil::TensorData filter,
    TensorUtil::TensorData x,
    std::pair<int, int> stride,
    std::pair<int, int> dilation, std::pair<int, int> padding,
    Optimizer::Optimizer* optimizer)
    : BackPropWrapper({ std::move(dx) }, { std::move(dy) },
                      { std::move(filter) }, { std::move(x) },
                      {}, optimizer),
      m_stride(std::move(stride)),
      m_dilation(std::move(dilation)),
      m_padding(std::move(padding)),
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
                                   kernel.GetDevice());
    dKernel.SetMode(kernel.Mode());

    Compute::Conv2DBackward(dx, dKernel, dy, x, kernel, strideRow, strideCol,
                            rowPadding, colPadding, dilationRow, dilationCol);
    m_optimizer->operator()(kernel, dKernel);

    if (m_hasBias)
    {
        auto bias = m_trainableData[biasIdx];
        auto mean0Shape = dy.GetShape();
        mean0Shape.SetCol(1);
        auto mean1Shape = mean0Shape;
        mean1Shape.SetRow(1);
        auto mean2Shape = mean1Shape;
        mean2Shape.Set(0, 1);
        assert(bias.GetShape() == mean2Shape);
        TensorUtil::TensorData mean0(mean0Shape, dy.GetType(), dy.GetDevice());
        TensorUtil::TensorData mean1(mean1Shape, dy.GetType(), dy.GetDevice());
        TensorUtil::TensorData mean2(mean2Shape, dy.GetType(), dy.GetDevice());
        mean0.SetMode(dy.Mode());
        mean1.SetMode(dy.Mode());
        mean2.SetMode(dy.Mode());
        Compute::Mean(mean0, dy, mean0Shape.Dim() - 1);
        Compute::Mean(mean1, mean0, mean0Shape.Dim() - 2);
        Compute::Mean(mean2, mean1, 0);

        m_optimizer->operator()(bias, mean1);
    }
}
}
