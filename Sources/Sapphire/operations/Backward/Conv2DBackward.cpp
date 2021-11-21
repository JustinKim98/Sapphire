// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cassert>
#include <Sapphire/operations/Backward/Conv2DBackward.hpp>
#include <Sapphire/compute/ConvolutionOps.hpp>
#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/Model.hpp>

namespace Sapphire::BackProp
{
constexpr int dxIdx = 0;
constexpr int dyIdx = 0;
constexpr int kernelIdx = 0;
constexpr int biasIdx = 1;
constexpr int xIdx = 0;

Conv2DBackProp::Conv2DBackProp(std::string name,
                               TensorData dx,
                               TensorData dy,
                               TensorData filter,
                               TensorData bias,
                               TensorData x,
                               std::pair<int, int> stride,
                               std::pair<int, int> dilation,
                               std::pair<int, int> padding)
    : BackPropWrapper(std::move(name), { std::move(dx) }, { std::move(dy) },
                      { std::move(filter), std::move(bias) }, { std::move(x) },
                      {}),
      m_stride(std::move(stride)),
      m_dilation(std::move(dilation)),
      m_padding(std::move(padding)),
      m_hasBias(true)
{
}

Conv2DBackProp::Conv2DBackProp(std::string name,
                               TensorData dx,
                               TensorData dy,
                               TensorData filter,
                               TensorData x,
                               std::pair<int, int> stride,
                               std::pair<int, int> dilation,
                               std::pair<int, int> padding)
    : BackPropWrapper(std::move(name), { std::move(dx) }, { std::move(dy) },
                      { std::move(filter) }, { std::move(x) }, {}),
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
    const auto& x = m_constants[xIdx];

    const auto [strideRow, strideCol] = m_stride;
    const auto [dilationRow, dilationCol] = m_dilation;
    const auto [rowPadding, colPadding] = m_padding;

    TensorData dKernel(kernel.GetShape(), kernel.GetType(),
                       kernel.GetDevice());
    dKernel.SetMode(kernel.Mode());

    Compute::Conv2DBackward(dx, dKernel, dy, x, kernel, strideRow, strideCol,
                            rowPadding, colPadding, dilationRow, dilationCol);
    ModelManager::CurModel().GetOptimizer()->operator()(kernel, dKernel,
        m_name);

    if (m_hasBias)
    {
        auto bias = m_trainableData[biasIdx];
        auto mean0Shape = dy.GetShape();
        mean0Shape[-1] = 1;
        auto mean1Shape = mean0Shape;
        mean1Shape[-2] = 1;
        auto mean2Shape = mean1Shape;
        mean2Shape.Set(0, 1);
        assert(bias.GetShape() == mean2Shape);
        TensorData mean0(mean0Shape, dy.GetType(), dy.GetDevice());
        TensorData mean1(mean1Shape, dy.GetType(), dy.GetDevice());
        TensorData mean2(mean2Shape, dy.GetType(), dy.GetDevice());
        mean0.SetMode(dy.Mode());
        mean1.SetMode(dy.Mode());
        mean2.SetMode(dy.Mode());
        Compute::Mean(mean0, dy, mean0Shape.Dim() - 1);
        Compute::Mean(mean1, mean0, mean0Shape.Dim() - 2);
        Compute::Mean(mean2, mean1, 0);

        ModelManager::CurModel().GetOptimizer()->operator()(bias, mean1,
            m_name);
    }
}
}
