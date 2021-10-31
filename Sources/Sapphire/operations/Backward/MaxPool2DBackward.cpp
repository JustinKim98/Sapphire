// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/operations/Backward/MaxPool2DBackward.hpp>
#include <Sapphire/compute/ConvolutionOps.hpp>

namespace Sapphire::BackProp
{
constexpr int dxIdx = 0;
constexpr int dyIdx = 0;
constexpr int xIdx = 0;
constexpr int yIdx = 1;

MaxPool2DBackProp::MaxPool2DBackProp(TensorData dx, TensorData dy, TensorData x,
                                     TensorData y,
                                     std::pair<int, int> windowSize,
                                     std::pair<int, int> stride,
                                     std::pair<int, int> padSize)
    : BackPropWrapper({ std::move(dx) }, { std::move(dy) },
                      { std::move(x), std::move(y) }, {}),
      m_windowSize(windowSize),
      m_stride(stride),
      m_padSize(padSize)
{
}

void MaxPool2DBackProp::m_runBackProp()
{
    auto dx = m_dxVector[dxIdx];
    auto dy = m_dyVector[dyIdx];
    const auto& x = m_constants[xIdx];
    const auto& y = m_constants[yIdx];

    const auto [windowRows, windowCols] = m_windowSize;
    const auto [strideRow, strideCol] = m_stride;
    const auto [rowPadding, colPadding] = m_padSize;

    Compute::MaxPool2DBackward(dx, dy, x, y, windowRows, windowCols, strideRow,
                               strideCol, rowPadding, colPadding);
}
}
