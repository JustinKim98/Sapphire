// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/operations/Forward/MaxPool2D.hpp>
#include <Sapphire/operations/Backward/MaxPool2DBackward.hpp>
#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/compute/ConvolutionOps.hpp>
#include <Sapphire/util/Shape.hpp>
#include <Sapphire/util/UnitUtils.hpp>
#include <Sapphire/compute/Initialize.hpp>

namespace Sapphire::NN
{
MaxPool2D::MaxPool2D(int channels,
                     std::pair<int, int> windowSize, std::pair<int, int> stride,
                     std::pair<int, int> padSize)
    : Unit(),
      m_channels(channels)
{
    const auto [filterRows, filterCols] = windowSize;

    m_windowSize = std::make_pair(filterRows, filterCols);
    m_stride = stride;
    m_padSize = padSize;
}

Tensor MaxPool2D::operator()(const Tensor& tensor)
{
    auto mode = tensor.Mode();
    auto& model = ModelManager::CurModel();

    const auto inputRows = tensor.GetShape().At(-2);
    const auto inputCols = tensor.GetShape().At(-1);
    const auto [windowRows, windowCols] = m_windowSize;
    const auto [rowPadding, colPadding] = m_padSize;
    const auto [strideRows, strideCols] = m_stride;

    m_yRows =
        (inputRows + 2 * rowPadding - (windowRows - 1) - 1) / strideRows + 1;
    m_yCols =
        (inputCols + 2 * colPadding - (windowCols - 1) - 1) / strideCols + 1;

    if (m_yRows <= 0 || m_yCols <= 0)
        throw std::invalid_argument(
            "MaxPool2D::MaxPool2D - invalid argument (could not derive size of y)");

    auto device = tensor.GetDevice();
    auto& xDesc = model.GetDescriptor(tensor.TensorDescriptorKey());
    m_checkArguments({ &xDesc });
    const auto yKey = m_registerOutputTensor(xDesc);
    auto& yDesc = model.GetDescriptor(yKey);
    yDesc.SetMode(mode);

    auto x = xDesc.GetForwardData();
    auto dx = xDesc.GetBackwardData();
    auto y = yDesc.GetForwardData();
    auto dy = yDesc.GetBackwardData();

    Util::ChangeTensorDataDimension(4, x, dx, y, dy);

    //! TODO : Do we need this?
    Compute::Initialize::Zeros(y);
    Compute::MaxPool2DForward(y, x, windowRows, windowCols, strideRows,
                              strideCols, rowPadding, colPadding);

    auto* backPropWrapper = new BackProp::MaxPool2DBackProp(
        dx, dy, x, y, m_windowSize, m_stride, m_padSize);

    Util::SaveHistory(backPropWrapper, std::make_tuple(&xDesc),
                      std::make_tuple(&yDesc));

    return Tensor(yKey);
}


int MaxPool2D::m_registerOutputTensor(
    const TensorUtil::TensorDescriptor& xDesc) const
{
    auto& model = ModelManager::CurModel();
    const auto x = xDesc.GetForwardData();
    const Shape xShape = xDesc.GetShape();
    Shape yShape = xShape;
    yShape[-1] = m_yCols;
    yShape[-2] = m_yRows;
    yShape[-3] = m_channels;
    const auto yKey =
        model.RegisterTensorDescriptor(yShape, x.GetType(), xDesc.GetDevice());
    return yKey;
}

void MaxPool2D::m_checkArguments(
    std::vector<TensorUtil::TensorDescriptor*> arguments) const
{
    const auto xDescPtr = arguments.at(0);
    const auto xShape = xDescPtr->GetShape();
    const auto device = xDescPtr->GetDevice();

    //! Check condition of X
    if (xShape.Dim() < 4)
        throw std::invalid_argument(
            "NN::MaxPool2D - input should have shape of (*, C, H, W)");
    if (xShape.At(xShape.Dim() - 3) != m_channels)
        throw std::invalid_argument(
            "NN::MaxPool2D - size of x channels does not match ");
}
}
