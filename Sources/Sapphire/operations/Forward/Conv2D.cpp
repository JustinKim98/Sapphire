// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/operations/Forward/Conv2D.hpp>
#include <Sapphire/compute/ConvolutionOps.hpp>
#include <Sapphire/operations/Backward/Conv2DBackward.hpp>
#include <Sapphire/util/UnitUtils.hpp>
#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/util/Shape.hpp>

namespace Sapphire::NN
{
Conv2D::Conv2D(std::pair<int, int> inputSize, std::pair<int, int> stride,
               std::pair<int, int> padSize, std::pair<int, int> dilation,
               Util::SharedPtr<Optimizer::Optimizer> optimizer,
               Tensor kernel,
               Tensor bias)
{
    const auto kernelShape = kernel.GetShape();
    const auto device = kernel.GetDevice();

    if (kernelShape.Dim() != 4)
    {
        throw std::invalid_argument(
            "NN::Conv2D - kernel should be in 4 dimensions");
    }

    if (bias.TensorDescriptorKey() > 0)
    {
        const auto biasShape = bias.GetShape();
        if (biasShape.Dim() != 1)
            throw std::invalid_argument(
                "NN::Conv2D - Bias should be in 1 dimension");
        if (biasShape.At(0) != kernelShape.At(0))
        {
            throw std::invalid_argument(
                "NN::Conv2D - Bias should have same size with output channels");
        }

        if (bias.GetDevice() != device)
            throw std::invalid_argument(
                "NN::Conv2D - Device mismatch between kernel and bias");
        m_useBias = true;
    }

    const auto outputChannels = kernelShape.At(0);
    const auto inputChannels = kernelShape.At(1);
    const auto kernelRows = kernelShape.At(2);
    const auto kernelCols = kernelShape.At(3);
    const auto [dilationRows, dilationCols] = dilation;
    const auto [inputRows, inputCols] = inputSize;
    const auto [rowPadding, colPadding] = padSize;
    const auto [strideRows, strideCols] = stride;

    m_xChannels = inputChannels;
    m_yChannels = outputChannels;
    m_inputSize = inputSize;
    m_kernelSize = std::make_pair(kernelRows, kernelCols);
    m_stride = stride;
    m_dilation = dilation;
    m_isSparse = false;
    m_optimizer = std::move(optimizer);
    m_padSize = padSize;

    m_yRows =
        (inputRows + 2 * rowPadding - dilationRows * (kernelRows - 1) - 1) /
        strideRows +
        1;
    m_yCols =
        (inputCols + 2 * colPadding - dilationCols * (kernelCols - 1) - 1) /
        strideCols +
        1;

    const int kernelDescKey = kernel.TensorDescriptorKey();
    auto& model = ModelManager::CurModel();
    const auto& kernelDesc = model.GetDescriptor(kernelDescKey);
    m_trainableDataMap["kernel"] = kernelDesc.GetForwardData();

    if (bias.TensorDescriptorKey() > 0)
    {
        const int biasDescKey = bias.TensorDescriptorKey();
        const auto& biasDesc = model.GetDescriptor(biasDescKey);
        m_trainableDataMap["bias"] = biasDesc.GetForwardData();
    }
}

Tensor Conv2D::operator()(Tensor& tensor)
{
    auto mode = tensor.Mode();
    auto& model = ModelManager::CurModel();

    auto device = tensor.GetDevice();
    auto& xDesc = model.GetDescriptor(tensor.TensorDescriptorKey());
    m_checkArguments({ &xDesc });
    const auto yKey = m_registerOutputTensor(xDesc);
    auto& yDesc = model.GetDescriptor(yKey);
    yDesc.SetMode(mode);

    auto [dilationRows, dilationCols] = m_dilation;
    auto [rowPadding, colPadding] = m_padSize;
    auto [strideRows, strideCols] = m_stride;

    auto x = xDesc.GetForwardData();
    auto dx = xDesc.GetBackwardData();
    auto y = yDesc.GetForwardData();
    auto dy = yDesc.GetBackwardData();

    auto kernel = m_trainableDataMap.at("kernel");
    if (device != kernel.GetDevice())
        throw std::runtime_error(
            "NN::Conv2D::operator() - kernel and tensor device mismatch");

    Util::ChangeTensorDataDimension(4, x, dx, y, dy);

    Compute::Initialize::Zeros(y);
    Compute::Conv2DForward(y, x, kernel, strideRows, strideCols, dilationRows,
                           dilationCols, rowPadding, colPadding);

    if (m_useBias)
    {
        auto bias = m_trainableDataMap.at("bias");
        if (device != bias.GetDevice())
            throw std::runtime_error(
                "NN::Conv2D::operator() - bias and tensor device mismatch");
        Compute::Add(y, y, bias);
        auto* backPropWrapper = new BackProp::Conv2DBackProp(
            dx, dy, kernel, bias, x, m_stride, m_dilation, m_padSize,
            m_optimizer);
        Util::SaveHistory(backPropWrapper, std::make_tuple(&xDesc),
                          std::make_tuple(&yDesc));
    }
    else
    {
        auto* backPropWrapper = new BackProp::Conv2DBackProp(
            dx, dy, kernel, x, m_stride, m_dilation, m_padSize, m_optimizer);
        Util::SaveHistory(backPropWrapper, std::make_tuple(&xDesc),
                          std::make_tuple(&yDesc));
    }
    return Tensor(yKey);
}

int Conv2D::m_registerOutputTensor(
    const TensorUtil::TensorDescriptor& xDesc) const
{
    auto& model = ModelManager::CurModel();
    const auto x = xDesc.GetForwardData();
    const Shape xShape = xDesc.GetShape();
    Shape yShape = xShape;
    yShape.SetCol(m_yCols);
    yShape.SetRow(m_yRows);
    yShape[yShape.Dim() - 3] = m_yChannels;
    const auto yKey =
        model.RegisterTensorDescriptor(yShape, x.GetType(), xDesc.GetDevice());
    return yKey;
}

void Conv2D::m_checkArguments(
    std::vector<TensorUtil::TensorDescriptor*> arguments) const
{
    const auto xDescPtr = arguments.at(0);
    const auto xShape = xDescPtr->GetShape();
    if (xShape.Dim() < 3)
        throw std::invalid_argument(
            "NN::Conv2D - input shape must have at least 3 dimension");
    if (xShape.At(xShape.Dim() - 3) != m_xChannels)
        throw std::invalid_argument(
            "NN::Conv2D - Number of channels does not match ");
}
}
