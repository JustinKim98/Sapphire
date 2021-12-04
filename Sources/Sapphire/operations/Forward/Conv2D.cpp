// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/compute/ConvolutionOps.hpp>
#include <Sapphire/operations/Backward/Conv2DBackward.hpp>
#include <Sapphire/tensor/CreateTensor.hpp>
#include <Sapphire/operations/Forward/Conv2D.hpp>
#include <Sapphire/util/Shape.hpp>
#include <Sapphire/util/UnitUtils.hpp>
#include <Sapphire/tensor/CreateTensor.hpp>

namespace Sapphire::NN
{
int Conv2D::m_unitIdCount = 0;

Conv2D::Conv2D(int yChannels, int xChannels, std::pair<int, int> filterSize,
               std::pair<int, int> stride, std::pair<int, int> padSize,
               std::pair<int, int> dilation, bool useBias)
    : Unit(std::string("Conv2D") + std::to_string(m_unitIdCount++)),
      m_yChannels(yChannels),
      m_xChannels(xChannels),
      m_useBias(useBias),
      m_filterSize(filterSize),
      m_stride(stride),
      m_padSize(padSize),
      m_dilation(dilation)
{
    const CudaDevice gpu(0, "cuda0");
    const auto [filterRows, filterCols] = filterSize;
    //const auto kernelSize = filterRows * filterCols * yChannels * xChannels;
    //const auto filterStd = 1.0f / std::sqrt(static_cast<float>(kernelSize));

    m_filter = MakeTensor(
        Shape({ yChannels, xChannels, filterRows, filterCols }), gpu,
        M<Initialize::Normal>(0.0f, 1.0f), true);
    if (useBias)
    {
        //const auto biasStd = 1.0f / std::sqrt(static_cast<float>(yChannels));
        m_bias = MakeTensor(Shape({ yChannels }), gpu,
                            M<Initialize::Normal>(0.0f, 1.0f),
                            true);
    }
}

Conv2D::Conv2D(std::string name, int yChannels, int xChannels,
               std::pair<int, int> filterSize,
               std::pair<int, int> stride, std::pair<int, int> padSize,
               std::pair<int, int> dilation,
               bool useBias)
    : Unit(std::move(name)),
      m_yChannels(yChannels),
      m_xChannels(xChannels),
      m_useBias(useBias),
      m_filterSize(filterSize),
      m_stride(stride),
      m_padSize(padSize),
      m_dilation(dilation)
{
}

Tensor Conv2D::operator()(Tensor& x)
{
    if (!m_useBias)
        throw std::runtime_error(
            "Conv2D::operator() - This unit was not configured to use bias, "
            "but it "
            "was called with bias");

    auto inputRows = x.GetShape().At(-2);
    auto inputCols = x.GetShape().At(-1);
    const auto [dilationRows, dilationCols] = m_dilation;
    const auto [rowPadding, colPadding] = m_padSize;
    const auto [strideRows, strideCols] = m_stride;
    const auto [filterRows, filterCols] = m_filterSize;
    m_inputSize = std::make_pair(inputRows, inputCols);

    m_yRows =
        (inputRows + 2 * rowPadding - dilationRows * (filterRows - 1) - 1) /
        strideRows +
        1;
    m_yCols =
        (inputCols + 2 * colPadding - dilationCols * (filterCols - 1) - 1) /
        strideCols +
        1;

    if (m_yRows <= 0 || m_yCols <= 0)
        throw std::invalid_argument("Conv2D::Conv2D - invalid argument");

    if (m_filter.Mode() != x.Mode())
    {
        if (x.Mode() == ComputeMode::Cuda)
            m_filter.ToCuda();
        else
            m_filter.ToHost();
    }
    if (m_useBias)
    {
        if (x.Mode() == ComputeMode::Cuda)
            m_bias.ToCuda();
        else
            m_bias.ToHost();
    }

    if (m_useBias)
        return this->operator()(x, m_filter, m_bias);
    return this->operator()(x, m_filter);
}


Tensor Conv2D::operator()(Tensor& tensor, Tensor& filter, Tensor& bias)
{
    if (!m_useBias)
        throw std::runtime_error(
            "Conv2D::operator() - This unit was not configured to use bias, "
            "but it "
            "was called with bias");

    auto inputRows = tensor.GetShape().At(-2);
    auto inputCols = tensor.GetShape().At(-1);
    const auto [dilationRows, dilationCols] = m_dilation;
    const auto [rowPadding, colPadding] = m_padSize;
    const auto [strideRows, strideCols] = m_stride;
    const auto [filterRows, filterCols] = m_filterSize;
    m_inputSize = std::make_pair(inputRows, inputCols);

    m_yRows =
        (inputRows + 2 * rowPadding - dilationRows * (filterRows - 1) - 1) /
        strideRows +
        1;
    m_yCols =
        (inputCols + 2 * colPadding - dilationCols * (filterCols - 1) - 1) /
        strideCols +
        1;

    if (m_yRows <= 0 || m_yCols <= 0)
        throw std::invalid_argument("Conv2D::Conv2D - invalid argument");

    auto mode = tensor.Mode();
    auto& model = ModelManager::CurModel();

    auto device = tensor.GetDevice();
    auto& xDesc = model.GetDescriptor(tensor.TensorDescriptorKey());
    auto& filterDesc = model.GetDescriptor(filter.TensorDescriptorKey());
    auto& biasDesc = model.GetDescriptor(bias.TensorDescriptorKey());
    m_checkArguments({ &xDesc, &filterDesc, &biasDesc });
    const auto yKey = m_registerOutputTensor(xDesc);
    auto& yDesc = model.GetDescriptor(yKey);
    yDesc.SetMode(mode);

    auto filterData = filterDesc.GetForwardData();
    auto biasData = biasDesc.GetForwardData();
    auto x = xDesc.GetForwardData();
    auto dx = xDesc.GetBackwardData();
    auto y = yDesc.GetForwardData();
    auto dy = yDesc.GetBackwardData();

    if (device != filter.GetDevice())
        throw std::runtime_error(
            "NN::Conv2D::operator() - kernel and tensor device mismatch");

    Util::ChangeTensorDataDimension(4, x, dx, y, dy);

    Compute::Initialize::Zeros(y);
    Compute::Conv2DForward(y, x, filterData, strideRows, strideCols,
                           dilationRows, dilationCols, rowPadding, colPadding);

    if (device != bias.GetDevice())
        throw std::runtime_error(
            "NN::Conv2D::operator() - bias and tensor device mismatch");
    biasData.Reshape(Shape({ 1, m_yChannels, 1, 1 }));

    auto copy = y.GetDataCopy();
    auto biasCopy = biasData.GetDataCopy();
    Compute::Add(y, y, biasData);
    auto yCopy = y.GetDataCopy();
    auto* backPropWrapper =
        new BackProp::Conv2DBackProp(m_name, dx, dy, filterData, biasData, x,
                                     m_stride, m_dilation, m_padSize);
    Util::SaveHistory(backPropWrapper, std::make_tuple(&xDesc),
                      std::make_tuple(&yDesc));

    return Tensor(yKey);
}

Tensor Conv2D::operator()(Tensor& tensor, Tensor& filter)
{
    if (m_useBias == true)
        throw std::runtime_error(
            "Conv2D::operator() - This unit was configured to use bias, but it "
            "wasn't called with bias");
    auto mode = tensor.Mode();
    auto& model = ModelManager::CurModel();

    auto device = tensor.GetDevice();
    auto& xDesc = model.GetDescriptor(tensor.TensorDescriptorKey());
    auto& filterDesc = model.GetDescriptor(filter.TensorDescriptorKey());
    m_checkArguments({ &xDesc, &filterDesc });
    const auto yKey = m_registerOutputTensor(xDesc);
    auto& yDesc = model.GetDescriptor(yKey);
    yDesc.SetMode(mode);

    auto [dilationRows, dilationCols] = m_dilation;
    auto [rowPadding, colPadding] = m_padSize;
    auto [strideRows, strideCols] = m_stride;

    auto filterData = filterDesc.GetForwardData();
    auto x = xDesc.GetForwardData();
    auto dx = xDesc.GetBackwardData();
    auto y = yDesc.GetForwardData();
    auto dy = yDesc.GetBackwardData();

    if (device != filter.GetDevice())
        throw std::runtime_error(
            "NN::Conv2D::operator() - kernel and tensor device mismatch");

    Util::ChangeTensorDataDimension(4, x, dx, y, dy);

    //! TODO : Do we need this?
    Compute::Initialize::Zeros(y);
    Compute::Conv2DForward(y, x, filterData, strideRows, strideCols,
                           dilationRows, dilationCols, rowPadding, colPadding);

    auto* backPropWrapper = new BackProp::Conv2DBackProp(m_name,
        dx, dy, filterData, x, m_stride, m_dilation, m_padSize);

    Util::SaveHistory(backPropWrapper, std::make_tuple(&xDesc),
                      std::make_tuple(&yDesc));

    return Tensor(yKey);
}

int Conv2D::m_registerOutputTensor(
    const TensorUtil::TensorDescriptor& xDesc) const
{
    auto& model = ModelManager::CurModel();
    const auto x = xDesc.GetForwardData();
    const Shape xShape = xDesc.GetShape();
    Shape yShape = xShape;
    yShape[-1] = m_yCols;
    yShape[-2] = m_yRows;
    yShape[- 3] = m_yChannels;
    const auto yKey =
        model.RegisterTensorDescriptor(yShape, x.GetType(), xDesc.GetDevice());
    return yKey;
}

void Conv2D::m_checkArguments(
    std::vector<TensorUtil::TensorDescriptor*> arguments) const
{
    const auto xDescPtr = arguments.at(0);
    const auto filterDescPtr = arguments.at(1);
    const auto xShape = xDescPtr->GetShape();
    const auto filterShape = filterDescPtr->GetShape();
    const auto [filterRows, filterCols] = m_filterSize;
    const auto [xRows, xCols] = m_inputSize;
    const auto device = xDescPtr->GetDevice();

    //! Check condition of X
    if (xShape.Dim() < 4)
        throw std::invalid_argument(
            "NN::Conv2D - input should have shape of (*, C, H, W)");
    if (xShape.At(xShape.Dim() - 3) != m_xChannels)
        throw std::invalid_argument(
            "NN::Conv2D - size of x channels does not match ");
    if (xShape.Rows() != xRows)
        throw std::invalid_argument(
            "NN::Conv2D - size of x height does not match ");
    if (xShape.Cols() != xCols)
        throw std::invalid_argument(
            "NN::Conv2D - size of x width does not match ");

    //! Check condition of filter
    if (filterShape !=
        Shape({ m_yChannels, m_xChannels, filterRows, filterCols }))
        throw std::invalid_argument(
            "NN::Conv2D - filter should have shape of (yC, xC, filterH, "
            "filterW)");
    if (filterDescPtr->GetDevice() != device)
        throw std::invalid_argument(
            "NN::Conv2D - filter is configured to use different device with x");

    //! Check condition of bias
    if (m_useBias)
    {
        const auto biasDescPtr = arguments.at(2);
        const auto biasShape = biasDescPtr->GetShape();
        if (biasShape != Shape({ 1, m_yChannels }) &&
            biasShape != Shape({ m_yChannels }))
            throw std::invalid_argument(
                "NN::Conv2D - Bias should have shape of (1, yChannels) or "
                "(yChannels)");
        if (biasDescPtr->GetDevice() != device)
            throw std::invalid_argument(
                "NN::Conv2D - Bias is configured to use different device with "
                "x");
    }
}
} // namespace Sapphire::NN
