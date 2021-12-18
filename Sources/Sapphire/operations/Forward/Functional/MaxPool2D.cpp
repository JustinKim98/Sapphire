// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/operations/Forward/Functional/MaxPool2D.hpp>
#include <Sapphire/operations/Backward/MaxPool2DBackward.hpp>
#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/compute/ConvolutionOps.hpp>
#include <Sapphire/util/Shape.hpp>
#include <Sapphire/util/UnitUtils.hpp>

namespace Sapphire::F
{

Tensor MaxPool2D(const Tensor& tensor, std::pair<int, int> windowSize,
                 std::pair<int, int> stride,
                 std::pair<int, int> padSize)
{
    auto mode = tensor.Mode();
    auto& model = ModelManager::CurModel();

    const auto inputRows = tensor.GetShape().At(-2);
    const auto inputCols = tensor.GetShape().At(-1);
    const auto [windowRows, windowCols] = windowSize;
    const auto [rowPadding, colPadding] = padSize;
    const auto [strideRows, strideCols] = stride;

    const auto yRows =
        (inputRows + 2 * rowPadding - (windowRows - 1) - 1) / strideRows + 1;
    const auto yCols =
        (inputCols + 2 * colPadding - (windowCols - 1) - 1) / strideCols + 1;
    if (yRows <= 0 || yCols <= 0)
        throw std::invalid_argument(
            "F:MaxPool2D - invalid argument (could not derive size of "
            "y)");

    const auto channels = tensor.GetShape().At(-3);
    auto device = tensor.GetDevice();
    auto& xDesc = model.GetDescriptor(tensor.TensorDescriptorKey());
    const Shape xShape = xDesc.GetShape();

    //! Check condition of X
    if (xShape.Dim() < 4)
        throw std::invalid_argument(
            "NN::MaxPool2D - input should have shape of (*, C, H, W)");
    if (xShape.At(-3) != channels)
        throw std::invalid_argument(
            "NN::MaxPool2D - size of x channels does not match ");

    Shape yShape = xShape;
    yShape[-1] = yCols;
    yShape[-2] = yRows;
    yShape[-3] = channels;
    const auto yKey =
        model.RegisterTensorDescriptor(yShape, xDesc.GetType(), xDesc.GetDevice());

    auto& yDesc = model.GetDescriptor(yKey);
    yDesc.SetMode(mode);
    
    auto x = xDesc.GetForwardData();
    auto dx = xDesc.GetBackwardData();
    auto y = yDesc.GetForwardData();
    auto dy = yDesc.GetBackwardData();

    Util::ChangeTensorDataDimension(4, x, dx, y, dy);

    Compute::MaxPool2DForward(y, x, windowRows, windowCols, strideRows,
                              strideCols, rowPadding, colPadding);

    auto* backPropWrapper = new BackProp::MaxPool2DBackProp(
        dx, dy, x, y, windowSize, stride, padSize);

    Util::SaveHistory(backPropWrapper, std::make_tuple(&xDesc),
                      std::make_tuple(&yDesc));

    return Tensor(yKey);
}
}
