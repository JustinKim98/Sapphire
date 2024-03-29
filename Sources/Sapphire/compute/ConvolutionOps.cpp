// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/ConvolutionOps.hpp>
#include <Sapphire/compute/dense/cuda/Pool.cuh>
#include <Sapphire/compute/dense/naive/Pool.hpp>
#include <Sapphire/compute/dense/cuda/Convolution.cuh>
#include <Sapphire/compute/dense/naive/Convolution.hpp>


namespace Sapphire::Compute
{
void Conv2DForward(TensorData& y, const TensorData& x, const TensorData& filter,
                   int strideRow, int strideCol, int dilationRow,
                   int dilationCol, int rowPadding, int columnPadding)
{
    assert(y.Mode() == x.Mode() && y.Mode() == filter.Mode());

    const auto device = y.GetCudaDevice();
    if (y.Mode() == ComputeMode::Cuda)
    {
        const Dense::Cuda::Shape4D filterShape = {
            filter.GetNumUnits(3),
            filter.GetShape().At(
                filter.GetShape().Dim() - 3),
            filter.GetShape().Rows(),
            filter.GetShape().Cols()
        };

        const Dense::Cuda::Shape4D xShape = {
            x.GetNumUnits(3),
            x.GetShape().At(x.GetShape().Dim() - 3),
            x.GetShape().Rows(),
            x.GetShape().Cols()
        };

        Dense::Cuda::Conv2DForward(
            y.CudaMutableRawPtr(), x.CudaRawPtr(), filter.CudaRawPtr(),
            xShape, filterShape, strideRow, strideCol, dilationRow, dilationCol,
            rowPadding, columnPadding, device.GetID());
    }
    else
    {
        Dense::Naive::Conv2D(y, x, filter, strideRow, strideCol, rowPadding,
                             columnPadding, dilationRow, dilationCol, device);
    }
}

void MaxPool2DForward(TensorData& y, const TensorData& x, int windowRows,
                      int windowCols, int strideRow, int strideCol,
                      int rowPadding, int colPadding)
{
    assert(y.Mode() == x.Mode());

    const auto device = y.GetCudaDevice();
    if (y.Mode() == ComputeMode::Cuda)
    {
        const Dense::Cuda::Shape4D xShape = {
            static_cast<int>(x.GetNumUnits(3)),
            static_cast<int>(x.GetShape().At(x.GetShape().Dim() - 3)),
            static_cast<int>(x.GetShape().Rows()),
            static_cast<int>(x.GetShape().Cols())
        };

        Dense::Cuda::Pool2DForward(
            y.CudaMutableRawPtr(), x.CudaRawPtr(), xShape, windowRows,
            windowCols, strideRow, strideCol, rowPadding, colPadding,
            Dense::Cuda::PoolingMode::Max, CUDNN_PROPAGATE_NAN, device.GetID());
    }
    else
    {
        Dense::Naive::MaxPool2D(y, x, std::make_pair(windowRows, windowCols),
                                std::make_pair(strideRow, strideCol),
                                std::make_pair(rowPadding, colPadding),
                                std::make_pair(1, 1));
    }
}

void AvgPool2DForward(TensorData& y, const TensorData& x, int windowRows,
                      int windowCols, int strideRow, int strideCol,
                      int rowPadding, int colPadding)
{
    assert(y.Mode() == x.Mode());
    const auto device = y.GetCudaDevice();
    if (y.Mode() == ComputeMode::Cuda)
    {
        const Dense::Cuda::Shape4D xShape = {
            static_cast<int>(x.GetNumUnits(3)),
            static_cast<int>(x.GetShape().At(x.GetShape().Dim() - 3)),
            static_cast<int>(x.GetShape().Rows()),
            static_cast<int>(x.GetShape().Cols())
        };

        Dense::Cuda::Pool2DForward(
            y.CudaMutableRawPtr(), x.CudaRawPtr(), xShape, windowRows,
            windowCols, strideRow, strideCol, rowPadding, colPadding,
            Dense::Cuda::PoolingMode::Avg, CUDNN_PROPAGATE_NAN, device.GetID());
    }
    else
    {
        throw std::invalid_argument(
            "Compute::Conv2DForward - Host mode Not implemented");
    }
}

void Conv2DBackward(TensorData& dx, TensorData& dFilter, const TensorData& dy,
                    const TensorData& x, const TensorData& filter,
                    int strideRow, int strideCol, int rowPadding,
                    int colPadding, int dilationRow, int dilationCol)
{
    assert(dy.Mode() == dx.Mode() && dy.Mode() == dFilter.Mode());
    assert(dy.Mode() == x.Mode() && dy.Mode() == filter.Mode());
    const auto device = dx.GetCudaDevice();
    if (dx.Mode() == ComputeMode::Cuda)
    {
        const Dense::Cuda::Shape4D filterShape = {
            static_cast<int>(filter.GetNumUnits(3)),
            static_cast<int>(filter.GetShape().At(x.GetShape().Dim() - 3)),
            static_cast<int>(filter.GetShape().Rows()),
            static_cast<int>(filter.GetShape().Cols())
        };

        const Dense::Cuda::Shape4D xShape = {
            static_cast<int>(x.GetNumUnits(3)),
            static_cast<int>(x.GetShape().At(x.GetShape().Dim() - 3)),
            static_cast<int>(x.GetShape().Rows()),
            static_cast<int>(x.GetShape().Cols())
        };

        Dense::Cuda::Conv2DBackward(
            dx.CudaMutableRawPtr(), filter.CudaRawPtr(),
            dFilter.CudaMutableRawPtr(),
            x.CudaRawPtr(), dy.CudaRawPtr(), xShape, filterShape, strideRow,
            strideCol, dilationRow, dilationCol, rowPadding, colPadding,
            device.GetID());
    }
    else
    {
        Dense::Naive::Conv2DBackward(dx, dFilter, dy, x, filter, strideRow,
                                     strideCol, rowPadding, colPadding,
                                     dilationRow, dilationCol, device);
    }
}

void MaxPool2DBackward(TensorData& dx, const TensorData& dy,
                       const TensorData& x,
                       const TensorData& y, int windowRows, int windowCols,
                       int strideRow, int strideCol, int rowPadding,
                       int colPadding)
{
    assert(dx.Mode() == dy.Mode() && dx.Mode() == x.Mode() &&
        dx.Mode() == y.Mode());

    const auto device = dx.GetCudaDevice();
    if (dx.Mode() == ComputeMode::Cuda)
    {
        const Dense::Cuda::Shape4D xShape = {
            x.GetNumUnits(3),
            x.GetShape().At(x.GetShape().Dim() - 3),
            x.GetShape().Rows(),
            x.GetShape().Cols()
        };

        Dense::Cuda::Pool2DBackward(
            y.CudaRawPtr(), dy.CudaRawPtr(), x.CudaRawPtr(),
            dx.CudaMutableRawPtr(),
            xShape, windowRows, windowCols, strideRow, strideCol, rowPadding,
            colPadding, Dense::Cuda::PoolingMode::Max, device.GetID());
    }
    else
    {
        Dense::Naive::MaxPool2DBackward(
            dx, x, dy, std::make_pair(windowRows, windowCols),
            std::make_pair(strideRow, strideCol),
            std::make_pair(rowPadding, colPadding), std::make_pair(1, 1)

            );
    }
}

void AvgPool2DBackward(TensorData& dx, const TensorData& dy,
                       const TensorData& x,
                       const TensorData& y, int windowRows, int windowCols,
                       int strideRow, int strideCol, int rowPadding,
                       int colPadding)
{
    assert(
        dx.Mode() == dy.Mode() && dx.Mode() == x.Mode() && dx.Mode() == y.Mode(
        ));

    const auto device = dx.GetCudaDevice();
    if (dx.Mode() == ComputeMode::Cuda)
    {
        const Dense::Cuda::Shape4D xShape = {
            static_cast<int>(x.GetNumUnits(3)),
            static_cast<int>(x.GetShape().At(x.GetShape().Dim() - 3)),
            static_cast<int>(x.GetShape().Rows()),
            static_cast<int>(x.GetShape().Cols())
        };

        Dense::Cuda::Pool2DBackward(
            y.CudaRawPtr(), dy.CudaRawPtr(), x.CudaRawPtr(),
            dx.CudaMutableRawPtr(),
            xShape, windowRows, windowCols, strideRow, strideCol, rowPadding,
            colPadding, Dense::Cuda::PoolingMode::Avg, device.GetID());
    }
    else
    {
        throw std::invalid_argument(
            "Compute::MaxPool2DBackward - Host mode Not implemented");
    }
}
}
