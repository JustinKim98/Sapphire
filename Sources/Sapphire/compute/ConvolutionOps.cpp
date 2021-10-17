// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/ConvolutionOps.hpp>
#include <Sapphire/compute/dense/cuda/Pool.cuh>
#include <Sapphire/compute/dense/cuda/Convolution.cuh>
#include <Sapphire/compute/dense/naive/Conv2D.hpp>


namespace Sapphire::Compute
{
void Conv2DForward(TensorData& y, const TensorData& x, const TensorData& filter,
                   int strideRow, int strideCol, int dilationRow,
                   int dilationCol, int rowPadding, int columnPadding)
{
    assert(y.Mode() == x.Mode() && y.Mode() == filter.Mode());
    assert(y.GetDevice() == x.GetDevice() &&
        y.GetDevice() == filter.GetDevice());

    const auto device = y.GetDevice();
    if (y.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        const Dense::Cuda::Shape4D filterShape = {
            static_cast<int>(filter.GetBatchSize(3)),
            static_cast<int>(filter.GetShape().At(
                filter.GetShape().Dim() - 3)),
            static_cast<int>(filter.GetShape().Rows()),
            static_cast<int>(filter.GetShape().Cols())
        };

        const Dense::Cuda::Shape4D xShape = {
            static_cast<int>(x.GetBatchSize(3)),
            static_cast<int>(x.GetShape().At(x.GetShape().Dim() - 3)),
            static_cast<int>(x.GetShape().Rows()),
            static_cast<int>(x.GetShape().Cols())
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
                      int rowPadding, int columnPadding)
{
    assert(y.Mode() == x.Mode());
    assert(y.GetDevice() == x.GetDevice());

    const auto device = y.GetDevice();
    if (y.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        const Dense::Cuda::Shape4D xShape = {
            static_cast<int>(x.GetBatchSize(3)),
            static_cast<int>(x.GetShape().At(x.GetShape().Dim() - 3)),
            static_cast<int>(x.GetShape().Rows()),
            static_cast<int>(x.GetShape().Cols())
        };

        Dense::Cuda::Pool2DForward(
            y.CudaMutableRawPtr(), x.CudaRawPtr(), xShape, windowRows,
            windowCols, strideRow, strideCol, rowPadding, columnPadding,
            Dense::Cuda::PoolingMode::Max, CUDNN_PROPAGATE_NAN, device.GetID());
    }
    else
    {
        throw std::invalid_argument(
            "Compute::Conv2DForward - Host mode Not implemented");
    }
}

void AvgPool2DForward(TensorData& y, const TensorData& x, int windowRows,
                      int windowCols, int strideRow, int strideCol,
                      int rowPadding, int columnPadding)
{
    assert(y.Mode() == x.Mode());
    assert(y.GetDevice() == x.GetDevice());
    const auto device = y.GetDevice();
    if (y.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        const Dense::Cuda::Shape4D xShape = {
            static_cast<int>(x.GetBatchSize(3)),
            static_cast<int>(x.GetShape().At(x.GetShape().Dim() - 3)),
            static_cast<int>(x.GetShape().Rows()),
            static_cast<int>(x.GetShape().Cols())
        };

        Dense::Cuda::Pool2DForward(
            y.CudaMutableRawPtr(), x.CudaRawPtr(), xShape, windowRows,
            windowCols, strideRow, strideCol, rowPadding, columnPadding,
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
    assert(dy.GetDevice() == x.GetDevice() &&
        dy.GetDevice() == filter.GetDevice());
    assert(dy.GetDevice() == dx.GetDevice() &&
        dy.GetDevice() == dFilter.GetDevice());
    const auto device = dx.GetDevice();
    if (dx.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        const Dense::Cuda::Shape4D filterShape = {
            static_cast<int>(filter.GetBatchSize(3)),
            static_cast<int>(filter.GetShape().At(x.GetShape().Dim() - 3)),
            static_cast<int>(filter.GetShape().Rows()),
            static_cast<int>(filter.GetShape().Cols())
        };

        const Dense::Cuda::Shape4D xShape = {
            static_cast<int>(x.GetBatchSize(3)),
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
                       int columnPadding)
{
    assert(dx.Mode() == dy.Mode() && dx.Mode() == x.Mode() &&
        dx.Mode() == y.Mode());
    assert(dx.GetDevice() == dy.GetDevice() &&
        dx.GetDevice() == x.GetDevice() &&
        dx.GetDevice() == y.GetDevice());

    const auto device = dx.GetDevice();
    if (dx.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        const Dense::Cuda::Shape4D xShape = {
            static_cast<int>(x.GetBatchSize(3)),
            static_cast<int>(x.GetShape().At(x.GetShape().Dim() - 3)),
            static_cast<int>(x.GetShape().Rows()),
            static_cast<int>(x.GetShape().Cols())
        };

        Dense::Cuda::Pool2DBackward(
            y.CudaRawPtr(), dy.CudaRawPtr(), x.CudaRawPtr(),
            dx.CudaMutableRawPtr(),
            xShape, windowRows, windowCols, strideRow, strideCol, rowPadding,
            columnPadding, Dense::Cuda::PoolingMode::Max, device.GetID());
    }
    else
    {
        throw std::invalid_argument(
            "Compute::MaxPool2DBackward - Host mode Not implemented");
    }
}

void AvgPool2DBackward(TensorData& dx, const TensorData& dy,
                       const TensorData& x,
                       const TensorData& y, int windowRows, int windowCols,
                       int strideRow, int strideCol, int rowPadding,
                       int columnPadding)
{
    assert(
        dx.Mode() == dy.Mode() && dx.Mode() == x.Mode() && dx.Mode() == y.Mode(
        ));
    assert(dx.GetDevice() == dy.GetDevice() &&
        dx.GetDevice() == x.GetDevice() &&
        dx.GetDevice() == y.GetDevice());

    const auto device = dx.GetDevice();
    if (dx.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        const Dense::Cuda::Shape4D xShape = {
            static_cast<int>(x.GetBatchSize(3)),
            static_cast<int>(x.GetShape().At(x.GetShape().Dim() - 3)),
            static_cast<int>(x.GetShape().Rows()),
            static_cast<int>(x.GetShape().Cols())
        };

        Dense::Cuda::Pool2DBackward(
            y.CudaRawPtr(), dy.CudaRawPtr(), x.CudaRawPtr(),
            dx.CudaMutableRawPtr(),
            xShape, windowRows, windowCols, strideRow, strideCol, rowPadding,
            columnPadding, Dense::Cuda::PoolingMode::Avg, device.GetID());
    }
    else
    {
        throw std::invalid_argument(
            "Compute::MaxPool2DBackward - Host mode Not implemented");
    }
}
}
