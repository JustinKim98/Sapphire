// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/ConvolutionOps.hpp>
#include <Sapphire/compute/dense/cuda/Pool.cuh>
#include <Sapphire/compute/dense/cuda/Convolution.cuh>

namespace Sapphire::Compute
{
void Conv2DForward(TensorData& y, const TensorData& x, const TensorData& filter,
                   int strideRow, int strideCol, int dilationRow,
                   int dilationCol, int rowPadding, int columnPadding)
{
    const auto device = y.GetCudaDevice();
    if (y.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        const Dense::Cuda::Shape4D filterShape = {
            static_cast<int>(filter.GetBatchSize(3)),
            static_cast<int>(filter.GetShape().At(2)),
            static_cast<int>(filter.GetShape().Rows()),
            static_cast<int>(filter.GetShape().Cols())
        };

        const Dense::Cuda::Shape4D xShape = {
            static_cast<int>(x.GetBatchSize(3)),
            static_cast<int>(x.GetShape().At(2)),
            static_cast<int>(x.GetShape().Rows()),
            static_cast<int>(x.GetShape().Cols())
        };

        Dense::Cuda::Conv2DForward(
            y.GetMutableDenseCuda(), x.GetDenseCuda(), filter.GetDenseCuda(),
            xShape, filterShape, strideRow, strideCol, dilationRow, dilationCol,
            rowPadding, columnPadding, device.GetID());
    }
    else
    {
        throw std::invalid_argument(
            "Compute::Conv2DForward - Host mode Not implemented");
    }
}

void MaxPool2DForward(TensorData& y, const TensorData& x, int windowHeight,
                      int windowWidth, int strideRow, int strideCol,
                      int rowPadding, int columnPadding)
{
    const auto device = y.GetCudaDevice();
    if (y.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        const Dense::Cuda::Shape4D xShape = {
            static_cast<int>(x.GetBatchSize(3)),
            static_cast<int>(x.GetShape().At(2)),
            static_cast<int>(x.GetShape().Rows()),
            static_cast<int>(x.GetShape().Cols())
        };

        Dense::Cuda::Pool2DForward(
            y.GetMutableDenseCuda(), x.GetDenseCuda(), xShape, windowHeight,
            windowWidth, strideRow, strideCol, rowPadding, columnPadding,
            Dense::Cuda::PoolingMode::Max, CUDNN_PROPAGATE_NAN, device.GetID());
    }
    else
    {
        throw std::invalid_argument(
            "Compute::Conv2DForward - Host mode Not implemented");
    }
}

void AvgPool2DForward(TensorData& y, const TensorData& x, int windowHeight,
                      int windowWidth, int strideRow, int strideCol,
                      int rowPadding, int columnPadding)
{
    const auto device = y.GetCudaDevice();
    if (y.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        const Dense::Cuda::Shape4D xShape = {
            static_cast<int>(x.GetBatchSize(3)),
            static_cast<int>(x.GetShape().At(2)),
            static_cast<int>(x.GetShape().Rows()),
            static_cast<int>(x.GetShape().Cols())
        };

        Dense::Cuda::Pool2DForward(
            y.GetMutableDenseCuda(), x.GetDenseCuda(), xShape, windowHeight,
            windowWidth, strideRow, strideCol, rowPadding, columnPadding,
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
                    int strideRow, int strideCol, int dilationRow,
                    int dilationCol, int rowPadding, int columnPadding)
{
    const auto device = dx.GetCudaDevice();
    if (dx.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        const Dense::Cuda::Shape4D filterShape = {
            static_cast<int>(filter.GetBatchSize(3)),
            static_cast<int>(filter.GetShape().At(2)),
            static_cast<int>(filter.GetShape().Rows()),
            static_cast<int>(filter.GetShape().Cols())
        };

        const Dense::Cuda::Shape4D xShape = {
            static_cast<int>(x.GetBatchSize(3)),
            static_cast<int>(x.GetShape().At(2)),
            static_cast<int>(x.GetShape().Rows()),
            static_cast<int>(x.GetShape().Cols())
        };

        Dense::Cuda::Conv2DBackward(
            dx.GetMutableDenseCuda(), filter.GetDenseCuda(),
            dFilter.GetMutableDenseCuda(),
            x.GetDenseCuda(), dy.GetDenseCuda(), xShape, filterShape, strideRow,
            strideCol, dilationRow, dilationCol, rowPadding, columnPadding,
            device.GetID());
    }
    else
    {
        throw std::invalid_argument(
            "Compute::Conv2DBackward - Host mode not implemented");
    }
}

void MaxPool2DBackward(TensorData& dx, const TensorData& dy,
                       const TensorData& x,
                       const TensorData& y, int windowHeight, int windowWidth,
                       int strideRow, int strideCol, int rowPadding,
                       int columnPadding)
{
    const auto device = dx.GetCudaDevice();
    if (dx.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        const Dense::Cuda::Shape4D xShape = {
            static_cast<int>(x.GetBatchSize(3)),
            static_cast<int>(x.GetShape().At(2)),
            static_cast<int>(x.GetShape().Rows()),
            static_cast<int>(x.GetShape().Cols())
        };

        Dense::Cuda::Pool2DBackward(
            y.GetDenseCuda(), dy.GetDenseCuda(), x.GetDenseCuda(),
            dx.GetMutableDenseCuda(),
            xShape, windowHeight, windowWidth, strideRow, strideCol, rowPadding,
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
                       const TensorData& y, int windowHeight, int windowWidth,
                       int strideRow, int strideCol, int rowPadding,
                       int columnPadding)
{
    const auto device = dx.GetCudaDevice();
    if (dx.Mode() == DeviceType::Cuda)
    {
        cudaSetDevice(device.GetID());
        const Dense::Cuda::Shape4D xShape = {
            static_cast<int>(x.GetBatchSize(3)),
            static_cast<int>(x.GetShape().At(2)),
            static_cast<int>(x.GetShape().Rows()),
            static_cast<int>(x.GetShape().Cols())
        };

        Dense::Cuda::Pool2DBackward(
            y.GetDenseCuda(), dy.GetDenseCuda(), x.GetDenseCuda(),
            dx.GetMutableDenseCuda(),
            xShape, windowHeight, windowWidth, strideRow, strideCol, rowPadding,
            columnPadding, Dense::Cuda::PoolingMode::Avg, device.GetID());
    }
    else
    {
        throw std::invalid_argument(
            "Compute::MaxPool2DBackward - Host mode Not implemented");
    }
}
}
