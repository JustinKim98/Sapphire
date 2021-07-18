// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/ComputeBackward.hpp>
#include <Sapphire/compute/dense/cuda/Convolution.cuh>
#include <Sapphire/compute/dense/cuda/Pool.cuh>
#include <Sapphire/compute/Broadcast.hpp>
#include <Sapphire/compute/dense/cuda/BasicBackward.cuh>

namespace Sapphire::Compute
{
void Conv2DBackward(TensorData& dx, TensorData& dFilter, const TensorData& dy,
                    const TensorData& x, const TensorData& filter,
                    int strideRow, int strideCol, int dilationRow,
                    int dilationCol, int rowPadding, int columnPadding)
{
    if (const auto device = dx.GetDevice(); device.Type() == DeviceType::CUDA)
    {
        const Dense::Cuda::Shape4D filterShape = {
            static_cast<int>(filter.BatchSize),
            static_cast<int>(filter.GetShape().At(2)),
            static_cast<int>(filter.GetShape().Rows()),
            static_cast<int>(filter.GetShape().Cols()) };

        const Dense::Cuda::Shape4D xShape = {
            static_cast<int>(x.BatchSize), static_cast<int>(x.GetShape().At(2)),
            static_cast<int>(x.GetShape().Rows()),
            static_cast<int>(x.GetShape().Cols())
        };

        Dense::Cuda::Conv2DBackward(
            dx.DenseMatCuda, filter.DenseMatCuda, dFilter.DenseMatCuda,
            x.DenseMatCuda, dy.DenseMatCuda, xShape, filterShape, strideRow,
            strideCol, dilationRow, dilationCol, rowPadding, columnPadding,
            device.GetID());
    }
    else
    {
        throw std::invalid_argument(
            "Compute::Conv2DBackward - Host mode not implemented");
    }
}

void MaxPool2DBackward(TensorData& dy, TensorData& dx, const TensorData& x,
                       const TensorData& y, int windowHeight, int windowWidth,
                       int strideRow, int strideCol, int rowPadding,
                       int columnPadding)
{
    if (const auto device = y.GetDevice(); device.Type() == DeviceType::CUDA)
    {
        const Dense::Cuda::Shape4D xShape = {
            static_cast<int>(x.BatchSize), static_cast<int>(x.GetShape().At(2)),
            static_cast<int>(x.GetShape().Rows()),
            static_cast<int>(x.GetShape().Cols())
        };

        Dense::Cuda::Pool2DBackward(
            y.DenseMatCuda, dy.DenseMatCuda, x.DenseMatCuda, dx.DenseMatCuda,
            xShape, windowHeight, windowWidth,
            strideRow, strideCol, rowPadding, columnPadding,
            Dense::Cuda::PoolingMode::Max, device.GetID());
    }
    else
    {
        throw std::invalid_argument(
            "Compute::MaxPool2DBackward - Host mode Not implemented");
    }
}

void AvgPool2DBackward(TensorData& dy, TensorData& dx, const TensorData& x,
                       const TensorData& y, int windowHeight, int windowWidth,
                       int strideRow, int strideCol, int rowPadding,
                       int columnPadding)
{
    if (const auto device = y.GetDevice(); device.Type() == DeviceType::CUDA)
    {
        const Dense::Cuda::Shape4D xShape = {
            static_cast<int>(x.BatchSize), static_cast<int>(x.GetShape().At(2)),
            static_cast<int>(x.GetShape().Rows()),
            static_cast<int>(x.GetShape().Cols())
        };

        Dense::Cuda::Pool2DBackward(
            y.DenseMatCuda, dy.DenseMatCuda, x.DenseMatCuda, dx.DenseMatCuda,
            xShape, windowHeight, windowWidth, strideRow, strideCol, rowPadding,
            columnPadding, Dense::Cuda::PoolingMode::Avg, device.GetID());
    }
    else
    {
        throw std::invalid_argument(
            "Compute::MaxPool2DBackward - Host mode Not implemented");
    }
}

void DotBackward(TensorData& da, TensorData& db, const TensorData& dy,
                 const TensorData& a, const TensorData& b)
{
    const auto device = dy.GetDevice();

    if (dy.TensorShape == a.TensorShape && dy.TensorShape == b.TensorShape &&
        dy.BatchSize > 1)
    {
        if (device.Type() == DeviceType::CUDA)
        {
            return;
        }
        if (device.Type() == DeviceType::HOST)
        {
            return;
        }
    }

    const auto maxDim = std::max(
        { dy.TensorShape.Dim(), da.TensorShape.Dim(), db.TensorShape.Dim() });

    auto shapeOut = dy.TensorShape;
    auto shapeA = a.TensorShape;
    auto shapeB = b.TensorShape;

    shapeOut.Expand(maxDim + 1);
    shapeA.Expand(maxDim + 1);
    shapeB.Expand(maxDim + 1);

    shapeOut.Set(0, dy.BatchSize);
    shapeA.Set(0, a.BatchSize);
    shapeB.Set(0, b.BatchSize);

    const auto sizeOut = shapeOut.Size();
    const auto sizeA = shapeA.Size();
    const auto sizeB = shapeB.Size();

    if (device.Type() == DeviceType::CUDA)
    {
        BroadcastBackwardWith2Inputs(shapeOut, shapeA, shapeB, sizeOut, sizeA,
                                     sizeB,
                                     dy.DenseMatCuda, da.DenseMatCuda,
                                     db.DenseMatCuda, a.DenseMatCuda,
                                     b.DenseMatCuda, 0,
                                     0, Dense::Cuda::DotBackward, 0, false,
                                     false);
    }
    else
    {
        throw std::runtime_error("Compute::DotBackward - Host not implemented");
    }
}
}
