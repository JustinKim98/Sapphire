// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/dense/cuda/CudnnStruct.cuh>
#include <tuple>

namespace Sapphire::Compute::Dense::Cuda
{
bool Shape4D::operator==(const Shape4D& shape4D) const
{
    return std::tie(N, Channels, Height, Width) ==
           std::tie(shape4D.N, shape4D.Channels, shape4D.Height, shape4D.Width);
}

bool Shape4D::operator!=(const Shape4D& shape4D) const
{
    return !(*this == shape4D);
}

bool ConvConfig::operator==(const ConvConfig& convConfig) const
{
    return std::tie(InputShape, FilterShape, StrideRow, StrideCol, DilationRow,
                    DilationCol, RowPadding, ColumnPadding) ==
           std::tie(convConfig.InputShape, convConfig.FilterShape,
                    convConfig.StrideRow, convConfig.StrideCol,
                    convConfig.DilationRow, convConfig.DilationCol,
                    convConfig.RowPadding, convConfig.ColumnPadding);
}

bool ConvConfig::operator!=(const ConvConfig& convConfig) const
{
    return !(*this == convConfig);
}

bool PoolConfig::operator==(const PoolConfig& poolConfig) const
{
    return std::tie(InputShape, WindowHeight, WindowWidth, StrideRow, StrideCol,
                    RowPadding, ColumnPadding) ==
           std::tie(poolConfig.InputShape, poolConfig.WindowHeight,
                    poolConfig.WindowWidth, poolConfig.StrideRow,
                    poolConfig.StrideCol, poolConfig.RowPadding,
                    poolConfig.ColumnPadding);
}

bool PoolConfig::operator!=(const PoolConfig& poolConfig) const
{
    return !(*this == poolConfig);
}


bool CudnnConv2DMetaData::operator==(
    const CudnnConv2DMetaData& conv2DMetaData) const
{
    return
        this->ConvDesc == conv2DMetaData.ConvDesc &&
        this->InputDesc == conv2DMetaData.InputDesc &&
        this->FilterDesc == conv2DMetaData.FilterDesc &&
        this->OutputDesc == conv2DMetaData.OutputDesc &&
        this->ForwardWorkSpace == conv2DMetaData.ForwardWorkSpace &&
        this->ForwardWorkSpaceBytes == conv2DMetaData.ForwardWorkSpaceBytes &&
        this->BackwardDataWorkSpace == conv2DMetaData.BackwardDataWorkSpace &&
        this->BackwardDataWorkSpaceBytes ==
        conv2DMetaData.BackwardDataWorkSpaceBytes &&
        this->BackwardFilterWorkSpace ==
        conv2DMetaData.BackwardFilterWorkSpace &&
        this->BackwardFilterWorkSpaceBytes ==
        conv2DMetaData.BackwardFilterWorkSpaceBytes;
}

bool CudnnConv2DMetaData::operator!=(
    const CudnnConv2DMetaData& conv2DMetaData) const
{
    return !(*this == conv2DMetaData);
}

bool CudnnPool2DMetaData::operator==(
    const CudnnPool2DMetaData& pool2DMetaData) const
{
    return std::tie(PoolDesc, xDesc, yDesc, dxDesc, dyDesc) ==
           std::tie(pool2DMetaData.PoolDesc, pool2DMetaData.xDesc,
                    pool2DMetaData.yDesc,
                    pool2DMetaData.dxDesc, pool2DMetaData.dyDesc);
}

bool CudnnPool2DMetaData::operator!=(
    const CudnnPool2DMetaData& pool2DMetaData) const
{
    return !(*this == pool2DMetaData);
}
}
