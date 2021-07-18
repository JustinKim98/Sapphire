// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_COMPUTE_DENSE_CUDA_CUDNN_STRUCT_CUH
#define SAPPHIRE_COMPUTE_DENSE_CUDA_CUDNN_STRUCT_CUH

#include <cudnn.h>
#include <stdexcept>
#include <string>

namespace Sapphire::Compute::Dense::Cuda
{
enum class PoolingMode
{
    Max,
    Avg,
};

struct Shape4D
{
    bool operator==(const Shape4D& shape4D) const;
    bool operator!=(const Shape4D& shape4D) const;

    int N;        // N
    int Channels; // C
    int Height;   // H
    int Width;    // W
};

struct ConvConfig
{
    bool operator==(const ConvConfig& convConfig) const;
    bool operator!=(const ConvConfig& convConfig) const;

    Shape4D InputShape;
    Shape4D FilterShape;
    int StrideRow;
    int StrideCol;
    int DilationRow;
    int DilationCol;
    int RowPadding;
    int ColumnPadding;
};

struct PoolConfig
{
    bool operator==(const PoolConfig& poolConfig) const;
    bool operator!=(const PoolConfig& poolConfig) const;

    PoolingMode Mode;
    Shape4D InputShape;
    int WindowHeight;
    int WindowWidth;
    int StrideRow;
    int StrideCol;
    int RowPadding;
    int ColumnPadding;
};

struct CudnnConv2DMetaData
{
    bool operator==(const CudnnConv2DMetaData& conv2DMetaData) const;
    bool operator!=(const CudnnConv2DMetaData& conv2DMetaData) const;

    cudnnConvolutionDescriptor_t ConvDesc;
    cudnnTensorDescriptor_t InputDesc;
    cudnnFilterDescriptor_t FilterDesc;
    cudnnTensorDescriptor_t OutputDesc;

    void* ForwardWorkSpace;
    size_t ForwardWorkSpaceBytes;
    void* BackwardDataWorkSpace;
    size_t BackwardDataWorkSpaceBytes;
    void* BackwardFilterWorkSpace;
    size_t BackwardFilterWorkSpaceBytes;

    cudnnConvolutionFwdAlgo_t ForwardAlgo;
    cudnnConvolutionBwdDataAlgo_t BackwardDataAlgo;
    cudnnConvolutionBwdFilterAlgo_t BackwardFilterAlgo;
};

struct CudnnPool2DMetaData
{
    bool operator==(const CudnnPool2DMetaData& pool2DMetaData) const;
    bool operator!=(const CudnnPool2DMetaData& pool2DMetaData) const;

    cudnnPoolingDescriptor_t PoolDesc;
    cudnnTensorDescriptor_t xDesc;
    cudnnTensorDescriptor_t yDesc;
    cudnnTensorDescriptor_t dxDesc;
    cudnnTensorDescriptor_t dyDesc;
};

__host__ inline void checkCuDNN(cudnnStatus_t status)
{
    if (status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("Check CUDNN failed with " +
                                 std::to_string(status));
    }
}

__host__ inline void checkCuda(cudaError_t status)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error("Check CUDA failed with " +
                                 std::to_string(status));
    }
}
}
#endif
