// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Sapphire_COMPUTE_CUDA_CONVOLUTION_CUH
#define Sapphire_COMPUTE_CUDA_CONVOLUTION_CUH

#include <cudnn.h>
#include <Sapphire/compute/cudaUtil/CudaParams.cuh>
#include <string>

namespace Sapphire::Compute::Dense::Cuda
{
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
    int PaddedRow;
    int PaddedCol;
};


struct CudnnMetaData
{
    bool operator==(const CudnnMetaData& cudnnMetaData) const;

    bool operator!=(const CudnnMetaData& cudnnMetaData) const;

    cudnnHandle_t Handle;
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

__host__ void CreateCudnnMetaData(CudnnMetaData* metadata, Shape4D inputShape,
                                  Shape4D filterShape, int strideRow,
                                  int strideCol, int dilationRow,
                                  int dilationCol, int paddedRow,
                                  int paddedCol);

__host__ void ConvolutionForward2D(CudnnMetaData* metadata, float* output,
                                   float* input, float* filter);

__host__ void ConvolutionBackward2D(
    CudnnMetaData* descriptors, float* dataGradientOut, float* filter,
    float* filterGradientOut, float* input, float* gradientInput);

//! Generalized function that can be called from outside
__host__ void ConvForward2D(float* output, float* input,
                            float
                            * filter, Shape4D inputShape, Shape4D filterShape,
                            int strideRow, int strideCol, int dilationRow,
                            int dilationCol, int paddedRow, int paddedCol);

__host__ void ConvBackward2D(float* dataGradientOut, float* filter,
                             float* filterGradientOut, float* input,
                             float* gradientInput, Shape4D inputShape,
                             Shape4D filterShape, int strideRow, int strideCol,
                             int dilationRow, int dilationCol, int paddedRow,
                             int paddedCol);
} // namespace Sapphire::Compute::Cuda

#endif  // Sapphire_CONVOLUTION_CUH
