// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Sapphire_COMPUTE_CUDA_CONVOLUTION_CUH
#define Sapphire_COMPUTE_CUDA_CONVOLUTION_CUH

#include <cudnn.h>
#include <Sapphire/compute/cudaUtil/CudaParams.cuh>
#include <cassert>

namespace Sapphire::Compute::Dense::Cuda
{
struct Shape4D
{
    int N;         // N
    int Channels;  // C
    int Height;    // H
    int Width;     // W
};

struct CudnnMetaData
{
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

__host__ void checkCuDNN(cudnnStatus_t status)
{
    assert(status == CUDNN_STATUS_SUCCESS);
}

__host__ void checkCuda(cudaError_t status)
{
    assert(status == cudaSuccess);
}

__host__ void CreateConvDescriptors(CudnnMetaData* metadata, Shape4D inputShape,
                                    Shape4D filterShape, int strideRow,
                                    int strideCol, int dilationRow,
                                    int dilationCol, int paddingRow,
                                    int paddingCol);

__host__ void ConvolutionForward2D(CudnnMetaData* metadata, float* output,
                                   float* input, float* filter,
                                   Shape4D inputShape, Shape4D filterShape,
                                   int strideRow, int strideCol,
                                   int dilationRow, int dilationCol,
                                   int paddingRow, int paddingCol);

__host__ void ConvolutionBackward2D(
    CudnnMetaData* descriptors, float* dataGradientOut, float* filter,
    float* filterGradientOut, float* input, float* gradientInput,
    Shape4D inputShape, Shape4D filterShape, int strideRow, int strideCol,
    int dilationRow, int dilationCol, int paddingRow, int paddingCol);

}  // namespace Sapphire::Compute::Cuda

#endif  // Sapphire_CONVOLUTION_CUH
