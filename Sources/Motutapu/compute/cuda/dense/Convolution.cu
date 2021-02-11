// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/cuda/dense/Convolution.cuh>

namespace Motutapu::Compute::Cuda
{
__host__ void CreateConvDescriptors(CudnnConvMetaData* metadata,
                                    Shape4D inputShape, Shape4D filterShape,
                                    int strideRow, int strideCol,
                                    int dilationRow, int dilationCol,
                                    int paddingRow, int paddingCol)
{
    int outputN = inputShape.N;
    int outputChannels = filterShape.N;
    int outputHeight = (inputShape.Height + 2 * paddingRow -
                        dilationRow * (filterShape.Height - 1) - 1) /
                           strideRow +
                       1;
    int outputWidth = (inputShape.Width + 2 * paddingCol -
                       dilationCol * (filterShape.Width - 1) - 1) /
                          strideCol +
                      1;

    checkCuDNN(cudnnCreate(&metadata->Handle));

    checkCuDNN(cudnnCreateConvolutionDescriptor(&metadata->ConvDesc));
    checkCuDNN(cudnnCreateTensorDescriptor(&metadata->InputDesc));
    checkCuDNN(cudnnCreateFilterDescriptor(&metadata->FilterDesc));
    checkCuDNN(cudnnCreateTensorDescriptor(&metadata->OutputDesc));

    checkCuDNN(cudnnSetConvolution2dDescriptor(
        metadata->ConvDesc, paddingRow, paddingCol, strideRow, strideCol,
        dilationRow, dilationCol, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    checkCuDNN(cudnnSetTensor4dDescriptor(
        metadata->InputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, inputShape.N,
        inputShape.Channels, inputShape.Height, inputShape.Width));

    checkCuDNN(cudnnSetFilter4dDescriptor(
        metadata->FilterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        filterShape.N, filterShape.Channels, filterShape.Height,
        filterShape.Width));

    checkCuDNN(cudnnSetTensor4dDescriptor(
        metadata->OutputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, outputN,
        outputChannels, outputHeight, outputWidth));

    int numAlgo;

    cudnnConvolutionFwdAlgoPerf_t forwardPerf[3];
    checkCuDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        metadata->Handle, metadata->InputDesc, metadata->FilterDesc,
        metadata->ConvDesc, metadata->OutputDesc, 1, &numAlgo, forwardPerf));
    metadata->ForwardAlgo = forwardPerf[0].algo;

    cudnnConvolutionBwdDataAlgoPerf_t backDataPerf[3];
    checkCuDNN(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        metadata->Handle, metadata->FilterDesc, metadata->OutputDesc,
        metadata->ConvDesc, metadata->InputDesc, 3, &numAlgo, backDataPerf));
    metadata->BackwardDataAlgo = backDataPerf[0].algo;

    cudnnConvolutionBwdFilterAlgoPerf_t backFilterPerf[3];
    checkCuDNN(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        metadata->Handle, metadata->InputDesc, metadata->OutputDesc,
        metadata->ConvDesc, metadata->FilterDesc, 3, &numAlgo, backFilterPerf));
    metadata->BackwardFilterAlgo = backFilterPerf[0].algo;

    checkCuDNN(cudnnGetConvolutionForwardWorkspaceSize(
        metadata->Handle, metadata->InputDesc, metadata->FilterDesc,
        metadata->ConvDesc, metadata->OutputDesc, metadata->ForwardAlgo,
        &metadata->ForwardWorkSpaceBytes));

    checkCuDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
        metadata->Handle, metadata->FilterDesc, metadata->OutputDesc,
        metadata->ConvDesc, metadata->InputDesc, metadata->BackwardDataAlgo,
        &metadata->BackwardDataWorkSpaceBytes));

    checkCuDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        metadata->Handle, metadata->InputDesc, metadata->OutputDesc,
        metadata->ConvDesc, metadata->FilterDesc, metadata->BackwardFilterAlgo,
        &metadata->BackwardFilterWorkSpaceBytes));

    checkCuda(cudaMalloc((void**)&metadata->ForwardWorkSpace,
                         metadata->ForwardWorkSpaceBytes));

    checkCuda(cudaMalloc((void**)&metadata->BackwardDataWorkSpace,
                         metadata->BackwardDataWorkSpaceBytes));

    checkCuda(cudaMalloc((void**)&metadata->BackwardFilterWorkSpace,
                         metadata->BackwardFilterWorkSpaceBytes));
}

__host__ void ConvolutionForward2D(CudnnConvMetaData* metadata, float* output,
                                   float* input, float* filter,
                                   Shape4D inputShape, Shape4D filterShape,
                                   int strideRow, int strideCol,
                                   int dilationRow, int dilationCol,
                                   int paddingRow, int paddingCol)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    checkCuDNN(cudnnConvolutionForward(
        metadata->Handle, &alpha, metadata->InputDesc, input,
        metadata->FilterDesc, filter, metadata->ConvDesc, metadata->ForwardAlgo,
        metadata->ForwardWorkSpace, metadata->ForwardWorkSpaceBytes, &beta,
        metadata->OutputDesc, output));
}

__host__ void ConvolutionBackward2D(
    CudnnConvMetaData* descriptors, float* dataGradientOut, float* filter,
    float* filterGradientOut, float* input, float* gradientInput,
    Shape4D inputShape, Shape4D filterShape, int strideRow, int strideCol,
    int dilationRow, int dilationCol, int paddingRow, int paddingCol)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    checkCuDNN(cudnnConvolutionBackwardData(
        descriptors->Handle, &alpha, descriptors->FilterDesc, filter,
        descriptors->OutputDesc, gradientInput, descriptors->ConvDesc,
        descriptors->BackwardDataAlgo, descriptors->BackwardDataWorkSpace,
        descriptors->BackwardDataWorkSpaceBytes, &beta, descriptors->InputDesc,
        dataGradientOut));

    checkCuDNN(cudnnConvolutionBackwardFilter(
        descriptors->Handle, &alpha, descriptors->InputDesc, input,
        descriptors->OutputDesc, gradientInput, descriptors->ConvDesc,
        descriptors->BackwardFilterAlgo, descriptors->BackwardFilterWorkSpace,
        descriptors->BackwardFilterWorkSpaceBytes, &beta,
        descriptors->FilterDesc, filterGradientOut));
}
}  // namespace Motutapu::Compute::Cuda