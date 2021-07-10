// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/dense/cuda/Convolution.cuh>
#include <Sapphire/util/MemoryManager.hpp>

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
                    DilationCol, PaddedRow, PaddedCol) ==
           std::tie(convConfig.InputShape, convConfig.FilterShape,
                    convConfig.StrideRow, convConfig.StrideCol,
                    convConfig.DilationRow, convConfig.DilationCol,
                    convConfig.PaddedRow, convConfig.PaddedCol);
}

bool ConvConfig::operator!=(const ConvConfig& convConfig) const
{
    return !(*this == convConfig);
}

bool CudnnMetaData::operator==(const CudnnMetaData& cudnnMetaData) const
{
    return this->Handle == cudnnMetaData.Handle && this->ConvDesc ==
           cudnnMetaData.ConvDesc &&
           this->InputDesc == cudnnMetaData.InputDesc &&
           this->FilterDesc == cudnnMetaData.FilterDesc &&
           this->OutputDesc == cudnnMetaData.OutputDesc &&
           this->ForwardWorkSpace == cudnnMetaData.ForwardWorkSpace &&
           this->ForwardWorkSpaceBytes == cudnnMetaData.ForwardWorkSpaceBytes &&
           this->BackwardDataWorkSpace == cudnnMetaData.BackwardDataWorkSpace &&
           this->BackwardDataWorkSpaceBytes == cudnnMetaData.
           BackwardDataWorkSpaceBytes &&
           this->BackwardFilterWorkSpace == cudnnMetaData.
           BackwardFilterWorkSpace &&
           this->BackwardFilterWorkSpaceBytes ==
           cudnnMetaData.BackwardFilterWorkSpaceBytes;
}

bool CudnnMetaData::operator!=(const CudnnMetaData& cudnnMetaData) const
{
    return !(*this == cudnnMetaData);
}

__host__ void CreateCudnnMetaData(CudnnMetaData* metadata,
                                  Shape4D inputShape, Shape4D filterShape,
                                  int strideRow, int strideCol,
                                  int dilationRow, int dilationCol,
                                  int paddedRow, int paddedCol)
{
    const int outputN = inputShape.N;
    const int outputChannels = filterShape.N;
    const int outputHeight = (inputShape.Height + 2 * paddedRow -
                              dilationRow * (filterShape.Height - 1) - 1) /
                             strideRow +
                             1;
    const int outputWidth = (inputShape.Width + 2 * paddedCol -
                             dilationCol * (filterShape.Width - 1) - 1) /
                            strideCol +
                            1;

    checkCuDNN(cudnnCreate(&metadata->Handle));
    checkCuDNN(cudnnCreateConvolutionDescriptor(&metadata->ConvDesc));
    checkCuDNN(cudnnCreateTensorDescriptor(&metadata->InputDesc));
    checkCuDNN(cudnnCreateFilterDescriptor(&metadata->FilterDesc));
    checkCuDNN(cudnnCreateTensorDescriptor(&metadata->OutputDesc));

    checkCuDNN(cudnnSetConvolution2dDescriptor(
        metadata->ConvDesc, paddedRow, paddedCol, strideRow, strideCol,
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

    cudnnConvolutionFwdAlgoPerf_t forwardPerf;
    checkCuDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        metadata->Handle, metadata->InputDesc, metadata->FilterDesc,
        metadata->ConvDesc, metadata->OutputDesc, 1, &numAlgo, &forwardPerf));
    metadata->ForwardAlgo = forwardPerf.algo;

    cudnnConvolutionBwdDataAlgoPerf_t backDataPerf;
    checkCuDNN(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        metadata->Handle, metadata->FilterDesc, metadata->OutputDesc,
        metadata->ConvDesc, metadata->InputDesc, 1, &numAlgo, &backDataPerf));
    metadata->BackwardDataAlgo = backDataPerf.algo;

    cudnnConvolutionBwdFilterAlgoPerf_t backFilterPerf;
    checkCuDNN(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        metadata->Handle, metadata->InputDesc, metadata->OutputDesc,
        metadata->ConvDesc, metadata->FilterDesc, 1, &numAlgo,
        &backFilterPerf));
    metadata->BackwardFilterAlgo = backFilterPerf.algo;

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

    checkCuda(cudaMalloc(static_cast<void**>(&metadata->ForwardWorkSpace),
                         metadata->ForwardWorkSpaceBytes));

    checkCuda(cudaMalloc(static_cast<void**>(&metadata->BackwardDataWorkSpace),
                         metadata->BackwardDataWorkSpaceBytes));

    checkCuda(cudaMalloc(
        static_cast<void**>(&metadata->BackwardFilterWorkSpace),
        metadata->BackwardFilterWorkSpaceBytes));
}

__host__ void ConvolutionForward2D(CudnnMetaData* metadata, float* output,
                                   float* input, float* filter)
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
    CudnnMetaData* descriptors, float* dataGradientOut, float* filter,
    float* filterGradientOut, float* input, float* gradientInput)
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

__host__ void ConvForward2D(float* output, float* input,
                            float
                            * filter, Shape4D inputShape, Shape4D filterShape,
                            int strideRow, int strideCol, int dilationRow,
                            int dilationCol, int paddedRow, int paddedCol)
{
    const ConvConfig convConfig = { inputShape, filterShape, strideRow,
                                    strideCol,
                                    dilationRow, dilationCol, paddedRow,
                                    paddedCol };

    if (!Util::MemoryManager::HasConvConfig(convConfig))
    {
        auto* metaData = new CudnnMetaData();
        CreateCudnnMetaData(metaData, inputShape, filterShape, strideRow,
                            strideCol, dilationRow, dilationCol, paddedRow,
                            paddedCol);
        Util::MemoryManager::AddCudnnMetaData(convConfig, metaData);
    }

    auto* metaData = Util::MemoryManager::GetCudnnMetaData(convConfig);
    ConvolutionForward2D(metaData, output, input, filter);
}

void ConvBackward2D(float* dataGradientOut, float* filter,
                    float* filterGradientOut, float* input,
                    float* gradientInput, Shape4D inputShape,
                    Shape4D filterShape, int strideRow, int strideCol,
                    int dilationRow, int dilationCol, int paddedRow,
                    int paddedCol)
{
    const ConvConfig convConfig = { inputShape, filterShape, strideRow,
                                    strideCol, dilationRow, dilationCol,
                                    paddedRow, paddedCol };

    if (!Util::MemoryManager::HasConvConfig(convConfig))
    {
        auto* metaData = new CudnnMetaData();
        CreateCudnnMetaData(metaData, inputShape, filterShape, strideRow,
                            strideCol, dilationRow, dilationCol, paddedRow,
                            paddedCol);
        Util::MemoryManager::AddCudnnMetaData(convConfig, metaData);
    }

    auto* metaData = Util::MemoryManager::GetCudnnMetaData(convConfig);
    ConvolutionBackward2D(metaData, dataGradientOut, filter, filterGradientOut,
                          input, gradientInput);
}
} // namespace Sapphire::Compute::Cuda
