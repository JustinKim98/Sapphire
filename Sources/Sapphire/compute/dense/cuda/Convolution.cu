// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/dense/cuda/Convolution.cuh>
#include <Sapphire/util/ResourceManager.hpp>

namespace Sapphire::Compute::Dense::Cuda
{
__host__ void CreateCudnnConv2DMetaData(CudnnConv2DMetaData* metaData,
                                        Shape4D xShape, Shape4D filterShape,
                                        int strideRow, int strideCol,
                                        int dilationRow, int dilationCol,
                                        int rowPadding, int columnPadding,
                                        int deviceId)
{
    int outputN = xShape.N;
    int outputChannels = filterShape.N;
    int outputHeight = (xShape.Height + 2 * rowPadding -
                        dilationRow * (filterShape.Height - 1) - 1) /
                       strideRow +
                       1;
    int outputWidth = (xShape.Width + 2 * columnPadding -
                       dilationCol * (filterShape.Width - 1) - 1) /
                      strideCol +
                      1;
    cudaSetDevice(deviceId);
    cudnnHandle_t* handle = Util::ResourceManager::GetCudnnHandle(
        deviceId, std::this_thread::get_id());
    checkCuDNN(cudnnCreateConvolutionDescriptor(&metaData->ConvDesc));
    checkCuDNN(cudnnCreateTensorDescriptor(&metaData->InputDesc));
    checkCuDNN(cudnnCreateFilterDescriptor(&metaData->FilterDesc));
    checkCuDNN(cudnnCreateTensorDescriptor(&metaData->OutputDesc));

    checkCuDNN(cudnnSetConvolution2dDescriptor(
        metaData->ConvDesc, rowPadding, columnPadding, strideRow, strideCol,
        dilationRow, dilationCol, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

    checkCuDNN(cudnnSetTensor4dDescriptor(
        metaData->InputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, xShape.N,
        xShape.Channels, xShape.Height, xShape.Width));

    checkCuDNN(cudnnSetFilter4dDescriptor(
        metaData->FilterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        filterShape.N, filterShape.Channels, filterShape.Height,
        filterShape.Width));

    checkCuDNN(cudnnGetConvolution2dForwardOutputDim(
        metaData->ConvDesc, metaData->InputDesc, metaData->FilterDesc, &outputN,
        &outputChannels, &outputHeight, &outputWidth));

    checkCuDNN(cudnnSetTensor4dDescriptor(
        metaData->OutputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, outputN,
        outputChannels, outputHeight, outputWidth));

    int numAlgo;

    cudnnConvolutionFwdAlgoPerf_t forwardPerf;
    checkCuDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        *handle, metaData->InputDesc, metaData->FilterDesc,
        metaData->ConvDesc, metaData->OutputDesc, 1, &numAlgo, &forwardPerf));
    metaData->ForwardAlgo = forwardPerf.algo;

    cudnnConvolutionBwdDataAlgoPerf_t backDataPerf;
    checkCuDNN(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        *handle, metaData->FilterDesc, metaData->OutputDesc,
        metaData->ConvDesc, metaData->InputDesc, 1, &numAlgo, &backDataPerf));
    metaData->BackwardDataAlgo = backDataPerf.algo;

    cudnnConvolutionBwdFilterAlgoPerf_t backFilterPerf;
    checkCuDNN(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        *handle, metaData->InputDesc, metaData->OutputDesc,
        metaData->ConvDesc, metaData->FilterDesc, 1, &numAlgo,
        &backFilterPerf));
    metaData->BackwardFilterAlgo = backFilterPerf.algo;

    checkCuDNN(cudnnGetConvolutionForwardWorkspaceSize(
        *handle, metaData->InputDesc, metaData->FilterDesc,
        metaData->ConvDesc, metaData->OutputDesc, metaData->ForwardAlgo,
        &metaData->ForwardWorkSpaceBytes));

    checkCuDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
        *handle, metaData->FilterDesc, metaData->OutputDesc,
        metaData->ConvDesc, metaData->InputDesc, metaData->BackwardDataAlgo,
        &metaData->BackwardDataWorkSpaceBytes));

    checkCuDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        *handle, metaData->InputDesc, metaData->OutputDesc,
        metaData->ConvDesc, metaData->FilterDesc, metaData->BackwardFilterAlgo,
        &metaData->BackwardFilterWorkSpaceBytes));

    checkCuda(cudaMalloc(static_cast<void**>(&metaData->ForwardWorkSpace),
                         metaData->ForwardWorkSpaceBytes));

    checkCuda(cudaMalloc(static_cast<void**>(&metaData->BackwardDataWorkSpace),
                         metaData->BackwardDataWorkSpaceBytes));

    checkCuda(cudaMalloc(
        static_cast<void**>(&metaData->BackwardFilterWorkSpace),
        metaData->BackwardFilterWorkSpaceBytes));
}

__host__ void CudnnConvolutionForward2D(CudnnConv2DMetaData* metadata,
                                        float* output,
                                        const float* input, const float* filter,
                                        int deviceId)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnHandle_t* handle = Util::ResourceManager::GetCudnnHandle(
        deviceId, std::this_thread::get_id());
    checkCuDNN(cudnnConvolutionForward(
        *handle, &alpha, metadata->InputDesc, input,
        metadata->FilterDesc, filter, metadata->ConvDesc, metadata->ForwardAlgo,
        metadata->ForwardWorkSpace, metadata->ForwardWorkSpaceBytes, &beta,
        metadata->OutputDesc, output));
}

__host__ void CudnnConvolutionBackward2D(
    CudnnConv2DMetaData* descriptors, float* dx, const float* filter,
    float* dFilter, const float* x, const float* dy, int deviceId)
{
    float alpha = 1.0f;
    float beta = 1.0f;
    cudnnHandle_t* handle = Util::ResourceManager::GetCudnnHandle(
        deviceId, std::this_thread::get_id());
    checkCuDNN(cudnnConvolutionBackwardData(
        *handle, &alpha, descriptors->FilterDesc, filter,
        descriptors->OutputDesc, dy, descriptors->ConvDesc,
        descriptors->BackwardDataAlgo, descriptors->BackwardDataWorkSpace,
        descriptors->BackwardDataWorkSpaceBytes, &beta, descriptors->InputDesc,
        dx));

    checkCuDNN(cudnnConvolutionBackwardFilter(
        *handle, &alpha, descriptors->InputDesc, x,
        descriptors->OutputDesc, dy, descriptors->ConvDesc,
        descriptors->BackwardFilterAlgo, descriptors->BackwardFilterWorkSpace,
        descriptors->BackwardFilterWorkSpaceBytes, &beta,
        descriptors->FilterDesc, dFilter));
}

__host__ void Conv2DForward(float* y, const float* x,
                            const float
                            * filter, Shape4D inputShape, Shape4D filterShape,
                            int strideRow, int strideCol, int dilationRow,
                            int dilationCol, int rowPadding, int columnPadding,
                            int deviceId)
{
    const ConvConfig convConfig = { inputShape, filterShape, strideRow,
                                    strideCol,
                                    dilationRow, dilationCol, rowPadding,
                                    columnPadding };

    cudaSetDevice(deviceId);
    const auto tid = std::this_thread::get_id();
    if (!Util::ResourceManager::HasCudnnHandle(deviceId, tid))
    {
        Util::ResourceManager::AddCudnnHandle(deviceId, tid);
    }
    if (!Util::ResourceManager::HasConvConfig(convConfig))
    {
        Util::ResourceManager::AddCudnnConv2DMetaData(
            convConfig, inputShape, filterShape, strideRow, strideCol,
            dilationRow, dilationCol, rowPadding, columnPadding, deviceId);
    }

    auto* metaData = Util::ResourceManager::GetCudnnConvMetaData(convConfig);
    CudnnConvolutionForward2D(metaData, y, x, filter, deviceId);
}

__host__ void Conv2DBackward(float* dx, const float* filter, float* dFilter,
                             const float* x, const float* dy,
                             Shape4D inputShape, Shape4D filterShape,
                             int strideRow, int strideCol, int dilationRow,
                             int dilationCol, int rowPadding, int columnPadding,
                             int deviceId)
{
    const ConvConfig convConfig = { inputShape, filterShape, strideRow,
                                    strideCol, dilationRow, dilationCol,
                                    rowPadding, columnPadding };

    cudaSetDevice(deviceId);
    const auto tid = std::this_thread::get_id();
    if (!Util::ResourceManager::HasCudnnHandle(deviceId, tid))
    {
        throw std::runtime_error(
            "Compute::Dense::Cuda::Conv2DBackward - CudnnHandle was not "
            "found");
    }

    if (!Util::ResourceManager::HasConvConfig(convConfig))
    {
        throw std::runtime_error(
            "Compute::Dense::Cuda::Conv2DBackward - CudnnConv2DMetaData was "
            "not found");
    }

    auto* metaData = Util::ResourceManager::GetCudnnConvMetaData(convConfig);
    CudnnConvolutionBackward2D(metaData, dx, filter, dFilter,
                               x, dy, deviceId);
}
} // namespace Sapphire::Compute::Cuda
