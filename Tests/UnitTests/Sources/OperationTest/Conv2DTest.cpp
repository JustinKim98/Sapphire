// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <OperationTest/Conv2DTest.hpp>
#include <Sapphire/Model.hpp>
#include <Sapphire/operations/Forward/Conv2D.hpp>
#include <Sapphire/operations/optimizers/SGD.hpp>
#include <TestUtil.hpp>
#include <iostream>
#include <doctest/doctest.h>
#include <random>

namespace Sapphire::Test
{
void TestConv2D(bool print)
{
    ModelManager::AddModel("myModel");
    ModelManager::SetCurrentModel("myModel");

    const CudaDevice gpu(0, "cuda0");
    const int batchSize = 4;
    const int inputChannels = 3;
    const int outputChannels = 3;
    const int inputRows = 4;
    const int inputCols = 4;
    const int kernelRows = 3;
    const int kernelCols = 3;
    const int strideRows = 1;
    const int strideCols = 1;
    const int dilationRows = 1;
    const int dilationCols = 1;
    const int padSizeRows = 1;
    const int padSizeCols = 1;

    const auto inputSize = std::make_pair(inputRows, inputCols);
    const auto kernelSize = std::make_pair(kernelRows, kernelCols);
    const auto stride = std::make_pair(strideRows, strideCols);
    const auto dilation = std::make_pair(dilationRows, dilationCols);
    const auto padSize = std::make_pair(padSizeRows, padSizeCols);

    const auto outputRows =
        (inputRows + 2 * padSizeRows - dilationRows * (kernelRows - 1) - 1) /
        strideRows +
        1;
    const auto outputCols =
        (inputCols + 2 * padSizeCols - dilationCols * (kernelCols - 1) - 1) /
        strideCols +
        1;

    //! Initialize backward data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> backwardData(batchSize *
                                    outputChannels * outputRows * outputCols);
    for (auto& data : backwardData)
        data = dist(gen);

    Tensor input(Shape({ batchSize, inputChannels, inputRows, inputCols }), gpu,
                 Type::Dense);
    Tensor kernel(Shape({ outputChannels, inputChannels,
                          std::get<0>(kernelSize), std::get<1>(kernelSize) }),
                  gpu, Type::Dense);
    Tensor bias(Shape({ outputChannels }), gpu, Type::Dense);
    input.SetMode(DeviceType::Host);
    kernel.SetMode(DeviceType::Host);
    bias.SetMode(DeviceType::Host);

    //! Initialize input, kernel and bias
    Initialize::Initialize(
        input, std::make_unique<Initialize::Normal>(0.0f, 1.0f));
    Initialize::Initialize(
        kernel, std::make_unique<Initialize::Normal>(0.0f, 1.0f));
    Initialize::Initialize(
        bias, std::make_unique<Initialize::Normal>(0.0f, 1.0f));

    //! Move tensors to gpu
    input.ToCuda();
    kernel.ToCuda();
    bias.ToCuda();

    //! Test Conv2D on gpu
    NN::Conv2D conv2D(inputSize, stride, padSize, dilation,
                      new Optimizer::SGD(0.0f), kernel,
                      bias);
    auto gpuOutput = conv2D(input);
    CHECK(gpuOutput.GetShape().Rows() == outputRows);
    CHECK(gpuOutput.GetShape().Cols() == outputCols);
    const auto gpuForwardPtr = gpuOutput.GetForwardDataCopy();
    gpuOutput.SetBackwardData(backwardData);
    ModelManager::CurModel().BackProp(gpuOutput);
    const auto gpuBackwardPtr = input.GetBackwardDataCopy();

    //! Move tensors to host
    input.ToHost();
    kernel.ToHost();
    bias.ToHost();

    //! Initialize backward data
    Initialize::InitializeBackwardData(input,
                                       std::make_unique<Initialize::Zeros>());

    //! Test Conv2D on host
    NN::Conv2D conv2DHost(inputSize, stride, padSize, dilation,
                          new Optimizer::SGD(0.0f),
                          kernel,
                          bias);
    auto hostOutput = conv2DHost(input);
    const auto hostForwardPtr = hostOutput.GetForwardDataCopy();
    const auto outputRowsHost = hostOutput.GetShape().Rows();
    const auto outputColsHost = hostOutput.GetShape().Cols();
    hostOutput.SetBackwardData(backwardData);
    ModelManager::CurModel().BackProp(hostOutput);
    const auto hostBackwardPtr = input.GetBackwardDataCopy();

    if (print)
    {
        std::cout << "Conv2D forward result (Host)" << std::endl;
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            std::cout << "batch : " << batchIdx << std::endl;
            for (int channelIdx = 0; channelIdx < outputChannels; ++channelIdx)
            {
                std::cout << "channel" << channelIdx << std::endl;
                for (int i = 0; i < outputRowsHost; ++i)
                {
                    for (int j = 0; j < outputColsHost; ++j)
                    {
                        std::cout
                            << hostForwardPtr[batchIdx * outputRows *
                                              outputCols * outputChannels +
                                              channelIdx * outputRows *
                                              outputCols +
                                              i * inputCols + j]
                            << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }

        std::cout << "Conv2D backward result (Host)" << std::endl;
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            std::cout << "batch : " << batchIdx << std::endl;
            for (int channelIdx = 0; channelIdx < outputChannels; ++channelIdx)
            {
                std::cout << "channel : " << channelIdx << std::endl;
                for (int i = 0; i < inputRows; ++i)
                {
                    for (int j = 0; j < inputCols; ++j)
                    {
                        std::cout
                            << hostBackwardPtr[batchIdx * outputRows *
                                               outputCols * outputChannels +
                                               channelIdx * outputRows *
                                               outputCols +
                                               i * inputCols + j]
                            << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }

        std::cout << "Conv2D forward result (Cuda)" << std::endl;
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            std::cout << "batch : " << batchIdx << std::endl;
            for (int channelIdx = 0; channelIdx < batchSize; ++channelIdx)
            {
                std::cout << "channel : " << channelIdx << std::endl;
                for (int i = 0; i < outputRows; ++i)
                {
                    for (int j = 0; j < outputCols; ++j)
                    {
                        std::cout << gpuForwardPtr[
                                batchIdx * outputRows * outputCols *
                                outputChannels +
                                channelIdx * outputRows *
                                outputCols +
                                i * inputCols + j]
                            << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }

        std::cout << "Conv2D backward result (Cuda)" << std::endl;
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            std::cout << "batch : " << batchIdx << std::endl;
            for (int channelIdx = 0; channelIdx < outputChannels; ++channelIdx)
            {
                std::cout << "channel : " << channelIdx << std::endl;
                for (int i = 0; i < inputRows; ++i)
                {
                    for (int j = 0; j < inputCols; ++j)
                    {
                        std::cout
                            << gpuBackwardPtr[batchIdx * outputRows *
                                              outputCols * outputChannels +
                                              channelIdx * outputRows *
                                              outputCols +
                                              i * inputCols + j]
                            << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }
    }

    CHECK(outputRows == outputRowsHost);
    CHECK(outputCols == outputColsHost);

    for (int idx = 0;
         idx < batchSize * outputChannels * outputRows * outputCols; ++idx)
        CHECK(TestEquality(hostForwardPtr[idx], gpuForwardPtr[idx]));

    for (int idx = 0;
         idx < batchSize * outputChannels * outputRows * outputCols; ++idx)
        CHECK(TestEquality(hostBackwardPtr[idx], gpuBackwardPtr[idx]));

    ModelManager::CurModel().Clear();
}
}
