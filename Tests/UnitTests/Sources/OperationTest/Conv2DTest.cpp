// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <OperationTest/Conv2DTest.hpp>
#include <Sapphire/Model.hpp>
#include <Sapphire/operations/Forward/Conv2D.hpp>
#include <Sapphire/operations/optimizers/SGD.hpp>
#include <Sapphire/operations/Forward/Linear.hpp>
#include <Sapphire/operations/Loss/MSE.hpp>
#include <Sapphire/util/ResourceManager.hpp>
#include <TestUtil.hpp>
#include <doctest/doctest.h>
#include <iostream>
#include <random>

namespace Sapphire::Test
{
void TestConv2D(bool print)
{
    //! Initialize model hyperparameters
    constexpr int batchSize = 4;
    constexpr int inputChannels = 2;
    constexpr int outputChannels = 4;
    constexpr int inputRows = 6;
    constexpr int inputCols = 8;
    constexpr int kernelRows = 3;
    constexpr int kernelCols = 4;
    constexpr int strideRows = 2;
    constexpr int strideCols = 1;
    constexpr int dilationRows = 2;
    constexpr int dilationCols = 2;
    constexpr int padSizeRows = 2;
    constexpr int padSizeCols = 1;

    constexpr auto filterSize = std::make_pair(kernelRows, kernelCols);
    constexpr auto stride = std::make_pair(strideRows, strideCols);
    constexpr auto dilation = std::make_pair(dilationRows, dilationCols);
    constexpr auto padSize = std::make_pair(padSizeRows, padSizeCols);

    constexpr auto outputRows =
        (inputRows + 2 * padSizeRows - dilationRows * (kernelRows - 1) - 1) /
        strideRows +
        1;
    constexpr auto outputCols =
        (inputCols + 2 * padSizeCols - dilationCols * (kernelCols - 1) - 1) /
        strideCols +
        1;

    //! Initialize random input data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution dist(-10.0f, 10.0f);
    std::vector<float> forwardData(batchSize * outputChannels * outputRows *
                                   outputCols);
    std::vector<float> backwardData(batchSize *
                                    outputChannels * outputRows * outputCols);

    for (auto& data : forwardData)
        data = dist(gen);
    for (auto& data : backwardData)
        data = dist(gen);

    //! Initialize model and device
    ModelManager::AddModel("conv2D test model");
    ModelManager::SetCurrentModel("conv2D test model");
    const DeviceInfo gpu(0, "cuda0");

    Tensor input(Shape({ batchSize, inputChannels, inputRows, inputCols }),
                 gpu, true);

    input.SetMode(ComputeMode::Host);

    //! Declare layer to test
    NN::Conv2D conv2D(outputChannels, inputChannels, filterSize, stride,
                      padSize, dilation, true);
    auto filterData = conv2D.GetFilter().GetData();
    auto biasData = conv2D.GetBias().GetData();

    //! Move tensors to gpu
    input.ToCuda();

    //! Setup optimizer
    Optimizer::SGD sgd(0.01f);
    ModelManager::CurModel().SetOptimizer(&sgd);

    //! Test Conv2D on gpu
    auto gpuOutput = conv2D(input);
    CHECK(gpuOutput.GetShape().Rows() == outputRows);
    CHECK(gpuOutput.GetShape().Cols() == outputCols);
    const auto gpuForwardData = gpuOutput.GetData();
    gpuOutput.LoadGradient(backwardData);
    ModelManager::CurModel().BackProp(gpuOutput);
    const auto gpuGradient = input.GetGradient();
    const auto gpuFilterData = conv2D.GetFilter().GetData();
    const auto gpuBiasData = conv2D.GetBias().GetData();

    //! Reset backward Gradient
    ModelManager::CurModel().Clear();
    conv2D.GetFilter().LoadData(filterData);
    conv2D.GetBias().LoadData(biasData);

    //! Move tensors to host
    input.ToHost();
    auto hostOutput = conv2D(input);
    const auto hostForwardData = hostOutput.GetData();
    CHECK(hostOutput.GetShape().Rows() == outputRows);
    CHECK(hostOutput.GetShape().Cols() == outputCols);
    hostOutput.LoadGradient(backwardData);
    ModelManager::CurModel().BackProp(hostOutput);
    const auto hostBackwardData = input.GetGradient();
    const auto hostFilterData = conv2D.GetFilter().GetData();
    const auto hostBiasData = conv2D.GetBias().GetData();

    if (print)
    {
        std::cout << "Conv2D forward result (Host)" << std::endl;
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            std::cout << "batch : " << batchIdx << std::endl;
            for (int channelIdx = 0; channelIdx < outputChannels; ++channelIdx)
            {
                std::cout << "channel" << channelIdx << std::endl;
                for (int i = 0; i < outputRows; ++i)
                {
                    for (int j = 0; j < outputCols; ++j)
                    {
                        std::cout
                            << hostForwardData[batchIdx * outputRows *
                                               outputCols * outputChannels +
                                               channelIdx * outputRows *
                                               outputCols +
                                               i * outputCols + j]
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
            for (int channelIdx = 0; channelIdx < outputChannels; ++channelIdx)
            {
                std::cout << "channel : " << channelIdx << std::endl;
                for (int i = 0; i < outputRows; ++i)
                {
                    for (int j = 0; j < outputCols; ++j)
                    {
                        std::cout
                            << gpuForwardData[batchIdx * outputRows *
                                              outputCols * outputChannels +
                                              channelIdx * outputRows *
                                              outputCols +
                                              i * outputCols + j]
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
                        std::cout << hostBackwardData[batchIdx * outputRows *
                                outputCols *
                                outputChannels +
                                channelIdx * outputRows *
                                outputCols +
                                i * outputCols + j]
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
                            << gpuGradient[batchIdx * outputRows *
                                           outputCols * outputChannels +
                                           channelIdx * outputRows *
                                           outputCols +
                                           i * outputCols + j]
                            << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }
    }

    for (int i = 0;
         i < batchSize * outputChannels * outputRows * outputCols; ++i)
        CHECK(TestEquality(hostForwardData[i], gpuForwardData[i]));
    for (int i = 0;
         i < batchSize * outputChannels * outputRows * outputCols; ++i)
        CHECK(TestEquality(hostBackwardData[i], gpuGradient[i]));
    for (std::size_t i = 0; i < filterData.size(); ++i)
        CHECK(TestEquality(hostFilterData[i], gpuFilterData[i]));
    for (std::size_t i = 0; i < biasData.size(); ++i)
        CHECK(TestEquality(hostBiasData[i], gpuBiasData[i]));

    ModelManager::CurModel().Clear();
}

//! Test simple weight decay
void TestConv2DTraining(bool printData)
{
    constexpr int epochs = 100;
    constexpr int batchSize = 2;
    constexpr int inputChannels = 2;
    constexpr int outputChannels = 4;
    constexpr int inputRows = 6;
    constexpr int inputCols = 6;
    constexpr int kernelRows = 2;
    constexpr int kernelCols = 2;
    constexpr int strideRows = 2;
    constexpr int strideCols = 2;
    constexpr int dilationRows = 1;
    constexpr int dilationCols = 1;
    constexpr int padSizeRows = 1;
    constexpr int padSizeCols = 1;

    constexpr auto filterSize = std::make_pair(kernelRows, kernelCols);
    constexpr auto stride = std::make_pair(strideRows, strideCols);
    constexpr auto dilation = std::make_pair(dilationRows, dilationCols);
    constexpr auto padSize = std::make_pair(padSizeRows, padSizeCols);

    constexpr auto outputRows =
        (inputRows + 2 * padSizeRows - dilationRows * (kernelRows - 1) - 1) /
        strideRows +
        1;
    constexpr auto outputCols =
        (inputCols + 2 * padSizeCols - dilationCols * (kernelCols - 1) - 1) /
        strideCols +
        1;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution dist(-10.0f, 10.0f);

    ModelManager::AddModel("SimpleConv2DModel");
    ModelManager::SetCurrentModel("SimpleConv2DModel");

    const DeviceInfo gpu(0, "cuda0");

    NN::Conv2D conv2D(outputChannels, inputChannels, filterSize, stride,
                      padSize, dilation, true);

    Tensor x(Shape({ batchSize, inputChannels, inputRows, inputCols }), gpu,
             true);
    Tensor label(
        Shape({ batchSize * outputChannels * outputRows * outputCols }), gpu,
        Type::Dense, true);

    Optimizer::SGD sgd(0.01f);
    ModelManager::CurModel().SetOptimizer(&sgd);

    std::vector<float> labelData(
        batchSize * outputChannels * outputRows * outputCols);
    std::vector<float> xData(batchSize * inputChannels * inputRows * inputCols);

    for (auto& data : labelData)
        data = dist(gen);
    for (auto& data : xData)
        data = dist(gen);

    for (int i = 0; i < epochs; ++i)
    {
        x.LoadData(xData);
        label.LoadData(labelData);
        auto tensor = conv2D(x);
        tensor.Flatten();
        const auto loss = NN::Loss::MSE(tensor, label);
        if (i % 10 == 0)
        {
            if (printData)
            {
                const auto yDataCopy = tensor.GetData();
                const auto labelDataCopy = label.GetData();
                for (const auto& elem : yDataCopy)
                    std::cout << elem << " ";
                std::cout << std::endl;
            }
            const auto lossData = loss.GetData();
            std::cout << "epoch: " << i << " loss : " << lossData[0]
                << std::endl;
        }
        ModelManager::CurModel().BackProp(loss);
        ModelManager::CurModel().Clear();

        if (i % 10 == 0)
            Util::ResourceManager::Clean();
    }
    Util::ResourceManager::ClearAll();
}
}
