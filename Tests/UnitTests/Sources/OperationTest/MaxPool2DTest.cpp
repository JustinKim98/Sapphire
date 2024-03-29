// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <OperationTest/MaxPool2DTest.hpp>
#include <Sapphire/operations/Forward/Functional/MaxPool2D.hpp>
#include <Sapphire/Model.hpp>
#include <Sapphire/operations/optimizers/SGD.hpp>
#include <Sapphire/operations/Initializers/Initialize.hpp>
#include <Sapphire/util/ResourceManager.hpp>
#include <TestUtil.hpp>
#include <random>
#include <iostream>
#include <doctest.h>

namespace Sapphire::Test
{
void TestMaxPool2D(bool print)
{
    ModelManager::AddModel("myModel");
    ModelManager::SetCurrentModel("myModel");

    const CudaDevice gpu(0, "cuda0");
    constexpr int batchSize = 4;
    constexpr int channels = 3;
    constexpr int inputRows = 6;
    constexpr int inputCols = 4;
    constexpr int windowRows = 3;
    constexpr int windowCols = 3;
    constexpr int strideRows = 1;
    constexpr int strideCols = 2;
    constexpr int padSizeRows = 2;
    constexpr int padSizeCols = 1;

    constexpr auto windowSize = std::make_pair(windowRows, windowCols);
    constexpr auto stride = std::make_pair(strideRows, strideCols);
    constexpr auto padSize = std::make_pair(padSizeRows, padSizeCols);

    constexpr auto outputRows =
        (inputRows + 2 * padSizeRows - (windowRows - 1) - 1) /
        strideRows +
        1;
    constexpr auto outputCols =
        (inputCols + 2 * padSizeCols - (windowCols - 1) - 1) /
        strideCols +
        1;

    //! Initialize backward data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> backwardData(batchSize * channels * outputRows *
                                    outputCols);
    for (auto& data : backwardData)
        data = dist(gen);

    Tensor input(Shape({ batchSize, channels, inputRows, inputCols }),
                 gpu);
    Tensor filter(Shape({ channels, channels,
                          std::get<0>(windowSize), std::get<1>(windowSize) }),
                  gpu);
    Tensor bias(Shape({ channels }), gpu);
    input.SetMode(ComputeMode::Host);
    filter.SetMode(ComputeMode::Host);
    bias.SetMode(ComputeMode::Host);

    //! Initialize input, kernel and bias
    Initialize::Initialize(input,
                           std::make_unique<Initialize::Normal>(
                               0.0f, 1.0f));
    Initialize::Initialize(filter,
                           std::make_unique<Initialize::Normal>(
                               0.0f, 1.0f));
    Initialize::Initialize(bias,
                           std::make_unique<Initialize::Normal>(
                               0.0f, 1.0f));

    //! Move tensors to gpu
    input.ToCuda();
    filter.ToCuda();
    bias.ToCuda();

    //! Test Conv2D on gpu
    auto gpuOutput = F::MaxPool2D(input, windowSize, stride, padSize);
    CHECK(gpuOutput.GetShape().Rows() == outputRows);
    CHECK(gpuOutput.GetShape().Cols() == outputCols);
    const auto gpuForwardData = gpuOutput.GetData();
    gpuOutput.LoadGradient(backwardData);

    Optimizer::SGD sgd(0.0f);
    ModelManager::CurModel().SetOptimizer(&sgd);
    ModelManager::CurModel().BackProp(gpuOutput);
    const auto gpuBackwardData = input.GetGradient();

    //! Move tensors to host
    input.ToHost();
    filter.ToHost();
    bias.ToHost();

    //! Initialize backward data
    Initialize::InitializeGradient(input,
                                   std::make_unique<Initialize::Zeros>());

    auto hostOutput = F::MaxPool2D(input, windowSize, stride, padSize);
    const auto hostForwardData = hostOutput.GetData();
    const auto outputRowsHost = hostOutput.GetShape().Rows();
    const auto outputColsHost = hostOutput.GetShape().Cols();
    hostOutput.LoadGradient(backwardData);
    ModelManager::CurModel().BackProp(hostOutput);
    const auto hostBackwardData = input.GetGradient();

    if (print)
    {
        std::cout << "Conv2D forward result (Host)" << std::endl;
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            std::cout << "batch : " << batchIdx << std::endl;
            for (int channelIdx = 0; channelIdx < channels; ++channelIdx)
            {
                std::cout << "channel" << channelIdx << std::endl;
                for (int i = 0; i < outputRowsHost; ++i)
                {
                    for (int j = 0; j < outputColsHost; ++j)
                    {
                        std::cout
                            << hostForwardData[batchIdx * outputRows *
                                               outputCols * channels +
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
            for (int channelIdx = 0; channelIdx < channels; ++channelIdx)
            {
                std::cout << "channel : " << channelIdx << std::endl;
                for (int i = 0; i < outputRows; ++i)
                {
                    for (int j = 0; j < outputCols; ++j)
                    {
                        std::cout
                            << gpuForwardData[batchIdx * outputRows *
                                              outputCols * channels +
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
            for (int channelIdx = 0; channelIdx < channels; ++channelIdx)
            {
                std::cout << "channel : " << channelIdx << std::endl;
                for (int i = 0; i < inputRows; ++i)
                {
                    for (int j = 0; j < inputCols; ++j)
                    {
                        std::cout << hostBackwardData[batchIdx * outputRows *
                                outputCols *
                                channels +
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
            for (int channelIdx = 0; channelIdx < channels; ++channelIdx)
            {
                std::cout << "channel : " << channelIdx << std::endl;
                for (int i = 0; i < inputRows; ++i)
                {
                    for (int j = 0; j < inputCols; ++j)
                    {
                        std::cout
                            << gpuBackwardData[batchIdx * outputRows *
                                               outputCols * channels +
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

    CHECK(outputRows == outputRowsHost);
    CHECK(outputCols == outputColsHost);

    for (int idx = 0;
         idx < batchSize * channels * outputRows * outputCols; ++idx)
        CHECK(TestEquality(hostForwardData[idx], gpuForwardData[idx]));

    for (int idx = 0;
         idx < batchSize * channels * outputRows * outputCols; ++idx)
        CHECK(TestEquality(hostBackwardData[idx], gpuBackwardData[idx]));

    ModelManager::CurModel().Clear();
    Util::ResourceManager::ClearAll();
}
}
