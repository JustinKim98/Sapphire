// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <ModelTest/Conv2DModel.hpp>
#include <Sapphire/util/DataLoader/BinaryLoader.hpp>
#include <Sapphire/Model.hpp>
#include <Sapphire/operations/Forward/Linear.hpp>
#include <Sapphire/operations/Forward/Conv2D.hpp>
#include <Sapphire/operations/Forward/Functional/ReLU.hpp>
#include <Sapphire/operations/Loss/CrossEntropy.hpp>
#include <Sapphire/operations/optimizers/SGD.hpp>
#include <Sapphire/util/ResourceManager.hpp>
#include <Sapphire/operations/Forward/Functional/MaxPool2D.hpp>
#include <Sapphire/util/FileManager.hpp>
#include <Sapphire/operations/Forward/Functional/Softmax.hpp>
#include <iostream>
#include <random>

namespace Sapphire::Test
{
void Conv2DModelTest(
    std::filesystem::path filePath,
    int batchSize,
    float learningRate, bool hostMode, int epochs)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution dist(0, batchSize - 1);
    Util::BinaryLoader<std::uint8_t> dataLoader(std::move(filePath), 3073,
                                                3073 * 9999, batchSize, 3073);

    ModelManager::AddModel("SimpleConv2DModel");
    ModelManager::SetCurrentModel("SimpleConv2DModel");

    const CudaDevice gpu(0, "cuda0");
    constexpr auto xRows = 32;
    constexpr auto xCols = 32;

    //! Declare conv2d Layer
    NN::Conv2D conv0(6, 3, std::make_pair(5, 5), std::pair(1, 1),
                     std::pair(0, 0), std::pair(1, 1), false);
    NN::Conv2D conv1(16, 6, std::make_pair(5, 5), std::pair(1, 1),
                     std::make_pair(0, 0), std::make_pair(1, 1), false);
    NN::Linear fc0(16 * 5 * 5, 120);
    NN::Linear fc1(120, 84);
    NN::Linear fc2(84, 10);

    Tensor x(Shape({ batchSize, 3, xRows, xCols }), gpu, Type::Dense,
             true);
    Tensor label(Shape({ batchSize, 10 }), gpu,
                 Type::Dense, true);

    //! Configure data to be on host if needed
    if (hostMode)
    {
        x.ToHost();
        label.ToHost();
    }

    Optimizer::SGD sgd(learningRate);
    ModelManager::CurModel().SetOptimizer(&sgd);

    auto dataPreProcess =
        [batchSize](std::vector<std::uint8_t> data) -> std::vector<float> {
        std::vector<float> outData(batchSize * 32 * 32 * 3);
        if (data.size() > static_cast<std::size_t>(32 * 32 * 3) * batchSize)
            throw std::runtime_error(
                "dataPreProcess - given data was larger than expected");
        for (std::size_t i = 0; i < data.size(); ++i)
            outData[i] = static_cast<float>(data[i]) / 255.0f;
        return outData;
    };

    auto labelOneHot =
        [batchSize](std::vector<std::uint8_t> label) -> std::vector<float> {
        std::vector oneHot(batchSize * 10, 0.0f);
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            const auto idx = label.at(batchIdx);
            if (idx >= 10 || idx < 0)
                throw std::runtime_error("labelOneHot - idx out of range");
            oneHot.at(batchIdx * 10 + idx) = 1.0f;
        }
        return oneHot;
    };

    std::vector<std::size_t> batches(batchSize);

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        for (auto& elem : batches)
        {
            elem = dist(gen);
        }

        dataLoader.LoadData(x, batches, 1, 32 * 32 * 3, dataPreProcess);
        dataLoader.LoadData(label, batches, 0, 0, labelOneHot);

        //! Load data to x and label here
        auto tensor = F::MaxPool2D(F::ReLU(conv0(x)), std::make_pair(2, 2),
                                   std::make_pair(2, 2));
        tensor = F::MaxPool2D(F::ReLU(conv1(tensor)), std::make_pair(2, 2),
                              std::make_pair(2, 2));
        tensor.Reshape(
            Shape({ batchSize, tensor.GetShape().Size() / batchSize }));
        tensor = F::ReLU(fc0(tensor));
        tensor = F::ReLU(fc1(tensor));
        tensor = fc2(tensor);
        tensor = F::SoftMax(tensor);

        auto loss = NN::Loss::CrossEntropy(tensor, label);

        //! Print loss and accuracy every 100 epochs
        if (epoch % 50 == 0)
        {
            const auto yData = tensor.GetData();
            const auto labelData = label.GetData();
            const auto lossData = loss.GetData();

            int correct = 0;
            for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            {
                int modelOutput = 0;
                float largest = 0.0f;
                for (int idx = 0; idx < 10; ++idx)
                    if (yData[batchIdx * 10 + idx] > largest)
                    {
                        largest = yData[batchIdx * 10 + idx];
                        modelOutput = idx;
                    }

                int trueLabel = 0;
                largest = 0.0f;
                for (int idx = 0; idx < 10; ++idx)
                    if (labelData[batchIdx * 10 + idx] > largest)
                    {
                        largest = labelData[batchIdx * 10 + idx];
                        trueLabel = idx;
                    }

                if (modelOutput == trueLabel)
                    correct += 1;
            }
            std::cout << "epoch: " << epoch << " loss : " << lossData[0]
                << " Accuracy : "
                << static_cast<float>(correct) / batchSize << std::endl;
        }

        //! Start back propagation and update weights
        ModelManager::CurModel().BackProp(loss);
        //! Clear the gradients for next back propagation
        ModelManager::CurModel().Clear();

        //! Clear resources for every 10 epochs
        if (epoch % 10 == 0)
            Util::ResourceManager::Clean();
    }

    Util::ResourceManager::ClearAll();
}
} // namespace Sapphire::Test
