// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <ModelTest/MnistLinear.hpp>
#include <Sapphire/Model.hpp>
#include <Sapphire/operations/Forward/Linear.hpp>
#include <Sapphire/operations/Forward/Functional/ReLU.hpp>
#include <Sapphire/operations/Forward/Conv2D.hpp>
#include <Sapphire/operations/Loss/CrossEntropy.hpp>
#include <Sapphire/operations/optimizers/SGD.hpp>
#include <Sapphire/operations/Forward/Functional/Softmax.hpp>
#include <Sapphire/util/DataLoader/CsvLoader.hpp>
#include <Sapphire/util/FileManager.hpp>
#include <Sapphire/util/ResourceManager.hpp>
#include <iostream>


namespace Sapphire::Test
{
void MnistLinear(std::filesystem::path filePath, int batchSize,
                 float learningRate,
                 int epochs, bool hostMode)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    Util::CsvLoader<int> dataLoader(filePath);
    std::uniform_int_distribution<int> dist(1, dataLoader.GetLineSize() - 1);

    ModelManager::AddModel("MnistLinear");
    ModelManager::SetCurrentModel("MnistLinear");

    const CudaDevice gpu(0, "cuda0");

    NN::Linear fc0(784, 100);
    NN::Linear fc1(100, 10);

    Tensor x(Shape({ batchSize, 784 }), gpu, Type::Dense, true);
    Tensor label(Shape({ batchSize, 10 }), gpu, Type::Dense, true);

    if (hostMode)
    {
        x.ToHost();
        label.ToHost();
    }

    Optimizer::SGD sgd(learningRate);
    ModelManager::CurModel().SetOptimizer(&sgd);

    auto labelOneHot = [batchSize
        ](std::vector<int> label) -> std::vector<float> {

        std::vector oneHot(10 * batchSize, 0.0f);
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            const auto idx = label.at(batchIdx);
            if (idx >= 10 || idx < 0)
                throw std::runtime_error("labelOneHot - idx out of range");
            oneHot.at(batchIdx * 10 + idx) = 1.0f;
        }
        return oneHot;
    };

    auto dataPreProcess = [batchSize](
        std::vector<int> data) -> std::vector<float> {
        std::vector<float> outData(batchSize * 784);
        if (data.size() > static_cast<std::size_t>(784) * batchSize)
            throw std::runtime_error(
                "dataPreProcess - given data was larger than expected");
        for (std::size_t i = 0; i < data.size(); ++i)
            outData[i] =
                static_cast<float>(data[i]) / 255.0f;
        return outData;
    };

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        std::vector<std::size_t> batches(batchSize);
        for (auto& elem : batches)
        {
            elem = dist(gen);
        }

        dataLoader.LoadData(x, batches, 1, 784, dataPreProcess);
        dataLoader.LoadData(label, batches, 0, 0, labelOneHot);

        auto tensor = F::ReLU(fc0(x));
        tensor = fc1(tensor);
        tensor = F::SoftMax(tensor);
        const auto loss = NN::Loss::CrossEntropy(tensor, label);

        //! Print loss and accuracy every 100 epochs
        if (epoch % 100 == 0)
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
            std::cout << "epoch: " << epoch << " loss : " << lossData[0] <<
                " Accuracy : "
                << static_cast<float>(correct) / batchSize << std::endl;
        }
        ModelManager::CurModel().BackProp(loss);
        ModelManager::CurModel().Clear(); //! initialize gradients to zero
        if (epoch % 10 == 0)
            Util::ResourceManager::Clean(); //! Clean the resource
    }

    Util::ResourceManager::ClearAll(); //! Clear all resources
}
}
