// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <ModelTest/Conv2DModel.hpp>
#include <Sapphire/Model.hpp>
#include <Sapphire/operations/Forward/Linear.hpp>
#include <Sapphire/operations/Forward/Conv2D.hpp>
#include <Sapphire/operations/Forward/ReLU.hpp>
#include <Sapphire/operations/Loss/CrossEntropy.hpp>
#include <Sapphire/operations/optimizers/SGD.hpp>
#include <Sapphire/util/ResourceManager.hpp>
#include <Sapphire/operations/Forward/MaxPool2D.hpp>
#include <Sapphire/tensor/CreateTensor.hpp>
#include <Sapphire/util/FileManager.hpp>
#include <Sapphire/operations/Forward/Softmax.hpp>
#include <Sapphire/operations/Loss/MSE.hpp>
#include <iostream>
#include <random>

namespace Sapphire::Test
{
void Conv2DModelTest(
    int batchSize,
    float learningRate,
    bool hostMode, int epochs)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution dist(0, batchSize - 1);

    ModelManager::AddModel("SimpleConv2DModel");
    ModelManager::SetCurrentModel("SimpleConv2DModel");

    const CudaDevice gpu(0, "cuda0");
    constexpr auto xRows = 32;
    constexpr auto xCols = 32;

    //! Declare conv2d Layer
    NN::Conv2D conv0(6, 3, std::make_pair(5, 5), std::pair(1, 1),
                     std::pair(0, 0), std::pair(1, 1), false);
    NN::MaxPool2D pool(std::make_pair(2, 2), std::make_pair(2, 2));
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

    const auto totalData = ReadFile<std::uint8_t>(
        std::string(
            "/mnt/c/Users/user/Documents/Sapphire/Datasets/cifar-10-batches-bin/"
            "data_batch_1.bin"));

    std::vector<float> labelData(batchSize * 10);
    std::vector<float> xData(batchSize * 32 * 32 * 3);
    std::vector<std::size_t> batches(batchSize);

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        for (auto& elem : batches)
        {
            elem = dist(gen);
        }

        std::fill(labelData.begin(), labelData.end(), 0.0f);
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            const auto batch = batches.at(batchIdx);
            const auto layerIdx =
                batchIdx * 10 +
                totalData.at(batch * (32 * 32 * 3 + 1));
            labelData[layerIdx] = 1.0f;
            for (int idx = 0; idx < 32 * 32 * 3; ++idx)
                xData[batchIdx * (32 * 32 * 3) + idx] =
                    static_cast<float>(totalData.at(
                        batch * (32 * 32 * 3 + 1) + idx + 1)) /
                    255.0f;
        }

        //! Load the data to model
        x.LoadData(xData);
        label.LoadData(labelData);

        //! Load data to x and label here
        auto tensor = pool(NN::ReLU(conv0(x)));
        tensor = pool(NN::ReLU(conv1(tensor)));
        tensor.Reshape(
            Shape({ batchSize, tensor.GetShape().Size() / batchSize }));
        tensor = NN::ReLU(fc0(tensor));
        tensor = NN::ReLU(fc1(tensor));
        tensor = fc2(tensor);
        tensor = NN::SoftMax(tensor);

        auto loss = NN::Loss::CrossEntropy(tensor, label);

        //! Print loss every 10 epochs
        if (epoch % 10 == 0)
        {
            const auto val = tensor.GetData();
            for (int idx = 0; idx < 10; ++idx)
                std::cout << val[idx] << " ";
            std::cout << std::endl;
            const auto lossData = loss.GetData();
            std::cout << "epoch: " << epoch << " loss : " << lossData[0]
                << std::endl;
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
