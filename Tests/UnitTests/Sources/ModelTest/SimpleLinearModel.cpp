// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <ModelTest/SimpleLinearModel.hpp>
#include <Sapphire/Model.hpp>
#include <Sapphire/operations/Forward/Linear.hpp>
#include <Sapphire/operations/Forward/ReLU.hpp>
#include <Sapphire/operations/Loss/MSE.hpp>
#include <Sapphire/operations/Loss/CrossEntropy.hpp>
#include <Sapphire/operations/optimizers/SGD.hpp>
#include <Sapphire/operations/Forward/Softmax.hpp>
#include <Sapphire/util/FileManager.hpp>
#include <Sapphire/util/ResourceManager.hpp>
#include <iostream>


namespace Sapphire::Test
{
void SimpleLinearModel(float learningRate, int epochs, bool hostMode)
{
    ModelManager::AddModel("SimpleLinearModel");
    ModelManager::SetCurrentModel("SimpleLinearModel");

    const CudaDevice gpu(0, "cuda0");

    const auto totalData = ReadFile<std::uint8_t>(std::string(
        "/mnt/c/Users/user/Documents/Sapphire/Datasets/cifar-10-batches-bin/"
        "data_batch_1.bin"));

    NN::Linear linear(32 * 32 * 3, 32 * 32);
    NN::Linear fc1(32 * 32, 32);
    NN::Linear fc2(32, 10);

    // Tensor weight(Shape({ inputSize, outputSize }), gpu, Type::Dense, true);
    // Tensor weight1(Shape({ outputSize, outputSize }), gpu, Type::Dense, true);
    //
    // Tensor bias(Shape({ 1, outputSize }), gpu, Type::Dense, true);
    // Tensor bias1(Shape({ 1, outputSize }), gpu, Type::Dense, true);
    // Initialize::Initialize(weight,
    //                        std::make_unique<Initialize::Normal>(0.0f, 0.01f));
    // Initialize::Initialize(weight1,
    //                        std::make_unique<Initialize::Normal>(0.0f, 0.01f));
    // Initialize::Initialize(bias,
    //                        std::make_unique<Initialize::Normal>(0.0f, 0.01f));
    // Initialize::Initialize(bias1,
    //                        std::make_unique<Initialize::Normal>(0.0f, 0.01f));

    // if (hostMode)
    // {
    //     weight.ToHost();
    //     weight1.ToHost();
    //     bias.ToHost();
    //     bias1.ToHost();
    // }

    Tensor x(Shape({ 1, 32 * 32 * 3 }), gpu, Type::Dense, true);
    Tensor label(Shape({ 1, 10 }), gpu, Type::Dense, true);

    if (hostMode)
    {
        x.ToHost();
        label.ToHost();
    }

    Optimizer::SGD sgd(learningRate);
    ModelManager::CurModel().SetOptimizer(&sgd);

    std::vector<float> labelData(10);
    std::vector<float> xData(32 * 32 * 3);

    for (int i = 0; i < epochs; ++i)
    {
        std::fill(labelData.begin(), labelData.end(), 0.0f);
        labelData[totalData.at(i * (32 * 32 * 3 + 1))] = 1.0f;
        for (int idx = 0; idx < 32 * 32 * 3; ++idx)
            xData[idx] = totalData.at(i * (32 * 32 * 3 + 1) + idx + 1);

        x.LoadData(xData);
        label.LoadData(labelData);

        auto tensor = NN::ReLU(linear(x));
        tensor = NN::ReLU(fc1(tensor));
        tensor = NN::ReLU(fc2(tensor));
        const auto yData = tensor.GetData();
        for (const auto& elem : yData)
            std::cout << elem << " ";
        std::cout << std::endl;
        //y = NN::ReLU(linear(y));
        const auto loss = NN::Loss::CrossEntropy(tensor, label);
        if (i % 10 == 0)
        {
            const auto lossData = loss.GetData();
            std::cout << "epoch: " << i << " loss : " << lossData[0]
                << std::endl;
        }
        //ModelManager::CurModel().InitGradient();
        ModelManager::CurModel().BackProp(loss);
        ModelManager::CurModel().Clear();
        if (i % 10 == 0)
            Util::ResourceManager::Clean();
    }
    Util::ResourceManager::ClearAll();
}
}
