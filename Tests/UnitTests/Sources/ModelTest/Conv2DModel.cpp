// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <ModelTest/Conv2DModel.hpp>
#include <Sapphire/Model.hpp>
#include <Sapphire/operations/Forward/Linear.hpp>
#include <Sapphire/operations/Forward/Conv2D.hpp>
#include <Sapphire/operations/Forward/ReLU.hpp>
#include <Sapphire/operations/Loss/MSE.hpp>
#include <Sapphire/operations/optimizers/SGD.hpp>
#include <Sapphire/util/ResourceManager.hpp>
#include <Sapphire/operations/Forward/MaxPool2D.hpp>
#include <iostream>

namespace Sapphire::Test
{
void Conv2DModel(std::vector<float> xData, std::vector<float> labelData,
                 int batchSize,
                 int yChannels, int xChannels, std::pair<int, int> xSize,
                 std::pair<int, int> ySize,
                 std::pair<int, int> filterSize, std::pair<int, int> stride,
                 std::pair<int, int> padSize,
                 std::pair<int, int> dilation, float learningRate,
                 bool hostMode, int epochs)
{
    ModelManager::AddModel("SimpleConv2DModel");
    ModelManager::SetCurrentModel("SimpleConv2DModel");

    const auto [filterRows, filterCols] = filterSize;
    const auto [xRows, xCols] = xSize;
    const auto [yRows, yCols] = ySize;

    const CudaDevice gpu(0, "cuda0");

    //! Declare conv2d Layer
    NN::Conv2D conv1(6, 3, std::make_pair(5, 5), stride, padSize,
                     dilation, true);
    NN::MaxPool2D pool1(yChannels, std::make_pair(2, 2), std::make_pair(2, 2));
    NN::Conv2D conv2(16, 6, std::make_pair(5, 5), stride, padSize,
                     dilation,
                     true);
    NN::Linear fc1(16 * 5 * 5, 120);
    NN::Linear fc2(120, 84);
    NN::Linear fc3(84, 10);

    //! Declare input tensors
    Tensor filter(Shape({ yChannels, xChannels, filterRows, filterCols }), gpu,
                  Type::Dense, true);

    Tensor bias(Shape({ yChannels }), gpu, Type::Dense, true);

    Tensor x(Shape({ batchSize, xChannels, xRows, xCols }), gpu, Type::Dense,
             true);
    Tensor label(Shape({ batchSize, yChannels, yRows, yCols }), gpu,
                 Type::Dense, true);

    //! Initialize weights to arbitrary values
    Initialize::Initialize(filter,
                           std::make_unique<Initialize::Normal>(0.0f, 0.01f));
    Initialize::Initialize(bias,
                           std::make_unique<Initialize::Normal>(0.0f, 0.01f));

    //! Configure data to be on host if needed
    if (hostMode)
    {
        filter.ToHost();
        bias.ToHost();
        x.ToHost();
        label.ToHost();
    }

    //! Load the data to model
    x.LoadData(xData);
    label.LoadData(labelData);

    Optimizer::SGD sgd(learningRate);
    ModelManager::CurModel().SetOptimizer(&sgd);

    for (int i = 0; i < epochs; ++i)
    {
        auto y = NN::ReLU(conv1(x, filter, bias));
        y = NN::ReLU(y);
        const auto loss = NN::Loss::MSE(y, label);

        //! Print loss every 10 epochs
        if (i % 10 == 0)
        {
            const auto lossData = loss.GetData();
            std::cout << "epoch: " << i << " loss : " << lossData[0]
                << std::endl;
        }

        //! Start back propagation and update weights
        ModelManager::CurModel().BackProp(loss);
        //! Clear the gradients for next back propagation
        ModelManager::CurModel().Clear();

        //! Clear resources for every 10 epochs
        if (i % 10 == 0)
            Util::ResourceManager::Clean();
    }

    Util::ResourceManager::ClearAll();
}
} // namespace Sapphire::Test
