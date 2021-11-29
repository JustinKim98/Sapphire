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
#include <iostream>

namespace Sapphire::Test
{
void Conv2DModel(std::vector<float> xData, std::vector<float> labelData,
                 int batchSize,
                 std::pair<int, int> dilation, float learningRate,
                 bool hostMode, int epochs)
{
    ModelManager::AddModel("SimpleConv2DModel");
    ModelManager::SetCurrentModel("SimpleConv2DModel");

    const CudaDevice gpu(0, "cuda0");

    //! Declare conv2d Layer
    NN::Conv2D conv0(6, 3, std::make_pair(5, 5), std::pair(1, 1),
                     std::pair(0, 0), std::pair(1, 1), true);
    NN::MaxPool2D pool(6, std::make_pair(2, 2), std::make_pair(2, 2));
    NN::Conv2D conv1(16, 6, std::make_pair(5, 5), std::pair(1, 1),
                     std::make_pair(0, 0),
                     dilation,

                     true
        );
    NN::Linear fc0(16 * 5 * 5, 120);
    NN::Linear fc1(120, 84);
    NN::Linear fc2(84, 10);

    //! Declare input tensors
    Tensor convFilter0 = MakeTensor(Shape({ 5, 5 }), gpu,
                                    M<Initialize::Normal>(
                                        0.0f, 1.0f), true);
    Tensor convBias0 = MakeTensor(Shape({ 6 }), gpu,
                                  M<Initialize::Normal>(0.0f, 1.0f),
                                  true);
    Tensor convFilter1 =
        MakeTensor(Shape({ 5, 5 }), gpu,
                   M<Initialize::Normal>(0.0f, 1.0f), true);
    Tensor convBias1 =
        MakeTensor(Shape({ 16 }), gpu, M<Initialize::Normal>(0.0f, 1.0f), true);
    Tensor fcWeight0 = MakeTensor(Shape({ 16 * 5 * 5, 120 }), gpu,
                                  M<Initialize::Normal>(0.0f, 1.0f),
                                  true);
    Tensor fcBias0 = MakeTensor(Shape({ 120 }), gpu,
                                M<Initialize::Normal>(0.0f, 1.0f),
                                true);
    Tensor fcWeight1 =
        MakeTensor(Shape({ 120, 84 }), gpu,
                   M<Initialize::Normal>(0.0f, 1.0f), true);
    Tensor fcBias1 =
        MakeTensor(Shape({ 84 }), gpu,
                   M<Initialize::Normal>(0.0f, 1.0f), true);
    Tensor fcWeight2 = MakeTensor(Shape({ 84, 10 }), gpu,
                                  M<Initialize::Normal>(0.0f, 1.0f),
                                  true);
    Tensor fcBias2 = MakeTensor(Shape({ 10 }), gpu,
                                M<Initialize::Normal>(0.0f, 1.0f), true);

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

    //! Load the data to model
    x.LoadData(xData);
    label.LoadData(labelData);

    Optimizer::SGD sgd(learningRate);
    ModelManager::CurModel().SetOptimizer(&sgd);

    for (int i = 0; i < epochs; ++i)
    {
        //! Load data to x and label here

        auto tensor = pool(NN::ReLU(conv0(x, convFilter0, convBias0)));
        tensor = pool(NN::ReLU(conv1(tensor, convFilter1, convBias1)));

        tensor = NN::ReLU(fc0(tensor, fcWeight0, fcBias0));
        tensor = NN::ReLU(fc1(tensor, fcWeight1, fcBias1));
        auto out = fc2(tensor, fcWeight2, fcBias2);

        auto loss = NN::Loss::CrossEntropy(out, label);

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
