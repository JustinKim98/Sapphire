// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <OperationTest/MSETest.hpp>
#include <Sapphire/operations/Initializers/Initialize.hpp>
#include <Sapphire/Model.hpp>
#include <Sapphire/operations/Loss/MSE.hpp>
#include <iostream>

namespace Sapphire::Test
{
void TestMSE(bool print)
{
    ModelManager::AddModel("myModel");
    ModelManager::SetCurrentModel("myModel");

    const CudaDevice gpu(0, "cuda0");

    const Shape xShape = Shape({ 10 });

    Tensor x(xShape, gpu, Type::Dense);
    Tensor label(xShape, gpu, Type::Dense);

    Initialize::Initialize(x, std::make_unique<Initialize::Ones>());

    const auto loss = NN::Loss::MSE(x, label);

    const auto lossShape = loss.GetShape();
    const auto lossData = loss.GetForwardDataCopy();

    if (print)
    {
        std::cout << "Loss " << std::endl;
        for (int i = 0; i < lossShape.Size(); ++i)
            std::cout << lossData[i] << std::endl;
    }

    ModelManager::GetCurrentModel().Clear();
}
}
