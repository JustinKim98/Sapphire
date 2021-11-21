// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <GraphTest/GraphFunctionalityTest.hpp>
#include <Sapphire/operations/Forward/Basic.hpp>
#include <Sapphire/operations/optimizers/SGD.hpp>
#include <Sapphire/Model.hpp>
#include <doctest.h>


namespace Sapphire::Test
{
void GraphFunctionalityTest()
{
    ModelManager::AddModel("BasicModel");
    ModelManager::SetCurrentModel("BasicModel");

    const CudaDevice gpu(0, "cuda0");

    NN::TwoOutputs twoOutputs;
    NN::Basic hidden1;
    NN::Basic hidden2;
    NN::TwoInputs twoInputs;
    NN::InplaceOp inplace1;
    NN::Basic output;

    Tensor x(Shape({ 10 }), gpu, Type::Dense);
    Initialize::Initialize(x, std::make_unique<Initialize::Ones>());

    auto [x11, x12] = twoOutputs(x);
    auto x21 = hidden1(x11);
    auto x22 = hidden2(x12);
    auto x31 = twoInputs(x21, x22);
    inplace1(x31);
    auto y = output(x31);

    Optimizer::SGD sgd(0.0f);
    ModelManager::CurModel().SetOptimizer(&sgd);

    ModelManager::CurModel().BackProp(y);
    ModelManager::CurModel().Clear();
}
}
