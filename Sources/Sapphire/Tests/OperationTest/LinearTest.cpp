// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Tests/OperationTest/LinearTest.hpp>
#include <Sapphire/Model.hpp>
#include <Sapphire/operations/Forward/Linear.hpp>
#include <Sapphire/operations/optimizers/SGD.hpp>

namespace Sapphire::Test::Operation
{
void LinearForwardTest()
{
    ModelManager::AddModel("myModel");
    ModelManager::SetCurrentModel("myModel");

    const Device gpu(0, "cuda0");
    const Device host("cpu");

    NN::Linear linear(200, 200, Util::SharedPtr<Optimizer::SGD>::Make(0.1f),
                      std::make_unique<Initialize::Ones>(),
                      std::make_unique<Initialize::Ones>(),
                      gpu);

    Tensor input(Shape({ 200 }), 1, host, Type::Dense);
    Initialize::Initialize(input, std::make_unique<Initialize::Ones>());
    input.SendTo(gpu);
    const auto output = linear(input);
    output.SendTo(host);
    output.SendTo(gpu);
    ModelManager::GetCurrentModel().BackProp(output);
    output.SendTo(host);

    ModelManager::GetCurrentModel().Clear();
}
}
