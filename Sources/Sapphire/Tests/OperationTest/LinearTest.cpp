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

    const Device device(0, "cuda0");
    NN::Linear linear(3, 3, Util::SharedPtr<Optimizer::SGD>::Make(0.1f),
                      device);

    const Tensor input(Shape({ 3 }), 1, device, Type::Dense);
    const auto output = linear(input);
    ModelManager::GetCurrentModel().BackProp(output);
    ModelManager::GetCurrentModel().Clear();
}
}
