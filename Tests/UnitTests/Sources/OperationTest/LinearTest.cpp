// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.


#include <OperationTest/LinearTest.hpp>
#include <Sapphire/Model.hpp>
#include <Sapphire/operations/Forward/Linear.hpp>
#include <Sapphire/operations/optimizers/SGD.hpp>
#include <iostream>
#include <doctest/doctest.h>

namespace Sapphire::Test
{
void TestLinear()
{
    ModelManager::AddModel("myModel");
    ModelManager::SetCurrentModel("myModel");

    const CudaDevice gpu(0, "cuda0");

    NN::Linear linear(200, 200, Util::SharedPtr<Optimizer::SGD>::Make(0.1f),
                      std::make_unique<Initialize::Ones>(),
                      std::make_unique<Initialize::Ones>(), gpu);

    Tensor input(Shape({ 200 }), gpu, Type::Dense);
    Initialize::Initialize(input, std::make_unique<Initialize::Ones>());
    input.ToCuda();
    auto output = linear(input);
    output.ToHost();

    const auto forwardDataPtr = output.GetForwardDataCopy();
    for (int i = 0; i < output.GetShape().Size(); ++i)
        CHECK(std::abs(201.0f - forwardDataPtr[i]) <
        std::numeric_limits<float>::epsilon());

    InitializeBackwardData(output, std::make_unique<Initialize::Ones>());

    output.ToCuda();
    ModelManager::GetCurrentModel().BackProp(output);

    input.ToHost();
    const auto backwardDataPtr = input.GetBackwardDataCopy();
    for (int i = 0; i < input.GetShape().Size(); ++i)
        CHECK(std::abs(200.0f - backwardDataPtr[i]) <
        std::numeric_limits<float>::epsilon());

    ModelManager::GetCurrentModel().Clear();
}
} // namespace Sapphire::Test
