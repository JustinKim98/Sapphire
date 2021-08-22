// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <iostream>
#include <Sapphire/Model.hpp>
#include <Sapphire/operations/Backward/BasicBackward.hpp>

namespace Sapphire::BackProp
{
BasicBackward::BasicBackward(TensorUtil::TensorData dx,
                             TensorUtil::TensorData dy)
    : BackPropWrapper({ std::move(dx) }, {
                          std::move(dy) })
{
}

void BasicBackward::m_runBackProp()
{
    std::cout << "BackProp called" << std::endl;
}

BackwardTwoInputs::BackwardTwoInputs(TensorUtil::TensorData dx1,
                                     TensorUtil::TensorData dx2,
                                     TensorUtil::TensorData dy)
    : BackPropWrapper({ std::move(dx1), std::move(dx2) }, { std::move(dy) })
{
}

void BackwardTwoInputs::m_runBackProp()
{
    std::cout << "BackProp Two inputs  called" << std::endl;
}

BackwardTwoOutputs::BackwardTwoOutputs(TensorUtil::TensorData dx,
                                       TensorUtil::TensorData dy1,
                                       TensorUtil::TensorData dy2)
    : BackPropWrapper({ std::move(dx) }, { std::move(dy1), std::move(dy2) })
{
}

void BackwardTwoOutputs::m_runBackProp()
{
    std::cout << "BackProp Two outputs called" << std::endl;
}

BackwardInplace::BackwardInplace(TensorUtil::TensorData dx,
                                 TensorUtil::TensorData dy)
    : BackPropWrapper({ std::move(dx) }, { std::move(dy) })
{
}

void BackwardInplace::m_runBackProp()
{
    std::cout << "BackProp Inplace called" << std::endl;
}
}
