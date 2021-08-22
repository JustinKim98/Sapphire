// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_BACKPROP_EXAMPLE_BACKWARD_HPP
#define SAPPHIRE_BACKPROP_EXAMPLE_BACKWARD_HPP

#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/operations/Backward/BackPropWrapper.hpp>

namespace Sapphire::BackProp
{
class BasicBackward : public BackPropWrapper
{
public:
    explicit BasicBackward(TensorUtil::TensorData dx,
                             TensorUtil::TensorData dy);

private:
    void m_runBackProp() override;
};

class BackwardTwoInputs : public BackPropWrapper
{
public:
    explicit BackwardTwoInputs(TensorUtil::TensorData dx1,
                               TensorUtil::TensorData dx2,
                               TensorUtil::TensorData dy);

private:
    void m_runBackProp() override;
};

class BackwardTwoOutputs : public BackPropWrapper
{
public:
    explicit BackwardTwoOutputs(TensorUtil::TensorData dx,
                                TensorUtil::TensorData dy1,
                                TensorUtil::TensorData dy2);

private:
    void m_runBackProp() override;
};
}

#endif
