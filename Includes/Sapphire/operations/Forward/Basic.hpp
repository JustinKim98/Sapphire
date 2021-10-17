// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_NN_EXAMPLE_HPP
#define SAPPHIRE_NN_EXAMPLE_HPP

#include <Sapphire/operations/Unit.hpp>
#include <Sapphire/tensor/Tensor.hpp>
#include <Sapphire/operations/optimizers/Optimizer.hpp>
#include <Sapphire/operations/Initializers/Initialize.hpp>


namespace Sapphire::NN
{
class Basic : public Unit
{
public:
    Basic() = default;
    ~Basic() override = default;

    Basic(const Basic& example) = default;
    Basic(Basic&& example) noexcept = default;
    Basic& operator=(const Basic& example) = default;
    Basic& operator=(Basic&& example) noexcept = default;

    Tensor operator()(Tensor& xTensor);

private:
    void m_checkArguments(
        std::vector<TensorUtil::TensorDescriptor*> arguments) const override
    {
    }
};

class TwoInputs : public Unit
{
public:
    TwoInputs() = default;
    ~TwoInputs() override = default;

    TwoInputs(const TwoInputs& receiveTwoInputs) = default;
    TwoInputs(TwoInputs&& receiveTwoInputs) noexcept = default;
    TwoInputs& operator=(const TwoInputs& receiveTwoInputs)
    = default;
    TwoInputs& operator=(TwoInputs&& receiveTwoInputs) noexcept
    = default;

    Tensor operator()(Tensor& x1Tensor, Tensor& x2Tensor);

private:
    void m_checkArguments(
        std::vector<TensorUtil::TensorDescriptor*> arguments) const override
    {
    }
};

class TwoOutputs : public Unit
{
public:
    TwoOutputs() = default;
    ~TwoOutputs() override = default;

    TwoOutputs(const TwoOutputs& twoOutputs) = default;
    TwoOutputs(TwoOutputs&& twoOutputs) noexcept = default;
    TwoOutputs& operator=(const TwoOutputs& twoOutputs) = default;
    TwoOutputs& operator=(TwoOutputs&& twoOutputs) noexcept = default;

    std::pair<Tensor, Tensor> operator()(Tensor& xTensor);

private:
    void m_checkArguments(
        std::vector<TensorUtil::TensorDescriptor*> arguments) const override
    {
    }
};

class InplaceOp : public Unit
{
public:
    InplaceOp() = default;
    ~InplaceOp() override = default;

    InplaceOp(const InplaceOp& inplaceOp) = default;
    InplaceOp(InplaceOp&& inplaceOp) noexcept = default;
    InplaceOp& operator=(const InplaceOp& inplaceOp) = default;
    InplaceOp& operator=(InplaceOp&& inplaceOp) noexcept = default;

    void operator()(Tensor& xTensor);

private:
    void m_checkArguments(
        std::vector<TensorUtil::TensorDescriptor*> arguments) const override
    {
    }
};
}

#endif
