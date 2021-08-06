// Copyright (c) 2021, Justin Kim
// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_INITIALIZER_INITIALIZER_HPP
#define SAPPHIRE_INITIALIZER_INITIALIZER_HPP

#include <Sapphire/tensor/TensorData.hpp>
#include <Sapphire/compute/Initialize.hpp>

namespace Sapphire::Initialize
{
using namespace TensorUtil;

class Initializer
{
public:
    Initializer() = default;
    virtual ~Initializer() = default;
    Initializer(const Initializer& initializer) = default;
    Initializer(Initializer&& initializer) noexcept = default;
    Initializer& operator=(const Initializer& initializer) = default;
    Initializer& operator=(Initializer&& initializer) noexcept = default;

    virtual void operator()(TensorData& tensorData)
    {
        throw std::runtime_error(
            "Optimizer::Optimizer::operator() -Default operator should not be "
            "called");
    }
};

class Zeros : public Initializer
{
public:
    Zeros()
        : Initializer()
    {
    }

    void operator()(TensorData& tensorData) override
    {
        Compute::Initialize::Zeros(tensorData);
    }
};

class Ones : public Initializer
{
public:
    Ones()
        : Initializer()
    {
    }

    void operator()(TensorData& tensorData) override
    {
        Compute::Initialize::Ones(tensorData);
    }
};

class Normal : public Initializer
{
public:
    Normal(float mean, float sd)
        : Initializer(),
          m_mean(mean),
          m_sd(sd)
    {
    }

    void operator()(TensorData& tensorData) override
    {
        Compute::Initialize::Normal(tensorData, m_mean, m_sd);
    }

private:
    float m_mean,
          m_sd;
};

class HeNormal : public Initializer
{
public:
    HeNormal(int fanIn)
        : m_fanIn(fanIn)
    {
    }

    void operator()(TensorData& tensorData) override
    {
        Compute::Initialize::HeNormal(tensorData, m_fanIn);
    }

private:
    int m_fanIn;
};

inline void Initialize(Tensor& tensor, std::unique_ptr<Initializer> initializer)
{
    auto& desc = ModelManager::GetCurrentModel().GetDescriptor(
        tensor.TensorDescriptorKey());
    auto forwardData = desc.GetForwardData();
    initializer->operator()(forwardData);
}
}

#endif
