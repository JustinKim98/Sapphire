// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_NN_LINEAR_HPP
#define SAPPHIRE_NN_LINEAR_HPP

#include <Sapphire/tensor/Tensor.hpp>
#include <Sapphire/operations/optimizers/Optimizer.hpp>
#include <Sapphire/operations/Unit.hpp>
#include <Sapphire/operations/Initializers/Initialize.hpp>

namespace Sapphire::NN
{
class Linear : public Unit
{
public:
    Linear(int inputFeatureSize, int outputFeatureSize,
           bool isSparse = false);
    Linear(std::string name, int inputFeatureSize, int outputFeatureSize,
           bool isSparse = false);

    ~Linear() override = default;

    Linear(const Linear& linear) = default;
    Linear(Linear&& linear) noexcept = default;
    Linear& operator=(const Linear& linear) = default;
    Linear& operator=(Linear&& linear) noexcept = default;

    Tensor operator()(Tensor& x);
    Tensor operator()(Tensor& x, Tensor weight, Tensor bias);

    Tensor GetWeight() const;
    Tensor GetBias() const;

protected:
    void m_addTensorData(std::string name, TensorUtil::TensorData tensorData)
    {
        m_tensorDataMap[name] = tensorData;
    }

    [[nodiscard]] bool m_exists(std::string name) const
    {
        if (m_tensorDataMap.find(name) == m_tensorDataMap.end())
            return false;
        return true;
    }

    TensorUtil::TensorData& m_getTensorData(std::string name)
    {
        return m_tensorDataMap.at(name);
    }

    std::unordered_map<std::string, TensorUtil::TensorData> m_tensorDataMap;

private:
    [[nodiscard]] int m_registerOutputTensor(
        const TensorUtil::TensorDescriptor& xDesc) const;

    void m_checkArguments(
        std::vector<TensorUtil::TensorDescriptor*> arguments) const override;

    static int m_unitIdCount;
    int m_inputs;
    int m_outputs;
    CudaDevice m_device;
    bool m_isSparse;
    Tensor m_weight, m_bias;
};
} // namespace Sapphire::NN

#endif  // Sapphire_LINEAR_HPP
