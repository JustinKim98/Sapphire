// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_NN_LINEAR_HPP
#define SAPPHIRE_NN_LINEAR_HPP

#include <Sapphire/tensor/Tensor.hpp>
#include <Sapphire/operations/optimizers/Optimizer.hpp>
#include <Sapphire/util/SharedPtr.hpp>
#include <Sapphire/operations/Unit.hpp>
#include <Sapphire/operations/Initializers/Initialize.hpp>

namespace Sapphire::NN
{
class Linear : public Unit
{
public:
    Linear(int inputFeatureSize, int outputFeatureSize,
           Util::SharedPtr<Optimizer::Optimizer> optimizer,
           CudaDevice device = CudaDevice(),
           bool isSparse = false);
    ~Linear() override = default;

    Linear(const Linear& linear) = default;
    Linear(Linear&& linear) noexcept = default;
    Linear& operator=(const Linear& linear) = default;
    Linear& operator=(Linear&& linear) noexcept = default;

    Tensor operator()(Tensor& x, Tensor weight, Tensor bias);

protected:
    void m_addTensorData(std::string name, TensorUtil::TensorData tensorData);

    std::unordered_map<std::string, TensorUtil::TensorData> m_tensorDataMap;

private:
    [[nodiscard]] int m_registerOutputTensor(
        const TensorUtil::TensorDescriptor& xDesc) const;

    void m_checkArguments(
        std::vector<TensorUtil::TensorDescriptor*> arguments) const override;

    int m_inputs;
    int m_outputs;
    Util::SharedPtr<Optimizer::Optimizer> m_optimizer;
    CudaDevice m_device;
    bool m_isSparse;
};
} // namespace Sapphire::NN

#endif  // Sapphire_LINEAR_HPP
