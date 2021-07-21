// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_LINEARBACKWARD_HPP
#define SAPPHIRE_LINEARBACKWARD_HPP

#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/operations/Backward/BackPropWrapper.hpp>

namespace Sapphire::BackProp
{
constexpr static int dxIdx = 0;
constexpr static int dyIdx = 0;
constexpr static int weightIdx = 0;
constexpr static int biasIdx = 0;

class LinearBackProp : public BackPropWrapper
{
public:
    explicit LinearBackProp(TensorUtil::TensorData& dx,
                            TensorUtil::TensorData& weight,
                            TensorUtil::TensorData& bias,
                            const TensorUtil::TensorData& dy,
                            const TensorUtil::TensorData& x,
                            std::weak_ptr<Optimizer::Optimizer> optimizer,
                            unsigned int batchSize, int unitKey);

    bool InvokeBackProp(const TensorUtil::TensorData& dy) override;

private:
    void m_backProp(const TensorUtil::TensorData& weight);

    void m_updateWeight(TensorUtil::TensorData& weight);

    void m_updateBias(TensorUtil::TensorData& bias);

    unsigned int m_batchSize;
};
} // namespace Sapphire::BackProp

#endif  // Sapphire_LINEARBACKWARD_HPP
