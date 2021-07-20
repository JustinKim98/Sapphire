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
class LinearBackProp : public BackPropWrapper
{
public:
    explicit LinearBackProp(const TensorUtil::TensorData& x,
                            TensorUtil::TensorData dx,
                            TensorUtil::TensorData dy, int unitKey);

    bool InvokeBackProp(const TensorUtil::TensorData& dy) override;

private:
    void m_backProp(const TensorUtil::TensorData& weight);

    void m_updateWeight(TensorUtil::TensorData& weight);

    void m_updateBias(TensorUtil::TensorData& bias);

    unsigned int m_batchSize;
};
} // namespace Sapphire::BackProp

#endif  // Sapphire_LINEARBACKWARD_HPP
