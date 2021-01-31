// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_BACKPROPWRAPPER_HPP
#define MOTUTAPU_BACKPROPWRAPPER_HPP

#include <Motutapu/tensor/TensorData.hpp>
#include <functional>

namespace Motutapu::BackProp
{
class BackPropWrapper
{
 public:
    BackPropWrapper() = default;
    virtual ~BackPropWrapper() = default;

    explicit BackPropWrapper(std::vector<unsigned int> gradientOutputKeys,
                             bool inplace)
        : m_gradientOutputKeys(std::move(gradientOutputKeys)),
          m_inplace(inplace)
    {
    }

    BackPropWrapper(std::vector<unsigned int> gradientOutputKeys, bool inplace,
                    int unitKey)
        : m_gradientOutputKeys(std::move(gradientOutputKeys)),
          m_inplace(inplace),
          m_unitKey(unitKey)
    {
    }

    [[nodiscard]] bool IsInplace() const
    {
        return m_inplace;
    }

    [[nodiscard]] const std::vector<unsigned int>& GetOutputTensorKeys() const
    {
        return m_gradientOutputKeys;
    }

    virtual void Backward(std::vector<TensorUtil::TensorData>& output,
                          const TensorUtil::TensorData& input) const = 0;

 protected:
    //! Vector of tensorData that should give its output
    std::vector<unsigned int> m_gradientOutputKeys;
    bool m_inplace = false;
    int m_unitKey = -1;
};
}  // namespace Motutapu::BackProp

#endif
