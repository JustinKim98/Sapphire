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

    BackPropWrapper(std::vector<int> outputTensorKeys)
        : m_outputTensorKeys(std::move(outputTensorKeys))
    {
    }


    BackPropWrapper(std::vector<int> outputTensorKeys, int unitKey)
        : m_outputTensorKeys(std::move(outputTensorKeys)),
          m_unitKey(unitKey)
    {
    }

    [[nodiscard]] const std::vector<int>& GetOutputTensorKeys() const
    {
        return m_outputTensorKeys;
    }

    virtual void Backward(std::vector<Util::TensorData>& output, const
                          Util::TensorData& input) const = 0;

protected :
    //! Vector of tensorData that should give its output
    std::vector<int> m_outputTensorKeys;
    int m_unitKey = -1;
};
}

#endif
