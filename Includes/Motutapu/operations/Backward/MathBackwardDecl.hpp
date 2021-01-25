// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.


#ifndef MOTUTAPU_BACKWARD_MATHBACKWARD_DECL_HPP
#define MOTUTAPU_BACKWARD_MATHBACKWARD_DECL_HPP

#include <Motutapu/operations/Backward/BackPropWrapper.hpp>
#include <vector>

namespace Motutapu::BackProp
{
template <typename T>
class MulBackProp : public BackPropWrapper<T>
{
public:
    MulBackProp(std::vector<int> outputTensorKeys)
        : BackPropWrapper<T>(outputTensorKeys)
    {
    }

    void Backward(std::vector<Util::TensorData<T>>& output,
                  const Util::TensorData<T>& input) const override
    {
        //! todo : Implement backward function for MulBackProp
    }
};
}

#endif
