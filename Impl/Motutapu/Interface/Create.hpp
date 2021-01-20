// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_CREATE_HPP
#define MOTUTAPU_CREATE_HPP

#include <Motutapu/Interface/CreateDecl.hpp>
#include <Motutapu/tensor/TensorData.hpp>

namespace Motutapu
{
template <typename T>
Empty<T>::Empty(Shape shape, Device device, bool sparse)
    : Unit<T>()
{
    Unit<T>::OutputTensorMap["Out"] = Tensor<T>(shape, device);
}

template <typename T>
Tensor<T> Empty<T>::Forward()
{
    return Unit<T>::OutputTensorMap["Out"];
    //! Call to initialization kernel
}
}

#endif
