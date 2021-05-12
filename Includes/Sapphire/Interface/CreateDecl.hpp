// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Sapphire_CREATION_DECL_HPP
#define Sapphire_CREATION_DECL_HPP

#include <Sapphire/tensor/TensorDecl.hpp>
#include <Sapphire/tensor/Shape.hpp>
#include <Sapphire/operations/Unit.hpp>
#include <cstdlib>

namespace Sapphire
{
//! \tparam T : data type of the tensor
template <typename T>
class Eye : Unit<T>
{
public:
    //! Creates Identity matrix
    //! \param size : row and column of the identity matrix
    //! \param device : device to locate output tensor
    //! \param type : Type of the data (Dense or Sparse)
    Eye(int size, Device device, Type type);

    //! Computes forward propagation
    Tensor<T> Forward();

    size_t BatchSize = 0;

private :
    Shape m_shape;
};

template <typename T>
Tensor<T> Full(T value, Shape shape, Device device);

template <typename T>
Tensor<T> Ones(Shape, Device device);

template <typename T>
Tensor<T> Zeros(Shape, Device device);

template <typename T>
Tensor<T> Rand(Shape, T min, T max, Device device);

template <typename T>
Tensor<T> RandNormal(Shape, T mean, T sd, Device device);

template <typename T>
Tensor<T> HeNormal(Shape, T mean, T sd, Device device);
}

#endif
