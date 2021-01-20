// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_CREATION_DECL_HPP
#define MOTUTAPU_CREATION_DECL_HPP

#include <Motutapu/tensor/TensorDecl.hpp>
#include <Motutapu/tensor/Shape.hpp>
#include <Motutapu/operations/Unit.hpp>
#include <cstdlib>

namespace Motutapu
{

//! \tparam T : data type of the tensor
template <typename T>
class Empty : Unit<T>
{
public:
    //! Creates Empty tensor
    //! \param shape : shape of the new tensor
    //! \param device : device to locate output tensor
    //! \param sparse : true if sparse false otherwise
    Empty(Shape shape, Device device, bool sparse);

    Tensor<T> Forward();

    size_t BatchSize = 0;
private:
    Shape m_shape;
};

//! \tparam T : data type of the tensor
template <typename T>
class Eye : Unit<T>
{
public:
    //! Creates Identity matrix
    //! \param size : row and column of the identity matrix
    //! \param device : device to locate output tensor
    //! \param sparse : true if sparse false otherwise
    Eye(int size, Device device, bool sparse);

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
}

#endif
