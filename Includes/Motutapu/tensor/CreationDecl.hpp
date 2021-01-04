// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_CREATION_DECL_HPP
#define MOTUTAPU_CREATION_DECL_HPP

#include <Motutapu/tensor/TensorDecl.hpp>
#include <Motutapu/tensor/Shape.hpp>

namespace Motutapu
{
//! Creates Empty tensor
//! \tparam T : data type of the tensor
//! \param shape : shape of the new tensor
//! \param device : device to locate new tensor
template <typename T>
Tensor<T> Empty(Shape shape, Device device);

//! Creates identity matrix
//! \tparam T : data type of the tensor
//! \param size : row/column size of the new tensor
//! \param device : device to locate new tensor
template <typename T>
Tensor<T> Eye(uint32_t size, Device device);

template <typename T>
Tensor<T> Full(T value, Shape shape, Device device);

template<typename T>
Tensor<T> Ones(Shape, Device device);

template<typename T>
Tensor<T> Zeros(Shape, Device device);

template<typename T>
Tensor<T> Rand(Shape, T min, T max, Device device);

template<typename T>
Tensor<T> RandNormal(Shape, T mean, T sd, Device device);


}

#endif