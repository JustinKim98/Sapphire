// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_UTIL_TENSOR_DECL_HPP
#define MOTUTAPU_UTIL_TENSOR_DECL_HPP

#include <Motutapu/tensor/Shape.hpp>
#include <Motutapu/util/Device.hpp>
#include <Motutapu/util/TensorDataDecl.hpp>
#include <vector>
#include <list>

namespace Motutapu
{
//! TensorData class contains data vector for processing
//! with attributes describing it
template <typename T>
class Tensor
{
 public:
    Tensor() = default;

    Tensor(Shape shape, Device device);

    ~Tensor();

    Tensor(const Tensor<T>& tensor);
    Tensor(Tensor<T>&& tensor) noexcept = delete;
    /// move assignment operator
    Tensor<T>& operator=(const Tensor<T>& tensor);
    Tensor<T>& operator=(Tensor<T>&& tensor) noexcept = delete;

    void ToGPU();
    void ToCPU();

    void ToSparse();
    void ToDense();

    std::vector<T> Data();
    Util::TensorData<T>& TensorData();
    Shape GetShape();

    void PushTrajectory(int operationId);
    int PopTrajectory();
    int PeekTrajectory();

 private:
    int m_tensorId;
    Device m_device;

    std::list<int> m_functionTrajectory;

    void m_freeData();
};

}

#endif