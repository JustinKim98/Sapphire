// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_UNIT_DECL_HPP
#define MOTUTAPU_UNIT_DECL_HPP

#include <unordered_map>
#include <Motutapu/tensor/TensorDecl.hpp>

namespace Motutapu
{
    template <typename T>
    class Unit
    {
     public:
        Unit() = default;
        virtual ~Unit() = default;

        Unit(const Unit<T>& unit) = default;
        Unit<T>& operator=(const Unit& unit) = default;

        std::unordered_map<std::string, Tensor<T>> OutputTensorMap;
        std::unordered_map<std::string, Tensor<T>> InputTensorMap;
        std::unordered_map<std::string, Tensor<T>> InternalTensorMap;

        std::unordered_map<std::string, std::string> StringLiterals;
        std::unordered_map<std::string, T> ScalarLiterals;
        std::unordered_map<std::string, int> IntegerLiterals;

        //! Pushes tensor into ForwardInputTensorPool
        //! Invokes forward propagation if tensor is ready
        //! Returns output vector of tensors if it was invoked and execution was
        //! finished successfully
        //! \param tensor : tensor to be used in backward propagation
        //! \return : vector of output tensors if function was invoked
        virtual std::optional<std::vector<Tensor<T>>> InvokeBackwardAsyncTensor(
            Tensor<T> tensor) = 0;

        //! Used in asynchronous backward execution
        std::list<Tensor<T>> BackwardInputTensorPool;

        Device Device;
        std::string Name;
        size_t BatchSize = 0;

     protected:
        bool m_checkBackwardReady();

        std::mutex m_mtx;
    };
}



#endif