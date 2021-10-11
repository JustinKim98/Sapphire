// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_TEST_TEST_UTIL_HPP
#define SAPPHIRE_TEST_TEST_UTIL_HPP

#include <Sapphire/util/Shape.hpp>
#include <Sapphire/tensor/TensorData.hpp>
#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/compute/Initialize.hpp>
#include <limits>
#include <random>
#include <cstring>
#include <tuple>
#include <iostream>
#include <doctest.h>

namespace Sapphire::Test
{
void InitIntegerDenseMatrix(float* matrixPtr, const size_t m, const size_t n,
                            const size_t paddedN, const size_t numMatrices,
                            const float sparsity);

void InitRandomDenseMatrix(float* matrixPtr, const size_t m, const size_t n,
                           const size_t paddedN, const size_t numMatrices,
                           const float sparsity);

Shape CreateRandomShape(int dim, int maxDim = 10);

template <typename TPtrA, typename TPtrB>
void CheckNoneZeroEquality(const TPtrA ptrA, const TPtrB ptrB, unsigned size,
                           bool print,
                           float equalThreshold = std::numeric_limits<
                               float>::epsilon())
{
    bool isAllZero = true;
    for (unsigned int i = 0; i < size; ++i)
    {
        if (print)
            std::cout << "ptrA : " << ptrA[i] << " ptrB : " << ptrB[i]
                << std::endl;
        if (!std::isnan(ptrA[i]) && !std::isnan(ptrB[i]))
            CHECK(std::abs(ptrA[i] - ptrB[i]) <= equalThreshold);
        if (ptrA[i] > 0.0f || ptrA[i] < 0.0f)
            isAllZero = false;
    }
    CHECK(!isAllZero);
}

void CheckNoneZero(const float* ptr, unsigned size, unsigned colSize,
                   unsigned padSize, bool print);

template <typename Func>
void TestWithTwoArgumentsWithSameShape(bool print, float equalThreshold,
                                       Func function)
{
    const Shape shape = CreateRandomShape(5);
    const CudaDevice cuda(0, "cuda0");

    //! Initialize data
    TensorUtil::TensorData A(shape, Type::Dense, cuda);
    TensorUtil::TensorData B(shape, Type::Dense, cuda);
    TensorUtil::TensorData Out(shape, Type::Dense, cuda);
    A.SetMode(DeviceType::Host);
    B.SetMode(DeviceType::Host);
    Out.SetMode(DeviceType::Host);

    //! Initialize the input data with normal distribution and output data as
    //! zeros
    Compute::Initialize::Normal(A, 10, 5);
    Compute::Initialize::Normal(B, 10, 5);
    Compute::Initialize::Zeros(Out);

    //! Invoke the function to be tested
    function(Out, A, B);

    //! Copy the host result to temporary buffer
    auto* cpuGemmResult = new float[Out.DenseTotalLengthHost];
    std::memcpy(cpuGemmResult, Out.HostRawPtr(),
                Out.DenseTotalLengthHost * sizeof(float));

    //! Initialize output as zeros on host
    Compute::Initialize::Zeros(Out);

    //! Send data to cuda
    A.ToCuda();
    B.ToCuda();
    Out.ToCuda();
    A.SetMode(DeviceType::Cuda);
    B.SetMode(DeviceType::Cuda);
    Out.SetMode(DeviceType::Cuda);

    function(Out, A, B);

    //! Send the data back to host
    Out.ToHost();
    Out.SetMode(DeviceType::Host);

    //! Check the equality
    CheckNoneZeroEquality(cpuGemmResult, Out.HostRawPtr(),
                          Out.DenseTotalLengthHost, print, equalThreshold);

    delete[] cpuGemmResult;
}

template <typename Func>
void TestWithOneArgumentStatic(bool print, float equalThreshold,
                               Func function)
{
    const Shape shape = CreateRandomShape(5);
    const CudaDevice cuda(0, "cuda0");

    //! Initialize data
    TensorUtil::TensorData In(shape, Type::Dense, cuda);
    TensorUtil::TensorData Out(shape, Type::Dense, cuda);
    In.SetMode(DeviceType::Host);
    Out.SetMode(DeviceType::Host);

    //! Initialize the input data with normal distribution and output data as
    //! zeros
    Compute::Initialize::Ones(In);
    Compute::Scale(In, In, 10);
    Compute::Initialize::Zeros(Out);

    //! Invoke the function to be tested
    function(Out, In);

    //! Copy the host result to temporary buffer
    auto* cpuResult = new float[Out.DenseTotalLengthHost];
    std::memcpy(cpuResult, Out.HostRawPtr(),
                Out.DenseTotalLengthHost * sizeof(float));

    //! Initialize output as zeros on host
    Compute::Initialize::Zeros(Out);

    //! Send data to cuda
    In.ToCuda();
    Out.ToCuda();
    In.SetMode(DeviceType::Cuda);
    Out.SetMode(DeviceType::Cuda);

    //! Invoke function on cuda
    function(Out, In);

    //! Send the data back to host
    Out.ToHost();
    Out.SetMode(DeviceType::Host);

    //! Check the equality
    CheckNoneZeroEquality(cpuResult, Out.HostRawPtr(),
                          Out.DenseTotalLengthHost, print, equalThreshold);

    delete[] cpuResult;
}

template <typename Func>
void TestWithOneArgumentNormal(bool print, float equalThreshold, Func function,
                               float mean, float sd)
{
    const Shape shape = CreateRandomShape(5);
    const CudaDevice cuda(0, "cuda0");

    //! Initialize data
    TensorUtil::TensorData In(shape, Type::Dense, cuda);
    TensorUtil::TensorData Out(shape, Type::Dense, cuda);
    In.SetMode(DeviceType::Host);
    Out.SetMode(DeviceType::Host);

    //! Initialize the input data with normal distribution and output data as
    //! zeros
    Compute::Initialize::Normal(In, mean, sd);
    Compute::Initialize::Zeros(Out);

    //! Invoke the function to be tested
    function(Out, In);

    //! Copy the host result to temporary buffer
    auto* cpuResult = new float[Out.DenseTotalLengthHost];
    std::memcpy(cpuResult, Out.HostRawPtr(),
                Out.DenseTotalLengthHost * sizeof(float));

    //! Initialize output as zeros on host
    Compute::Initialize::Zeros(Out);

    //! Send data to cuda
    In.ToCuda();
    Out.ToCuda();
    In.SetMode(DeviceType::Cuda);
    Out.SetMode(DeviceType::Cuda);

    //! Invoke function on cuda
    function(Out, In);

    //! Send the data back to host
    Out.ToHost();
    Out.SetMode(DeviceType::Host);

    //! Check the equality
    CheckNoneZeroEquality(cpuResult, Out.HostRawPtr(),
                          Out.DenseTotalLengthHost, print, equalThreshold);

    delete[] cpuResult;
}

template <typename Func>
void EqualInitializeTest(Func function, bool print)
{
    const Shape shape = CreateRandomShape(5);
    const CudaDevice cuda(0, "cuda0");

    TensorUtil::TensorData data(shape, Type::Dense, cuda);
    data.SetMode(DeviceType::Host);

    function(data);

    //! Copy the host result to temporary memory
    auto* cpuResult = new float[data.DenseTotalLengthHost];
    std::memcpy(cpuResult, data.HostRawPtr(),
                data.DenseTotalLengthHost * sizeof(float));

    //! Initialize host with zeros
    Compute::Initialize::Zeros(data);

    data.ToCuda();
    data.SetMode(DeviceType::Cuda);
    function(data);

    data.ToHost();
    data.SetMode(DeviceType::Host);

    CheckNoneZeroEquality(cpuResult, data.HostRawPtr(),
                          data.DenseTotalLengthHost, print);
}

template <typename Func, typename ...Ts>
void NoneZeroTest(Func function, bool print, Ts ...params)
{
    const Shape shape = CreateRandomShape(5);
    const CudaDevice cuda(0, "cuda0");

    TensorUtil::TensorData data(shape, Type::Dense, cuda);
    data.SetMode(DeviceType::Host);

    function(data, params...);

    CheckNoneZero(data.HostRawPtr(),
                  data.DenseTotalLengthHost, data.GetShape().Cols()
                  , data.PaddedHostColSize, print);

    //! Initialize host with zeros
    Compute::Initialize::Zeros(data);

    data.ToCuda();
    data.SetMode(DeviceType::Cuda);

    function(data, params...);

    data.ToHost();
    data.SetMode(DeviceType::Host);

    CheckNoneZero(data.HostRawPtr(),
                  data.DenseTotalLengthHost,
                  data.GetShape().Cols()
                  , data.PaddedHostColSize, print);
}


template <typename T, std::size_t... Indices>
auto VectorToTuple(const std::vector<T>& v,
                   std::index_sequence<Indices...>)
{
    return std::make_tuple(v[Indices]...);
}

template <std::size_t N, typename T>
auto VectorToTuple(const std::vector<T>& v)
{
    assert(v.size() >= N);
    return VectorToTuple(v, std::make_index_sequence<N>());
}

template <typename T>
auto SendTo(DeviceType device, T& tensors)
{
    if (device == DeviceType::Cuda)
        tensors.ToCuda();
    else
        tensors.ToHost();
    tensors.SetMode(device);
}

template <typename T, typename ...Ts>
auto SendTo(DeviceType device, T tensor, Ts&& ... tensors)
{
    static_assert(std::is_same_v<TensorUtil::TensorData, T>);
    if (device == DeviceType::Cuda)
        tensor.ToCuda();
    else
        tensor.ToHost();
    tensor.SetMode(device);
    SendTo(std::forward<Ts>(tensors)...);
}

template <typename TFunc, typename ...TFuncParams>
void TestOperation(CudaDevice cudaDevice, std::tuple<Shape> inputShapes,
                   std::tuple<Shape> outputShapes,
                   TFunc function,
                   std::tuple<TFuncParams ...> funcParams)
{
    auto inputTensors = VectorToTuple(std::apply(
        [&cudaDevice](auto&&... shape) {
            std::vector<TensorUtil::TensorData> dataVector;
            dataVector.reserve(sizeof...(TFuncParams));
            (dataVector.emplace_back(
                    TensorUtil::TensorData(shape, Type::Dense, cudaDevice)),
                ...);
            return dataVector;
        },
        inputShapes));

    auto outputTensors = VectorToTuple(std::apply(
        [&cudaDevice](auto&&... shape) {
            std::vector<TensorUtil::TensorData> dataVector;
            dataVector.reserve(sizeof...(TFuncParams));
            (dataVector.emplace_back(
                    TensorUtil::TensorData(shape, Type::Dense, cudaDevice)),
                ...);
            return dataVector;
        },
        outputShapes));

    std::apply([](auto&& ...tensor) {
                   ((Compute::Initialize::Normal(tensor, 0.0f, 1.0f)), ...);
               },
               inputTensors);

    std::apply(
        [](auto&&... tensor) {
            ((Compute::Initialize::Zeros(tensor)), ...);
        },
        outputTensors);

    auto tensors = std::tuple_cat(inputTensors, outputTensors);

    std::apply(SendTo,
               std::make_tuple(
                   std::tuple_cat(std::make_tuple(DeviceType::Cuda),
                                  inputTensors,
                                  outputTensors))
        );

    auto params = std::tuple_cat(outputTensors, inputTensors, funcParams);
    auto outputCuda = std::apply(function, params);
}
} // namespace Sapphire::Test

#endif  // SAPPHIRE_TESTUTIL_HPP
