// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/compute/Initialize.hpp>
#include <Sapphire/tensor/Shape.hpp>
#include <Sapphire/tensor/TensorData.hpp>
#include <Sapphire/util/CudaDevice.hpp>
#include <Sapphire/util/ResourceManager.hpp>
#include <Sapphire/Tests/TestUtil.hpp>
#include <Sapphire/Model.hpp>
#include <iostream>
#include <random>

namespace Sapphire::Test
{
void TransposeTest(bool printResult)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(1, 100);

    const int dim = distrib(gen) % 1 + 2;

    const auto shapeInput = CreateRandomShape(dim, 30);
    const auto shapeTransposed = shapeInput.GetTranspose();

    const CudaDevice cuda(0, "device0");

    //! Define TensorData for input and transposed
    TensorUtil::TensorData inputTensor(shapeInput, Type::Dense, cuda);
    TensorUtil::TensorData transposedTensor(shapeTransposed, Type::Dense, cuda);
    inputTensor.SetMode(DeviceType::Host);
    transposedTensor.SetMode(DeviceType::Host);

    //! Initialize input Tensor with normal distribution
    Compute::Initialize::Normal(inputTensor, 10, 5);
    //! Initialize output Tensor with zeros
    Compute::Initialize::Zeros(transposedTensor);

    //! Perform transpose on the host side
    Compute::Transpose(transposedTensor, inputTensor);
    //! Use the buffer to store the host result temporarily
    auto* cpuResult = new float[transposedTensor.DenseTotalLengthHost];
    std::memcpy(cpuResult, transposedTensor.GetDenseHost(),
                transposedTensor.DenseTotalLengthHost * sizeof(float));

    //! Re-initialize the transoised tensor with zeros
    Compute::Initialize::Zeros(transposedTensor);

    //! Send the input tensor to cuda
    inputTensor.ToCuda();
    transposedTensor.ToCuda();
    inputTensor.SetMode(DeviceType::Cuda);
    transposedTensor.SetMode(DeviceType::Cuda);

    //! Transpose the tensor on cuda
    Compute::Transpose(transposedTensor, inputTensor);

    //! Send the transposed result to host
    //! Zero initialized data on the host should be overwritten
    transposedTensor.ToHost();
    transposedTensor.SetMode(DeviceType::Host);

    //! Compare the results
    const float* cudaResult = transposedTensor.GetDenseHost();
    CheckNoneZeroEquality(cpuResult, cudaResult,
                          transposedTensor.DenseTotalLengthHost, printResult);

    delete[] cpuResult;
}
}
