// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/compute/Initialize.hpp>
#include <Sapphire/tensor/Shape.hpp>
#include <Sapphire/tensor/TensorData.hpp>
#include <Sapphire/util/Device.hpp>
#include <Sapphire/util/ResourceManager.hpp>
#include <Sapphire/Tests/TestUtil.hpp>
#include <iostream>
#include <random>
#include "doctest.h"

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

    const Device cuda(0, "device0");
    const Device host("host");

    //! Define TensorData for input and transposed
    TensorUtil::TensorData inputTensor(shapeInput, Type::Dense, host);
    TensorUtil::TensorData transposedTensor(shapeTransposed, Type::Dense, host);

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
    inputTensor.SendTo(cuda);
    transposedTensor.SendTo(cuda);

    //! Transpose the tensor on cuda
    Compute::Transpose(transposedTensor, inputTensor);

    //! Send the transposed result to host
    //! Zero initialized data on the host should be overwritten
    transposedTensor.SendTo(host);

    //! Compare the results
    const float* cudaResult = transposedTensor.GetDenseHost();
    CheckNoneZeroEquality(cpuResult, cudaResult,
                          transposedTensor.DenseTotalLengthHost, printResult);

    delete[] cpuResult;
}
}
