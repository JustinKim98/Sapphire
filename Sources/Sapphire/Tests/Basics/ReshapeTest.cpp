// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Tests/Basics/ReshapeTest.hpp>
#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/compute/Initialize.hpp>
#include <Sapphire/util/Shape.hpp>
#include <Sapphire/tensor/TensorData.hpp>
#include <Sapphire/util/CudaDevice.hpp>
#include <Sapphire/util/ResourceManager.hpp>
#include <Sapphire/compute/IndexingOps.hpp>
#include <Sapphire/Tests/TestUtil.hpp>
#include <Sapphire/Model.hpp>
#include <iostream>
#include <random>
#include "doctest.h"

namespace Sapphire::Test
{
void ReshapeTest(bool printResult)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(1, 100);

    const int dim = dist(gen) % 1 + 2;

    const auto shapeInput = CreateRandomShape(dim, 30);
    const auto newShape = shapeInput;

    const CudaDevice cuda(0, "device0");

    //! Define TensorData for input and transposed
    TensorUtil::TensorData inputTensor(shapeInput, Type::Dense, cuda);
    inputTensor.SetMode(DeviceType::Host);

    //! Initialize input Tensor with normal distribution
    Compute::Initialize::Normal(inputTensor, 10, 5);
    //! Initialize output Tensor with zeros
    auto transposedTensor = inputTensor.CreateCopy();

    //! Perform reshape on host
    Compute::Reshape(transposedTensor, newShape);
    CHECK(transposedTensor.GetShape() == newShape);

    //! Use the buffer to store the host result temporarily
    auto* cpuResult = new float[transposedTensor.DenseTotalLengthHost];
    std::memcpy(cpuResult, transposedTensor.GetDenseHost(),
                transposedTensor.DenseTotalLengthHost * sizeof(float));

    //! Send the input tensor to cuda
    inputTensor.ToCuda();
    inputTensor.SetMode(DeviceType::Cuda);
    transposedTensor = inputTensor.CreateCopy();

    //! Perform reshape on cuda
    Compute::Reshape(transposedTensor, newShape);
    CHECK(transposedTensor.GetShape() == newShape);

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
