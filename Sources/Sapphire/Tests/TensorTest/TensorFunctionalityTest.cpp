// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Tests/TensorTest/TensorFunctionalityTest.hpp>
#include <Sapphire/Tests/TestUtil.hpp>
#include <Sapphire/tensor/TensorData.hpp>
#include <Sapphire/compute/Initialize.hpp>
#include <Sapphire/util/Device.hpp>
#include <random>
#include <doctest.h>


namespace Sapphire::Test
{
void SendDataBetweenHostDevice(bool print)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> intDistrib(1, 100);

    const int dim = intDistrib(gen) % 5 + 1;
    const Device host("host");
    const Device cuda(0, "cuda0");

    //! Randomly initialize the shape with random dimension and size
    const auto shape = CreateRandomShape(dim);

    TensorUtil::TensorData tensorData(shape, Type::Dense, host);

    //! Randomly initialize data on host
    Compute::Initialize::Normal(tensorData, 10, 5);

    //! Save the original data on the separate memory
    auto* originalData = new float[tensorData.DenseTotalLengthHost];
    std::memcpy(originalData, tensorData.GetMutableDenseHost(),
                tensorData.DenseTotalLengthHost * sizeof(float));

    //! Random data should be copied to cuda
    tensorData.SendTo(cuda);

    //! Re-Initialize data on host with zero
    for (unsigned long i = 0; i < tensorData.DenseTotalLengthHost; ++i)
        tensorData.GetMutableDenseHost()[i] = 0.0f;

    //! Zeros on the host memory should be overwritten by copied data on cuda
    tensorData.SendTo(host);

    //! Checks whether data has been succesfully copied to host 
    CheckNoneZeroEquality(tensorData.GetDenseHost(), originalData,
                          tensorData.DenseTotalLengthHost, print, 0.1f);

    //! Clear the Resourcemanager
    delete[] originalData;
}

void TensorDataCopyOnCuda(bool print)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> intDistrib(1, 100);

    const int dim = intDistrib(gen) % 5 + 1;
    const auto shape = CreateRandomShape(dim);
    const Device host("host");
    const Device cuda(0, "cuda0");

    TensorUtil::TensorData tensorData(shape, Type::Dense, host);

    //! Randomly initialize the data on host
    Compute::Initialize::Normal(tensorData, 10, 5);

    //! Save the original data on the separate memory
    auto* originalData = new float[tensorData.DenseTotalLengthHost];
    std::memcpy(originalData, tensorData.GetMutableDenseHost(),
                tensorData.DenseTotalLengthHost * sizeof(float));

    //! Copy data to cuda
    tensorData.SendTo(cuda);

    //! Re-Initialize data on host with zero
    for (unsigned long i = 0; i < tensorData.DenseTotalLengthHost; ++i)
        tensorData.GetMutableDenseHost()[i] = 0.0f;

    //! Create deep copy of the original tensorData
    TensorUtil::TensorData copiedTensorData = tensorData.CreateCopy();

    //! Copy data to host from copied tensorData
    copiedTensorData.SendTo(host);

    //! Checks whether data has been succesfully copied to host
    CheckNoneZeroEquality(copiedTensorData.GetDenseHost(), originalData,
                          copiedTensorData.DenseTotalLengthHost, print);

    //! Check equality of pointers
    CHECK((std::uintptr_t)(tensorData.GetDenseHost()) !=
        (std::uintptr_t)(copiedTensorData.GetDenseHost()));

    CHECK((std::uintptr_t)(tensorData.GetDenseCuda()) !=
        (std::uintptr_t)(copiedTensorData.GetDenseCuda()));

    TensorUtil::TensorData copyConstructedTensorData = tensorData;
    delete[] originalData;
}

void TensorDataCopyOnHost(bool print)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> intDistrib(1, 100);

    const int dim = intDistrib(gen) % 5 + 1;
    const auto shape = CreateRandomShape(dim);
    const Device host("host");

    TensorUtil::TensorData tensorData(shape, Type::Dense, host);

    //! Randomly initialize the data on host
    Compute::Initialize::Normal(tensorData, 10, 5);

    //! Create deep copy of the original tensorData
    TensorUtil::TensorData copiedTensorData = tensorData.CreateCopy();

    //! Checks whether data has been succesfully copied to host
    CheckNoneZeroEquality(copiedTensorData.GetDenseHost(),
                          tensorData.GetDenseHost(),
                          copiedTensorData.DenseTotalLengthHost, print);

    //! Check equality of pointers
    CHECK((std::uintptr_t)(tensorData.GetDenseHost()) !=
        (std::uintptr_t)(copiedTensorData.GetDenseHost()));

    TensorUtil::TensorData copyConstructedTensorData = tensorData;
}
}
