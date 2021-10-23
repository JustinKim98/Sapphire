// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Tests/TensorTest/TensorFunctionalityTest.hpp>
#include <Sapphire/Tests/TestUtil.hpp>
#include <Sapphire/tensor/TensorData.hpp>
#include <Sapphire/compute/Initialize.hpp>
#include <Sapphire/util/CudaDevice.hpp>
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
    const CudaDevice cuda(0, "cuda0");

    //! Randomly initialize the shape with random dimension and size
    const auto shape = CreateRandomShape(dim);

    TensorUtil::TensorData tensorData(shape, Type::Dense, cuda);
    tensorData.ToHost();

    //! Randomly initialize data on host
    Compute::Initialize::Normal(tensorData, 10, 5);

    //! Save the original data on the separate memory
    auto originalData = tensorData.GetDataCopy();
    //! Random data should be copied to cuda
    tensorData.ToCuda();

    //! Re-Initialize data on host with zero
    for (unsigned long i = 0; i < tensorData.HostTotalSize; ++i)
        tensorData.HostMutableRawPtr()[i] = 0.0f;

    //! Zeros on the host memory should be overwritten by copied data on cuda
    tensorData.ToHost();
    const auto copiedData = tensorData.GetDataCopy();

    //! Checks whether data has been succesfully copied to host 
    CheckNoneZeroEquality(copiedData, originalData,
                          tensorData.Size(), print, 0.1f);
}

void TensorDataCopyOnCuda(bool print)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> intDistrib(1, 100);

    const int dim = intDistrib(gen) % 5 + 1;
    const auto shape = CreateRandomShape(dim);
    const CudaDevice cuda(0, "cuda0");

    TensorUtil::TensorData tensorData(shape, Type::Dense, cuda);
    tensorData.ToHost();

    //! Randomly initialize the data on host
    Compute::Initialize::Normal(tensorData, 10, 5);

    //! Copy data to cuda
    tensorData.ToCuda();

    //! Save the original data on the separate memory
    auto originalData = tensorData.GetDataCopy();

    //! Create deep copy of the original tensorData
    TensorUtil::TensorData copiedTensorData = tensorData.CreateCopy();

    auto copiedData = copiedTensorData.GetDataCopy();

    CHECK(tensorData.GetShape() == copiedTensorData.GetShape());

    //! Checks whether data has been succesfully copied to host
    CheckNoneZeroEquality(copiedData, originalData,
                          copiedTensorData.Size(), print);

    //! Check equality of pointers
    CHECK((std::uintptr_t)(tensorData.HostRawPtr()) !=
        (std::uintptr_t)(copiedTensorData.HostRawPtr()));

    CHECK((std::uintptr_t)(tensorData.CudaRawPtr()) !=
        (std::uintptr_t)(copiedTensorData.CudaRawPtr()));

    TensorUtil::TensorData copyConstructedTensorData = tensorData;
}

void TensorDataCopyOnHost(bool print)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> intDistrib(1, 100);

    const int dim = intDistrib(gen) % 5 + 1;
    const auto shape = CreateRandomShape(dim);

    CudaDevice cuda(0, "cuda0");
    TensorUtil::TensorData tensorData(shape, Type::Dense, cuda);
    tensorData.ToHost();
    tensorData.SetMode(ComputeMode::Host);

    //! Randomly initialize the data on host
    Compute::Initialize::Normal(tensorData, 10, 5);

    //! Create deep copy of the original tensorData
    TensorUtil::TensorData copiedTensorData = tensorData.CreateCopy();

    //! Checks whether data has been succesfully copied to host
    CheckNoneZeroEquality(copiedTensorData.HostRawPtr(),
                          tensorData.HostRawPtr(),
                          copiedTensorData.HostTotalSize, print);

    //! Check equality of pointers
    CHECK((std::uintptr_t)(tensorData.HostRawPtr()) !=
        (std::uintptr_t)(copiedTensorData.HostRawPtr()));

    TensorUtil::TensorData copyConstructedTensorData = tensorData;
}
}
