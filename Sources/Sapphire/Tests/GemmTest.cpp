// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Tests/GemmTest.hpp>
#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/compute/Initialize.hpp>
#include <Sapphire/util/Shape.hpp>
#include <Sapphire/tensor/TensorData.hpp>
#include <Sapphire/util/CudaDevice.hpp>
#include <Sapphire/util/ResourceManager.hpp>
#include <Sapphire/Tests/TestUtil.hpp>
#include <iostream>
#include <random>

namespace Sapphire::Test
{
void Gemm1(bool print)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distribution(1, 100);

    const int M = distribution(gen);
    const int N = distribution(gen);
    const int K = distribution(gen);
    const int batchSize = distribution(gen) % 30 + 1;

    //! Randomly create with dimension size 3 with batchsize intergrated in the shape
    const Shape shapeA({ batchSize, M, K });
    const Shape shapeB({ batchSize, K, N });
    const Shape shapeC({ batchSize, M, N });
    const Shape shapeOut({ batchSize, M, N });

    std::cout << "M : " << M << " N: " << N << " K: " << K
        << " batchSize : " << batchSize << std::endl;

    //! Declare cuda device
    const CudaDevice cuda(0, "device0");

    //! Create tensors with cuda available
    TensorUtil::TensorData A(shapeA, Type::Dense, cuda);
    TensorUtil::TensorData B(shapeB, Type::Dense, cuda);
    TensorUtil::TensorData C(shapeC, Type::Dense, cuda);
    TensorUtil::TensorData Out(shapeOut, Type::Dense, cuda);

    //! Set tensor mode as host
    A.SetMode(DeviceType::Host);
    B.SetMode(DeviceType::Host);
    C.SetMode(DeviceType::Host);
    Out.SetMode(DeviceType::Host);

    //! Initialize with normal distribution
    Compute::Initialize::Normal(A, 10, 5);
    Compute::Initialize::Normal(B, 10, 5);
    Compute::Initialize::Normal(C, 10, 5);
    Compute::Initialize::Zeros(Out);

    //! Perform Gemm on host
    Compute::Gemm(Out, A, B, C);

    //! Copy the result to temporary buffer
    auto* cpuGemmResult = new float[Out.HostTotalSize];
    std::memcpy(cpuGemmResult, Out.HostRawPtr(),
                Out.HostTotalSize * sizeof(float));

    //! Initialize output as zeros
    Compute::Initialize::Zeros(Out);

    //! Move All data to Cuda and change to cuda mode
    A.ToCuda();
    B.ToCuda();
    C.ToCuda();
    Out.ToCuda();
    A.SetMode(DeviceType::Cuda);
    B.SetMode(DeviceType::Cuda);
    C.SetMode(DeviceType::Cuda);
    Out.SetMode(DeviceType::Cuda);

    //! Compute Gemm on Output
    Compute::Gemm(Out, A, B, C);

    //! Send output data to host and set to host mode
    Out.ToHost();
    Out.SetMode(DeviceType::Host);

    //! Check equality between host and cuda
    CheckNoneZeroEquality(cpuGemmResult, Out.HostRawPtr(),
                          Out.HostTotalSize, print, 2.0f);

    delete[] cpuGemmResult;
}
} // namespace Sapphire::Test
