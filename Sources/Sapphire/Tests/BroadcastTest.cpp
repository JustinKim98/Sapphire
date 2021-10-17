// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Tests/BroadcastTest.hpp>
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
void BroadcastWithOneDimension(bool print)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(1, 10);

    const int M = distrib(gen);
    const int N = distrib(gen);
    const int K = distrib(gen);
    const int batchSize = distrib(gen) % 3 + 1;

    //! Set shape so they can be broadcasted
    const Shape shapeA({ batchSize, 1, M, K });
    const Shape shapeB({ batchSize, M, K, N });
    const Shape shapeOut({ batchSize, M, M, N });

    std::cout << "M : " << M << " N: " << N << " K: " << K
        << " batchSize : " << batchSize << std::endl;
    const CudaDevice cuda(0, "device0");

    //! Initialize tensors with cuda mode
    TensorUtil::TensorData A(shapeA, Type::Dense, cuda);
    TensorUtil::TensorData B(shapeB, Type::Dense, cuda);
    TensorUtil::TensorData Out(shapeOut, Type::Dense, cuda);

    A.SetMode(DeviceType::Host);
    B.SetMode(DeviceType::Host);
    Out.SetMode(DeviceType::Host);

    //! Initialize input tensors with normal distribution and output tensors as zeros
    Compute::Initialize::Normal(A, 10, 5);
    Compute::Initialize::Normal(B, 10, 5);
    Compute::Initialize::Zeros(Out);

    Compute::Gemm(Out, A, B);

    //! Set temporary buffer and copy the result
    auto* cpuGemmResult = new float[Out.HostTotalSize];
    std::memcpy(cpuGemmResult, Out.HostRawPtr(),
                Out.HostTotalSize * sizeof(float));

    //! Initialize output with zeros
    Compute::Initialize::Zeros(Out);

    //! Send data to cuda
    A.ToCuda();
    B.ToCuda();
    Out.ToCuda();

    //! Perform Gemm on cuda
    Compute::Gemm(Out, A, B);

    //! Send output data to host
    Out.ToHost();

    //! Check for non zero equality
    CheckNoneZeroEquality(cpuGemmResult, Out.HostRawPtr(),
                          Out.HostTotalSize, print, 1.5f);

    delete[] cpuGemmResult;
}

void BroadcastWithMissingDimension(bool print)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(1, 10);

    const int M = distrib(gen);
    const int N = distrib(gen);
    const int K = distrib(gen);

    //! Set shape so they can be broadcasted
    const Shape shapeA({ M, K });
    const Shape shapeB({ M, K, N });
    const Shape shapeOut({ M, M, N });

    std::cout << "M : " << M << " N: " << N << " K: " << K << std::endl;
    const CudaDevice cuda(0, "device0");

    //! Initialize tensors with cuda mode
    TensorUtil::TensorData A(shapeA, Type::Dense, cuda);
    TensorUtil::TensorData B(shapeB, Type::Dense, cuda);
    TensorUtil::TensorData Out(shapeOut, Type::Dense, cuda);

    A.SetMode(DeviceType::Host);
    B.SetMode(DeviceType::Host);
    Out.SetMode(DeviceType::Host);

    //! Initialize input tensors with normal distribution and output tensors as
    //! zeros
    Compute::Initialize::Normal(A, 10, 5);
    Compute::Initialize::Normal(B, 10, 5);
    Compute::Initialize::Zeros(Out);

    //! Perform Gemm on host
    Compute::Gemm(Out, A, B);

    //! Set temporary buffer and copy the result
    auto* cpuGemmResult = new float[Out.HostTotalSize];
    std::memcpy(cpuGemmResult, Out.HostRawPtr(),
                Out.HostTotalSize * sizeof(float));
    //! Initialize output with zeros
    Compute::Initialize::Zeros(Out);

    //! Send data to cuda
    A.ToCuda();
    B.ToCuda();
    Out.ToCuda();

    //! Perform Gemm on cuda
    Compute::Gemm(Out, A, B);

    //! Send output data to host
    Out.ToHost();

    //! Check for non zero equality
    CheckNoneZeroEquality(cpuGemmResult, Out.HostRawPtr(),
                          Out.HostTotalSize, print, 1.5f);

    delete[] cpuGemmResult;
}

void BroadcastMixed(bool print)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(1, 10);

    const int M = distrib(gen);
    const int N = distrib(gen);
    const int K = distrib(gen);

    //! Set shape so they can be broadcasted
    const Shape shapeA({ M, K });
    const Shape shapeB({ N, M, K, N });
    const Shape shapeOut({ N, M, M, N });

    std::cout << "M : " << M << " N: " << N << " K: " << K << std::endl;
    const CudaDevice cuda(0, "device0");

    //! Initialize tensors with cuda mode
    TensorUtil::TensorData A(shapeA, Type::Dense, cuda);
    TensorUtil::TensorData B(shapeB, Type::Dense, cuda);
    TensorUtil::TensorData Out(shapeOut, Type::Dense, cuda);

    A.SetMode(DeviceType::Host);
    B.SetMode(DeviceType::Host);
    Out.SetMode(DeviceType::Host);

    //! Initialize input tensors with normal distribution and output tensors
    //! as zeros
    Compute::Initialize::Normal(A, 10, 5);
    Compute::Initialize::Normal(B, 10, 5);
    Compute::Initialize::Zeros(Out);

    //! Perform Gemm on host
    Compute::Gemm(Out, A, B);

    //! Set temporary buffer and copy the result
    auto* cpuGemmResult = new float[Out.HostTotalSize];
    std::memcpy(cpuGemmResult, Out.HostRawPtr(),
                Out.HostTotalSize * sizeof(float));
    //! Initialize output with zeros
    Compute::Initialize::Zeros(Out);

    //! Send data to cuda
    A.ToCuda();
    B.ToCuda();
    Out.ToCuda();

    //! Perform Gemm on cuda
    Compute::Gemm(Out, A, B);

    //! Send output data to host
    Out.ToHost();

    //! Check for non zero equality
    CheckNoneZeroEquality(cpuGemmResult, Out.HostRawPtr(),
                          Out.HostTotalSize, print, 1.5f);

    delete[] cpuGemmResult;
}
} // namespace Sapphire::Test
