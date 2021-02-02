// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/Compute.hpp>
#include <Motutapu/compute/Initialize.hpp>
#include <Motutapu/compute/naive/NaiveInitialize.hpp>
#include <Motutapu/tensor/Shape.hpp>
#include <Motutapu/tensor/TensorData.hpp>
#include <Motutapu/util/Device.hpp>
#include <Motutapu/util/MemoryManager.hpp>
#include <atomic>
#include <cmath>
#include <iostream>
#include <random>
#include "doctest.h"

namespace Motutapu::Test
{
void TestGemm()
{
    const auto M = 240;
    const auto N = 160;
    const auto K = 480;
    const Shape shapeA({ M, K });
    const Shape shapeB({ K, N });
    const Shape shapeC({ M, N });
    const Shape shapeOut({ M, N });

    const Device cuda(0, "device0");
    const Device host("host");

    const auto batchSize = 10;

    TensorUtil::TensorData A(shapeA, Type::Dense, host, batchSize);

    TensorUtil::TensorData B(shapeB, Type::Dense, host, batchSize);

    TensorUtil::TensorData C(shapeC, Type::Dense, host, batchSize);

    TensorUtil::TensorData Out(shapeOut, Type::Dense, host, batchSize);

    std::random_device
        rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with
                             // rd()
    std::uniform_real_distribution<> distrib(1, 6);

#pragma omp parallel for default(shared) schedule(static)
    for (size_t i = 0; i < A.DenseTotalLength; ++i)
    {
        A.DenseMatHost[i] = static_cast<float>(distrib(gen));
    }

#pragma omp parallel for default(shared) schedule(static)
    for (size_t i = 0; i < B.DenseTotalLength; ++i)
    {
        B.DenseMatHost[i] = static_cast<float>(distrib(gen));
    }

#pragma omp parallel for default(shared) schedule(static)
    for (size_t i = 0; i < C.DenseTotalLength; ++i)
    {
        C.DenseMatHost[i] = static_cast<float>(distrib(gen));
    }

    Compute::Initialize::Zeros(Out);

    Compute::Gemm(Out, A, B, C);

    float cpuGemmResult[Out.DenseTotalLength];

    for (size_t i = 0; i < Out.DenseTotalLength; ++i)
    {
        cpuGemmResult[i] = Out.DenseMatHost[i];
    }

    Compute::Initialize::Zeros(Out);

    A.SendTo(cuda);
    B.SendTo(cuda);
    C.SendTo(cuda);
    Out.SendTo(cuda);

    Compute::Gemm(Out, A, B, C);
    Compute::Naive::Scalar(Out.DenseMatHost, 0.0f, Out.DenseTotalLength);

    Out.SendTo(host);

    std::atomic<float> largestError = 0.0f;

#pragma omp parallel for default(shared) schedule(static)
    for (size_t i = 0; i < Out.DenseTotalLength; ++i)
    {
        auto error = std::abs(cpuGemmResult[i] - Out.DenseMatHost[i]);
        if (largestError < error)
            largestError = error;
        //CHECK(std::abs(cpuGemmResult[i] - Out.DenseMatHost[i]) < 1.0f);
    }

    A.Free();
    B.Free();
    C.Free();
    Out.Free();

    std::cout << "Largest error : " << largestError << std::endl;

    // Util::MemoryManager::ClearCudaMemoryPool();
    // Util::MemoryManager::ClearHostMemoryPool();
}

}  // namespace Motutapu::Test
