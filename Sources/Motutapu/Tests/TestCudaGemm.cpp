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
    {
        std::random_device
            rd;  // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(
            rd());  // Standard mersenne_twister_engine seeded with rd()
        std::uniform_int_distribution<> distrib(1, 50);

        const unsigned int M = distrib(gen) % 30;
        const unsigned int N = distrib(gen) % 30;
        const unsigned int K = distrib(gen) % 30;
        const Shape shapeA({ M, K });
        const Shape shapeB({ K, N });
        const Shape shapeC({ M, N });
        const Shape shapeOut({ M, N });

        const Device cuda(0, "device0");
        const Device host("host");

        const auto batchSize = 3;

        TensorUtil::TensorData A(shapeA, Type::Dense, host, batchSize);

        TensorUtil::TensorData B(shapeB, Type::Dense, host, batchSize);

        TensorUtil::TensorData C(shapeC, Type::Dense, host, batchSize);

        TensorUtil::TensorData Out(shapeOut, Type::Dense, host, batchSize);

        Compute::Initialize::Normal(A, 0, 5);
        Compute::Initialize::Normal(B, 0, 5);
        Compute::Initialize::Normal(C, 0, 5);
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
            CHECK(error < 1.0f);
        }

        std::cout << "Largest error : " << largestError << std::endl;
    }
    Util::MemoryManager::ClearCudaMemoryPool();
    Util::MemoryManager::ClearHostMemoryPool();
}

}  // namespace Motutapu::Test
