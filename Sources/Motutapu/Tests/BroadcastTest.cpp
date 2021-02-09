// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/Tests/BroadcastTest.hpp>
#include <Motutapu/compute/Compute.hpp>
#include <Motutapu/compute/Initialize.hpp>
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
void BroadcastWithOneDimension()
{
    for (int j = 0; j < 1; j++)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(1, 10);

        const unsigned int M = distrib(gen);
        const unsigned int N = distrib(gen);
        const unsigned int K = distrib(gen);
        const auto batchSize = distrib(gen) % 3 + 1;

        const Shape shapeA({ 1, M, K });
        const Shape shapeB({ M, K, N });
        const Shape shapeC({ 1, M, N });
        const Shape shapeOut({ M, M, N });

        std::cout << "M : " << M << " N: " << N << " K: " << K
                  << " batchSize : " << batchSize << std::endl;

        const Device cuda(0, "device0");
        const Device host("host");

        TensorUtil::TensorData A(shapeA, Type::Dense, host, batchSize);

        TensorUtil::TensorData B(shapeB, Type::Dense, host, batchSize);

        TensorUtil::TensorData C(shapeC, Type::Dense, host, batchSize);

        TensorUtil::TensorData Out(shapeOut, Type::Dense, host, batchSize);

        Compute::Initialize::Normal(A, 0, 5);
        Compute::Initialize::Normal(B, 0, 5);
        Compute::Initialize::Normal(C, 0, 5);
        Compute::Initialize::Zeros(C);

        Compute::Gemm(Out, A, B, C);

        float cpuGemmResult[Out.DenseTotalLengthHost];

#pragma omp parallel for default(shared) schedule(static)
        for (size_t i = 0; i < Out.DenseTotalLengthHost; ++i)
        {
            cpuGemmResult[i] = Out.DenseMatHost[i];
        }

        Compute::Initialize::Zeros(Out);

        A.SendTo(cuda);
        B.SendTo(cuda);
        C.SendTo(cuda);
        Out.SendTo(cuda);

        Compute::Gemm(Out, A, B, C);

        Out.SendTo(host);

        std::atomic<float> largestError = 0.0f;

        //#pragma omp parallel for default(shared) schedule(static)
        for (size_t i = 0; i < Out.DenseTotalLengthHost; ++i)
        {
            auto error = std::abs(cpuGemmResult[i] - Out.DenseMatHost[i]);
            if (largestError < error)
                largestError = error;

            //            std::cout << "cpu : " << cpuGemmResult[i]
            //                      << " cuda : " << Out.DenseMatHost[i] <<
            //                      std::endl;

            CHECK(error < 1.5f);
        }

        std::cout << "Largest error : " << largestError << std::endl;
    }

    Util::MemoryManager::ClearCudaMemoryPool();
    Util::MemoryManager::ClearHostMemoryPool();
}

void BroadcastWithMissingDimension()
{
    for (int j = 0; j < 1; j++)
    {
        std::random_device rd;
        std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with
        std::uniform_int_distribution<> distrib(1, 16);

        const unsigned int M = distrib(gen);
        const unsigned int N = distrib(gen);
        const unsigned int K = distrib(gen);
        const auto batchSize = distrib(gen) % 3 + 1;

        const Shape shapeA({ M, K });
        const Shape shapeB({ M, K, N });
        const Shape shapeC({ M, N });
        const Shape shapeOut({ M, M, N });

        std::cout << "M : " << M << " N: " << N << " K: " << K
                  << " batchSize : " << batchSize << std::endl;

        const Device cuda(0, "device0");
        const Device host("host");

        TensorUtil::TensorData A(shapeA, Type::Dense, host, batchSize);

        TensorUtil::TensorData B(shapeB, Type::Dense, host, batchSize);

        TensorUtil::TensorData C(shapeC, Type::Dense, host, batchSize);

        TensorUtil::TensorData Out(shapeOut, Type::Dense, host, batchSize);

        Compute::Initialize::Normal(A, 0, 5);
        Compute::Initialize::Normal(B, 0, 5);
        Compute::Initialize::Normal(C, 0, 5);
        Compute::Initialize::Zeros(C);

        Compute::Gemm(Out, A, B, C);

        float cpuGemmResult[Out.DenseTotalLengthHost];

#pragma omp parallel for default(shared) schedule(static)
        for (size_t i = 0; i < Out.DenseTotalLengthHost; ++i)
        {
            cpuGemmResult[i] = Out.DenseMatHost[i];
        }

        Compute::Initialize::Zeros(Out);

        A.SendTo(cuda);
        B.SendTo(cuda);
        C.SendTo(cuda);
        Out.SendTo(cuda);

        Compute::Gemm(Out, A, B, C);

        Out.SendTo(host);

        std::atomic<float> largestError = 0.0f;

        //#pragma omp parallel for default(shared) schedule(static)
        for (size_t i = 0; i < Out.DenseTotalLengthHost; ++i)
        {
            auto error = std::abs(cpuGemmResult[i] - Out.DenseMatHost[i]);
            if (largestError < error)
                largestError = error;

            //            std::cout << "cpu : " << cpuGemmResult[i]
            //                      << " cuda : " << Out.DenseMatHost[i] <<
            //                      std::endl;

            CHECK(error < 1.5f);
        }

        std::cout << "Largest error : " << largestError << std::endl;
    }

    Util::MemoryManager::ClearCudaMemoryPool();
    Util::MemoryManager::ClearHostMemoryPool();
}

void BroadcastMixed()
{
    for (int j = 0; j < 1; j++)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(1, 10);

        const unsigned int M = distrib(gen);
        const unsigned int N = distrib(gen);
        const unsigned int K = distrib(gen);
        const auto batchSize = distrib(gen) % 3 + 1;

        const Shape shapeA({ N, 1, M, K });
        const Shape shapeB({ M, K, N });
        const Shape shapeC({ 1, 1, M, N });
        const Shape shapeOut({ N, M, M, N });

        std::cout << "M : " << M << " N: " << N << " K: " << K
                  << " batchSize : " << batchSize << std::endl;

        const Device cuda(0, "device0");
        const Device host("host");

        TensorUtil::TensorData A(shapeA, Type::Dense, host, batchSize);

        TensorUtil::TensorData B(shapeB, Type::Dense, host, batchSize);

        TensorUtil::TensorData C(shapeC, Type::Dense, host, batchSize);

        TensorUtil::TensorData Out(shapeOut, Type::Dense, host, batchSize);

        Compute::Initialize::Normal(A, 0, 5);
        Compute::Initialize::Normal(B, 0, 5);
        Compute::Initialize::Normal(C, 0, 5);
        Compute::Initialize::Zeros(C);

        Compute::Gemm(Out, A, B, C);

        float cpuGemmResult[Out.DenseTotalLengthHost];

#pragma omp parallel for default(shared) schedule(static)
        for (size_t i = 0; i < Out.DenseTotalLengthHost; ++i)
        {
            cpuGemmResult[i] = Out.DenseMatHost[i];
        }

        Compute::Initialize::Zeros(Out);

        A.SendTo(cuda);
        B.SendTo(cuda);
        C.SendTo(cuda);
        Out.SendTo(cuda);

        Compute::Gemm(Out, A, B, C);

        Out.SendTo(host);

        std::atomic<float> largestError = 0.0f;

        //#pragma omp parallel for default(shared) schedule(static)
        for (size_t i = 0; i < Out.DenseTotalLengthHost; ++i)
        {
            auto error = std::abs(cpuGemmResult[i] - Out.DenseMatHost[i]);
            if (largestError < error)
                largestError = error;

            //            std::cout << "cpu : " << cpuGemmResult[i]
            //                      << " cuda : " << Out.DenseMatHost[i] <<
            //                      std::endl;

            CHECK(error < 1.5f);
        }

        std::cout << "Largest error : " << largestError << std::endl;
    }

    Util::MemoryManager::ClearCudaMemoryPool();
    Util::MemoryManager::ClearHostMemoryPool();
}
}  // namespace Motutapu::Test