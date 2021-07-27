// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Tests/BasicComputationTest.hpp>
#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/compute/Initialize.hpp>
#include <Sapphire/tensor/Shape.hpp>
#include <Sapphire/tensor/TensorData.hpp>
#include <Sapphire/util/Device.hpp>
#include <Sapphire/util/ResourceManager.hpp>
#include <Sapphire/compute/TrigonometricOps.hpp>
#include <Sapphire/compute/ActivationOps.hpp>
#include <atomic>
#include <cmath>
#include <iostream>
#include <random>
#include "doctest.h"

namespace Sapphire::Test
{
void TestTranspose(bool printResult)
{
    for (int j = 0; j < 5; j++)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(1, 100);

        const unsigned int M = distrib(gen);
        const unsigned int N = distrib(gen);
        const auto batchSize = distrib(gen) % 5 + 1;

        const Shape shapeA({ M, N });
        const Shape shapeB({ M, N });
        const Shape shapeOut({ M, N });

        std::cout << "M : " << M << " N: " << N << " batchSize : " << batchSize
            << std::endl;

        const Device cuda(0, "device0");
        const Device host("host");

        TensorUtil::TensorData A(shapeA, Type::Dense, host, batchSize);
        TensorUtil::TensorData B(shapeB, Type::Dense, host, batchSize);
        TensorUtil::TensorData out(shapeOut, Type::Dense, host, batchSize);
        TensorUtil::TensorData transposedOut(shapeOut.GetTranspose(),
                                             Type::Dense, host, batchSize);

        Compute::Initialize::Normal(A, 10, 5);
        Compute::Initialize::Normal(B, 10, 5);

        Compute::Add(out, A, B);
        Compute::Scale(out, out, 2);
        Compute::Transpose(transposedOut, out);

        auto* cpuResult = new float[transposedOut.DenseTotalLengthHost];

        for (long i = 0; i < static_cast<long>(transposedOut.
                             DenseTotalLengthHost); ++i)
        {
            cpuResult[i] = transposedOut.GetMutableDenseHost()[i];
        }

        A.SendTo(cuda);
        B.SendTo(cuda);
        out.SendTo(cuda);
        transposedOut.SendTo(cuda);

        Compute::Initialize::Zeros(transposedOut);

        Compute::Add(out, A, B);
        Compute::Scale(out, out, 2);
        Compute::Transpose(transposedOut, out);

        out.SendTo(host);
        transposedOut.SendTo(host);

        std::atomic<float> largestError = 0.0f;

        const size_t offset = 1 * out.Rows() * out.PaddedHostColSize;
        const size_t offsetTransposed =
            1 * transposedOut.Rows() * transposedOut.PaddedHostColSize;

        if (printResult)
        {
            std::cout << "\nOriginal result" << std::endl;

            for (size_t rowIdx = 0; rowIdx < out.Rows(); ++rowIdx)
            {
                for (size_t colIdx = 0; colIdx < out.Cols(); ++colIdx)
                {
                    std::cout
                        << out.GetMutableDenseHost()[offset +
                                            rowIdx * out.PaddedHostColSize +
                                            colIdx]
                        << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;

            std::cout << "\nHost Result" << std::endl;
            for (size_t rowIdx = 0; rowIdx < transposedOut.Rows(); ++rowIdx)
            {
                for (size_t colIdx = 0; colIdx < transposedOut.Cols(); ++colIdx)
                {
                    std::cout
                        << cpuResult[offsetTransposed +
                                     rowIdx * transposedOut.PaddedHostColSize +
                                     colIdx]
                        << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "\nCuda Result" << std::endl;
            for (size_t rowIdx = 0; rowIdx < transposedOut.Rows(); ++rowIdx)
            {
                for (size_t colIdx = 0; colIdx < transposedOut.Cols(); ++colIdx)
                {
                    std::cout << transposedOut.GetMutableDenseHost()
                        [offsetTransposed +
                         rowIdx * transposedOut.PaddedHostColSize +
                         colIdx]
                        << " ";
                }
                std::cout << std::endl;
            }
        }

#pragma omp parallel for default(shared) schedule(static)
        for (long i = 0; i < static_cast<long>(transposedOut.
                             DenseTotalLengthHost); ++i)
        {
            auto error = std::abs(cpuResult[i] - transposedOut.GetMutableDenseHost()[i]);
            if (largestError < error)
                largestError = error;

            CHECK(error <= 1.0f);
            if (error > 1.0f)
                std::cout << i / (transposedOut.Rows() *
                                  transposedOut.PaddedHostColSize)
                    << std::endl;
        }

        std::cout << "Largest error : " << largestError << std::endl;

        delete[] cpuResult;
    }

    Util::ResourceManager::ClearAll();
}

void TestBasics1()
{
    for (int j = 0; j < 5; j++)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(1, 100);

        const unsigned int M = distrib(gen);
        const unsigned int N = distrib(gen);
        const auto batchSize = distrib(gen) % 5 + 1;

        const Shape shapeA({ M, N });
        const Shape shapeB({ M, N });
        const Shape shapeOut({ M, N });

        std::cout << "M : " << M << " N: " << N << " batchSize : " << batchSize
            << std::endl;

        const Device cuda(0, "device0");
        const Device host("host");

        TensorUtil::TensorData A(shapeA, Type::Dense, host, 1);
        TensorUtil::TensorData B(shapeB, Type::Dense, host, batchSize);
        TensorUtil::TensorData Out(shapeOut, Type::Dense, host, batchSize);

        Compute::Initialize::Normal(A, 10, 5);
        Compute::Initialize::Normal(B, 10, 5);

        Compute::Add(Out, A, B);
        Compute::Scale(Out, Out, 3);
        Compute::Sub(Out, Out, A);
        Compute::Tanh(Out, Out);

        auto* cpuGemmResult = new float[Out.DenseTotalLengthHost];

#pragma omp parallel for default(shared) schedule(static)
        for (long i = 0; i < static_cast<long>(Out.DenseTotalLengthHost); ++i)
        {
            cpuGemmResult[i] = Out.GetMutableDenseHost()[i];
        }

        Compute::Initialize::Zeros(Out);

        A.SendTo(cuda);
        B.SendTo(cuda);
        Out.SendTo(cuda);

        Compute::Add(Out, A, B);
        Compute::Scale(Out, Out, 3);
        Compute::Sub(Out, Out, A);
        Compute::Tanh(Out, Out);

        Out.SendTo(host);

        std::atomic<float> largestError = 0.0f;

#pragma omp parallel for default(shared) schedule(static)
        for (long i = 0; i < static_cast<long>(Out.DenseTotalLengthHost); ++i)
        {
            auto error = std::abs(cpuGemmResult[i] - Out.GetMutableDenseHost()[i]);
            if (largestError < error)
                largestError = error;

            CHECK(error <= 1.0f);
        }

        std::cout << "Largest error : " << largestError << std::endl;

        delete[] cpuGemmResult;
    }

    Util::ResourceManager::ClearAll();
}

void TestBasics2()
{
    for (int j = 0; j < 5; j++)
    {
        std::random_device
            rd; // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(
            rd()); // Standard mersenne_twister_engine seeded with rd()
        std::uniform_int_distribution<> distribution(8, 16);

        const unsigned int M = distribution(gen);
        const unsigned int N = distribution(gen);
        const auto batchSize = distribution(gen) % 5 + 1;

        std::cout << "M : " << M << " N: " << N << " batchSize : " << batchSize
            << std::endl;

        const Shape shapeA({ M, N });
        const Shape shapeB({ M, N });
        const Shape shapeOut({ M, N });

        const Device cuda(0, "device0");
        const Device host("host");

        TensorUtil::TensorData A(shapeA, Type::Dense, cuda, 1);
        TensorUtil::TensorData B(shapeB, Type::Dense, cuda, batchSize);
        TensorUtil::TensorData out(shapeOut, Type::Dense, cuda, batchSize);

        Compute::Initialize::Normal(A, 10, 5);
        Compute::Initialize::Normal(B, 10, 5);
        Compute::Initialize::Zeros(out);

        Compute::Dot(out, A, B);
        Compute::Add(out, out, A);
        Compute::Sub(out, out, B);
        Compute::ReLU(out, out);
        Compute::Sin(out, out);

        A.SendTo(host);
        B.SendTo(host);
        out.SendTo(host);

        auto* cudaGemmResult = new float[out.DenseTotalLengthHost];

#pragma omp parallel for default(shared) schedule(static)
        for (long i = 0; i < static_cast<long>(out.DenseTotalLengthHost); ++i)
        {
            cudaGemmResult[i] = out.GetMutableDenseHost()[i];
        }

        Compute::Initialize::Zeros(out);
        Compute::Dot(out, A, B);
        Compute::Add(out, out, A);
        Compute::Sub(out, out, B);
        Compute::ReLU(out, out);
        Compute::Sin(out, out);

        std::atomic<float> largestError = 0.0f;

#pragma omp parallel for default(shared) schedule(static)
        for (long i = 0; i < static_cast<long>(out.DenseTotalLengthHost); ++i)
        {
            auto error = std::abs(cudaGemmResult[i] - out.GetMutableDenseHost()[i]);
            if (largestError < error)
                largestError = error;

            CHECK(error <= std::abs(out.GetMutableDenseHost()[i] / 100.0f));
        }

        std::cout << "Largest error : " << largestError << std::endl;
        delete[] cudaGemmResult;
    }

    Util::ResourceManager::ClearAll();
}

void TestAddBroadcast1()
{
    for (int j = 0; j < 5; j++)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distribution(1, 32);

        const unsigned int M = distribution(gen);
        const unsigned int N = distribution(gen);
        const auto batchSize = distribution(gen) % 5 + 1;

        std::cout << "M : " << M << " N: " << N << " batchSize : " << batchSize
            << std::endl;

        const Shape shapeA({ M, M, N });
        const Shape shapeB({ 1, M, N });
        const Shape shapeOut({ M, M, N });

        const Device cuda(0, "device0");
        const Device host("host");

        TensorUtil::TensorData A(shapeA, Type::Dense, cuda, batchSize);
        TensorUtil::TensorData B(shapeB, Type::Dense, cuda, batchSize);
        TensorUtil::TensorData out(shapeOut, Type::Dense, cuda, batchSize);

        Compute::Initialize::Normal(A, 100, 1);
        Compute::Initialize::Normal(B, 100, 4);
        Compute::Initialize::Zeros(out);

        const unsigned int M2 = distribution(gen);
        const unsigned int N2 = distribution(gen);
        const auto batchSize2 = distribution(gen) % 5 + 1;

        std::cout << "M2 : " << M2 << " N2: " << N2
            << " batchSize : " << batchSize2 << std::endl;

        TensorUtil::TensorData A2(shapeA, Type::Dense, cuda, batchSize2);
        TensorUtil::TensorData B2(shapeB, Type::Dense, cuda, batchSize2);
        TensorUtil::TensorData out2(shapeOut, Type::Dense, cuda, batchSize2);

        Compute::Initialize::Normal(A2, 100, 1);
        Compute::Initialize::Normal(B2, 100, 4);
        Compute::Initialize::Zeros(out2);

        Compute::Add(out2, A2, B2);
        Compute::Add(out2, A2, out2);
        Compute::Add(out2, B2, out2);

        Compute::Add(out, A, B);
        Compute::Add(out, A, out);
        Compute::Add(out, B, out);

        A2.SendTo(host);
        B2.SendTo(host);
        out2.SendTo(host);

        A.SendTo(host);
        B.SendTo(host);
        out.SendTo(host);

        auto* cudaGemmResult1 = new float[out.DenseTotalLengthHost];
        auto* cudaGemmResult2 = new float[out2.DenseTotalLengthHost];

#pragma omp parallel for default(shared) schedule(static)
        for (long i = 0; i < static_cast<long>(out2.DenseTotalLengthHost); ++i)
        {
            cudaGemmResult2[i] = out2.GetMutableDenseHost()[i];
        }

        Compute::Initialize::Zeros(out2);
        Compute::Add(out2, A2, B2);
        Compute::Add(out2, A2, out2);
        Compute::Add(out2, B2, out2);

#pragma omp parallel for default(shared) schedule(static)
        for (long i = 0; i < static_cast<long>(out.DenseTotalLengthHost); ++i)
        {
            cudaGemmResult1[i] = out.GetMutableDenseHost()[i];
        }

        Compute::Initialize::Zeros(out);
        Compute::Add(out, A, B);
        Compute::Add(out, A, out);
        Compute::Add(out, B, out);

        std::atomic largestError = 0.0f;

#pragma omp parallel for default(shared) schedule(static)
        for (long i = 0; i < static_cast<long>(out.DenseTotalLengthHost); ++i)
        {
            auto error = std::abs(cudaGemmResult1[i] - out.GetMutableDenseHost()[i]);
            if (largestError < error)
                largestError = error;

            CHECK(error <= std::abs(out.GetMutableDenseHost()[i] / 100.0f));
        }

        std::cout << "Largest error : " << largestError << std::endl;

#pragma omp parallel for default(shared) schedule(static)
        for (long i = 0; i < static_cast<long>(out2.DenseTotalLengthHost); ++i)
        {
            auto error = std::abs(cudaGemmResult2[i] - out2.GetMutableDenseHost()[i]);
            if (largestError < error)
                largestError = error;

            CHECK(error <= std::abs(out2.GetMutableDenseHost()[i] / 100.0f));
        }

        std::cout << "Largest error : " << largestError << std::endl;

        delete[] cudaGemmResult1;
        delete[] cudaGemmResult2;
    }
    Util::ResourceManager::ClearAll();
}

void TestAddBroadcast2()
{
    for (int j = 0; j < 5; j++)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distribution(1, 32);

        const unsigned int M = distribution(gen);
        const unsigned int N = distribution(gen);
        const auto batchSize = distribution(gen) % 5 + 1;

        std::cout << "M : " << M << " N: " << N << " batchSize : " << batchSize
            << std::endl;

        const Shape shapeA({ 1, M, N });
        const Shape shapeB({ N, M, N });
        const Shape shapeOut({ 1, M, N });

        const Device cuda(0, "device0");
        const Device host("host");

        TensorUtil::TensorData A(shapeA, Type::Dense, cuda, 1);
        TensorUtil::TensorData B(shapeB, Type::Dense, cuda, batchSize);
        TensorUtil::TensorData out(shapeOut, Type::Dense, cuda, batchSize);

        Compute::Initialize::Normal(A, 10, 1);
        Compute::Initialize::Normal(B, 10, 4);

        Compute::Add(out, A, B);
        Compute::Add(out, A, out);
        Compute::Add(out, B, out);

        A.SendTo(host);
        B.SendTo(host);
        out.SendTo(host);

        auto* cudaGemmResult = new float[out.DenseTotalLengthHost];

#pragma omp parallel for default(shared) schedule(static)
        for (long i = 0; i < static_cast<long>(out.DenseTotalLengthHost); ++i)
        {
            cudaGemmResult[i] = out.GetMutableDenseHost()[i];
        }

        Compute::Initialize::Zeros(out);
        Compute::Add(out, A, B);
        Compute::Add(out, A, out);
        Compute::Add(out, B, out);

        std::atomic<float> largestError = 0.0f;

        for (size_t i = 0; i < out.DenseTotalLengthHost; ++i)
        {
            auto error = std::abs(cudaGemmResult[i] - out.GetMutableDenseHost()[i]);
            if (largestError < error)
                largestError = error;

            CHECK(error <= std::abs(out.GetMutableDenseHost()[i] / 100.0f));
            if (error > std::abs(out.GetMutableDenseHost()[i] / 100.0f))
                break;
        }

        std::cout << "Largest error : " << largestError << std::endl;
        delete[] cudaGemmResult;
    }
    Util::ResourceManager::ClearAll();
}
} // namespace Sapphire::Test
