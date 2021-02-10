// Copyright (c) 2021, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/Tests/BasicComputationTest.hpp>
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
void TestTranspose(bool printResult)
{
    for (int j = 0; j < 10; j++)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(1, 100);

        const unsigned int M = distrib(gen);
        const unsigned int N = distrib(gen);
        const auto batchSize = distrib(gen) % 5 + 1;

        //        const unsigned int M = 1;  // distrib(gen);
        //        const unsigned int N = 7;  // distrib(gen);
        //        const auto batchSize = 3;  // distrib(gen) % 5 + 1;

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
        //        Compute::Initialize::Ones(A);
        //        Compute::Initialize::Ones(B);

        Compute::Add(out, A, B);
        Compute::Scale(out, out, 2);
        Compute::Transpose(transposedOut, out);

        float cpuResult[transposedOut.DenseTotalLengthHost];

#pragma omp parallel for default(shared) schedule(static)
        for (size_t i = 0; i < transposedOut.DenseTotalLengthHost; ++i)
        {
            cpuResult[i] = transposedOut.DenseMatHost[i];
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
                        << out.DenseMatHost[offset +
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
                    std::cout << transposedOut.DenseMatHost
                                     [offsetTransposed +
                                      rowIdx * transposedOut.PaddedHostColSize +
                                      colIdx]
                              << " ";
                }
                std::cout << std::endl;
            }
        }

#pragma omp parallel for default(shared) schedule(static)
        for (size_t i = 0; i < transposedOut.DenseTotalLengthHost; ++i)
        {
            auto error = std::abs(cpuResult[i] - transposedOut.DenseMatHost[i]);
            if (largestError < error)
                largestError = error;

            CHECK(error <= 1.0f);
            if (error > 1.0f)
                std::cout << i / (transposedOut.Rows() *
                                  transposedOut.PaddedHostColSize)
                          << std::endl;
        }

        std::cout << "Largest error : " << largestError << std::endl;
    }

    Util::MemoryManager::ClearCudaMemoryPool();
    Util::MemoryManager::ClearHostMemoryPool();
}

void TestBasics1()
{
    for (int j = 0; j < 10; j++)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(1, 100);

        const unsigned int M = distrib(gen);
        const unsigned int N = distrib(gen);
        const auto batchSize = distrib(gen) % 5 + 1;

        //        const unsigned int M = 10;   // distrib(gen);
        //        const unsigned int N = 235;  // distrib(gen);
        //        const auto batchSize = 2;    // distrib(gen) % 5 + 1;

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

        float cpuGemmResult[Out.DenseTotalLengthHost];

#pragma omp parallel for default(shared) schedule(static)
        for (size_t i = 0; i < Out.DenseTotalLengthHost; ++i)
        {
            cpuGemmResult[i] = Out.DenseMatHost[i];
        }

        Compute::Initialize::Zeros(Out);

        A.SendTo(cuda);
        B.SendTo(cuda);
        Out.SendTo(cuda);

        Compute::Add(Out, A, B);
        Compute::Scale(Out, Out, 3);
        Compute::Sub(Out, Out, A);

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

            CHECK(error <= 1.0f);
        }

        std::cout << "Largest error : " << largestError << std::endl;
    }

    Util::MemoryManager::ClearCudaMemoryPool();
    Util::MemoryManager::ClearHostMemoryPool();
}

void TestBasics2()
{
    for (int j = 0; j < 10; j++)
    {
        std::random_device
            rd;  // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(
            rd());  // Standard mersenne_twister_engine seeded with rd()
        std::uniform_int_distribution<> distrib(8, 16);

        const unsigned int M = distrib(gen);
        const unsigned int N = distrib(gen);
        const auto batchSize = 3;  // distrib(gen) % 3 + 1;

        //        const unsigned int M = distrib(gen);
        //        const unsigned int N = distrib(gen);
        //        const auto batchSize = distrib(gen) % 5 + 1;

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

        A.SendTo(host);
        B.SendTo(host);
        out.SendTo(host);

        float cudaGemmResult[out.DenseTotalLengthHost];

#pragma omp parallel for default(shared) schedule(static)
        for (size_t i = 0; i < out.DenseTotalLengthHost; ++i)
        {
            cudaGemmResult[i] = out.DenseMatHost[i];
        }

        Compute::Initialize::Zeros(out);
        Compute::Dot(out, A, B);
        Compute::Add(out, out, A);
        Compute::Sub(out, out, B);

        std::atomic<float> largestError = 0.0f;

        //#pragma omp parallel for default(shared) schedule(static)
        for (size_t i = 0; i < out.DenseTotalLengthHost; ++i)
        {
            auto error = std::abs(cudaGemmResult[i] - out.DenseMatHost[i]);
            if (largestError < error)
                largestError = error;
            //
            //            std::cout << "cuda : " << cudaGemmResult[i]
            //                      << " cpu : " << out.DenseMatHost[i] <<
            //                      std::endl;

            CHECK(error <= std::abs(out.DenseMatHost[i] / 100.0f));
        }

        std::cout << "Largest error : " << largestError << std::endl;
    }
    Util::MemoryManager::ClearCudaMemoryPool();
    Util::MemoryManager::ClearHostMemoryPool();
}

void TestAddBroadcast1()
{
    for (int j = 0; j < 10; j++)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(1, 32);

        const unsigned int M = distrib(gen);
        const unsigned int N = distrib(gen);
        const auto batchSize = distrib(gen) % 5 + 1;

        std::cout << "M : " << M << " N: " << N << " batchSize : " << batchSize
                  << std::endl;

        const Shape shapeA({ M, M, N });
        const Shape shapeB({ M, M, N });
        const Shape shapeOut({ M, M, N });

        const Device cuda(0, "device0");
        const Device host("host");

        TensorUtil::TensorData A(shapeA, Type::Dense, cuda, batchSize);
        TensorUtil::TensorData B(shapeB, Type::Dense, cuda, batchSize);
        TensorUtil::TensorData out(shapeOut, Type::Dense, cuda, batchSize);

        Compute::Initialize::Normal(A, 100, 1);
        Compute::Initialize::Normal(B, 100, 4);
        Compute::Initialize::Zeros(out);

        const unsigned int M2 = distrib(gen);
        const unsigned int N2 = distrib(gen);
        const auto batchSize2 = distrib(gen) % 5 + 1;

        std::cout << "M2 : " << M2 << " N2: " << N2
                  << " batchSize : " << batchSize2 << std::endl;

        const Shape shapeA2({ N2, 1, M2, N2 });
        const Shape shapeB2({ M2, N2 });
        const Shape shapeOut2({ N2, M2, M2, N2 });

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

        float cudaGemmResult[out.DenseTotalLengthHost];

        float cudaGemmResult2[out2.DenseTotalLengthHost];

#pragma omp parallel for default(shared) schedule(static)
        for (size_t i = 0; i < out2.DenseTotalLengthHost; ++i)
        {
            cudaGemmResult2[i] = out2.DenseMatHost[i];
        }

        Compute::Initialize::Zeros(out2);
        Compute::Add(out2, A2, B2);
        Compute::Add(out2, A2, out2);
        Compute::Add(out2, B2, out2);

#pragma omp parallel for default(shared) schedule(static)
        for (size_t i = 0; i < out.DenseTotalLengthHost; ++i)
        {
            cudaGemmResult[i] = out.DenseMatHost[i];
        }

        Compute::Initialize::Zeros(out);
        Compute::Add(out, A, B);
        Compute::Add(out, A, out);
        Compute::Add(out, B, out);

        std::atomic<float> largestError = 0.0f;

#pragma omp parallel for default(shared) schedule(static)
        for (size_t i = 0; i < out.DenseTotalLengthHost; ++i)
        {
            auto error = std::abs(cudaGemmResult[i] - out.DenseMatHost[i]);
            if (largestError < error)
                largestError = error;
            //            std::cout << "cuda : " << cudaGemmResult[i]
            //                      << " cpu : " << out.DenseMatHost[i] <<
            //                      std::endl;

            CHECK(error <= std::abs(out.DenseMatHost[i] / 100.0f));
        }

        std::cout << "Largest error : " << largestError << std::endl;

        std::atomic<float> largestError2 = 0.0f;

#pragma omp parallel for default(shared) schedule(static)
        for (size_t i = 0; i < out2.DenseTotalLengthHost; ++i)
        {
            auto error = std::abs(cudaGemmResult2[i] - out2.DenseMatHost[i]);
            if (largestError < error)
                largestError = error;

            //            std::cout << "cuda : " << cudaGemmResult[i]
            //                      << " cpu : " << out.DenseMatHost[i] <<
            //                      std::endl;

            CHECK(error <= std::abs(out2.DenseMatHost[i] / 100.0f));
        }

        std::cout << "Largest error : " << largestError << std::endl;
    }
    Util::MemoryManager::ClearCudaMemoryPool();
    Util::MemoryManager::ClearHostMemoryPool();
}

void TestAddBroadcast2()
{
    for (int j = 0; j < 10; j++)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(1, 32);

        const unsigned int M = distrib(gen);
        const unsigned int N = distrib(gen);
        const auto batchSize = distrib(gen) % 10 + 1;

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

        float cudaGemmResult[out.DenseTotalLengthHost];

#pragma omp parallel for default(shared) schedule(static)
        for (size_t i = 0; i < out.DenseTotalLengthHost; ++i)
        {
            cudaGemmResult[i] = out.DenseMatHost[i];
        }

        Compute::Initialize::Zeros(out);
        Compute::Add(out, A, B);
        Compute::Add(out, A, out);
        Compute::Add(out, B, out);

        std::atomic<float> largestError = 0.0f;

        for (size_t i = 0; i < out.DenseTotalLengthHost; ++i)
        {
            auto error = std::abs(cudaGemmResult[i] - out.DenseMatHost[i]);
            if (largestError < error)
                largestError = error;

            //            std::cout << "cuda : " << cudaGemmResult[i]
            //                      << " cpu : " << out.DenseMatHost[i] <<
            //                      std::endl;

            CHECK(error <= std::abs(out.DenseMatHost[i] / 100.0f));
            if (error > std::abs(out.DenseMatHost[i] / 100.0f))
                break;
        }

        std::cout << "Largest error : " << largestError << std::endl;
    }
    Util::MemoryManager::ClearCudaMemoryPool();
    Util::MemoryManager::ClearHostMemoryPool();
}

}  // namespace Motutapu::Test