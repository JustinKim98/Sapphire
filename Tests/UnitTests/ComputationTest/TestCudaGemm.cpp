// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include "TestCudaGemm.hpp"
#include <iostream>
#include <Motutapu/compute/naive/NaiveGemm.hpp>
#include <Motutapu/compute/cuda/dense/Gemm.cuh>

namespace Motutapu::Test
{

void TensorGemmTest()
{
    const auto M = 64;
    const auto N = 64;
    const auto K = 64;
    const Shape shapeA({ M, K });
    const Shape shapeB({ K, N });
    const Shape shapeC({ M, N });
    const Shape shapeOut({ M, N });

    const Device CudaDevice(1, DeviceType::CUDA, "device1");
    const Device HostDevice(0, DeviceType::CPU, "device0");

    const auto batchSize = 2;
    auto* cudaA = Util::TensorData<half>::CreateTensorData(shapeA, CudaDevice,
                                                           Util::Type::Dense, batchSize);

    auto* cudaB = Util::TensorData<half>::CreateTensorData(shapeB, CudaDevice,
        Util::Type::Dense, batchSize);

    auto* cudaC = Util::TensorData<half>::CreateTensorData(shapeC, CudaDevice,
        Util::Type::Dense, batchSize);

    auto* cudaOut = Util::TensorData<half>::CreateTensorData(
        shapeOut, CudaDevice, Util::Type::Dense, batchSize);

    auto* A = Util::TensorData<float>::CreateTensorData(shapeA, CudaDevice,
        Util::Type::Dense, batchSize);

    auto* B = Util::TensorData<float>::CreateTensorData(shapeB, CudaDevice,
        Util::Type::Dense, batchSize);

    auto* C = Util::TensorData<float>::CreateTensorData(shapeC, CudaDevice,
        Util::Type::Dense, batchSize);

    auto* Out = Util::TensorData<float>::CreateTensorData(shapeOut, CudaDevice,
        Util::Type::Dense, batchSize);

    Cuda::Dense::GemmTensor(
        cudaOut->DenseMatCuda, cudaA->DenseMatCuda, cudaB->DenseMatCuda,
        cudaC->DenseMatCuda, cudaOut->PaddedRowSize, cudaOut->PaddedColumnSize,
        cudaA->PaddedColumnSize, batchSize, false, false, false);

    Naive::Gemm<float>(Out->DenseMatHost, A->DenseMatHost, B->DenseMatHost,
                C->DenseMatHost, Out->PaddedRowSize, Out->PaddedColumnSize,
                A->PaddedColumnSize, batchSize, false, false, false);

    auto maxDiff = 0.0f;

    for (size_t i = 0;
         i < Out->PaddedRowSize * Out->PaddedColumnSize * batchSize; ++i)
    {
        auto diff = __half2float(*(cudaOut->DenseMatHost + i)) - *(Out->DenseMatHost + i);
        maxDiff = (maxDiff > diff) ? maxDiff : diff;
    }

    std::cout << "Maximum error : %f" << maxDiff;
}

void FloatGemmTest()
{
    const auto M = 64;
    const auto N = 64;
    const auto K = 64;
    const Shape shapeA({ M, K });
    const Shape shapeB({ K, N });
    const Shape shapeC({ M, N });
    const Shape shapeOut({ M, N });

    const Device CudaDevice(1, DeviceType::CUDA, "device1");
    const Device HostDevice(0, DeviceType::CPU, "device0");

    const auto batchSize = 2;
    auto* cudaA = Util::TensorData<float>::CreateTensorData(shapeA, CudaDevice,
        Util::Type::Dense, batchSize);

    auto* cudaB = Util::TensorData<float>::CreateTensorData(shapeB, CudaDevice,
        Util::Type::Dense, batchSize);

    auto* cudaC = Util::TensorData<float>::CreateTensorData(shapeC, CudaDevice,
        Util::Type::Dense, batchSize);

    auto* cudaOut = Util::TensorData<float>::CreateTensorData(
        shapeOut, CudaDevice, Util::Type::Dense, batchSize);

    auto* A = Util::TensorData<float>::CreateTensorData(shapeA, CudaDevice,
        Util::Type::Dense, batchSize);

    auto* B = Util::TensorData<float>::CreateTensorData(shapeB, CudaDevice,
        Util::Type::Dense, batchSize);

    auto* C = Util::TensorData<float>::CreateTensorData(shapeC, CudaDevice,
        Util::Type::Dense, batchSize);

    auto* Out = Util::TensorData<float>::CreateTensorData(shapeOut, CudaDevice,
        Util::Type::Dense, batchSize);

    Util::TensorData<float>::m_copyHostToGpu(cudaA,);
    Util::TensorData<float>::m_copyHostToGpu(cudaB,);
    Util::TensorData<float>::m_copyHostToGpu(cudaC,);

    Cuda::Dense::GemmNormalFloat(
        cudaOut->DenseMatCuda, cudaA->DenseMatCuda, cudaB->DenseMatCuda,
        cudaC->DenseMatCuda, cudaOut->PaddedRowSize, cudaOut->PaddedColumnSize,
        cudaA->PaddedColumnSize, batchSize, Util::Type::Dense, Util::Type::Dense, Util::Type::Dense);

    Util::TensorData<float>::m_copyGpuToHost(cudaOut,);

    Naive::Gemm<float>(Out->DenseMatHost, A->DenseMatHost, B->DenseMatHost,
                C->DenseMatHost, Out->PaddedRowSize, Out->PaddedColumnSize,
                A->PaddedColumnSize, batchSize, Util::Type::Dense, Util::Type::Dense, Util::Type::Dense);

    auto maxDiff = 0.0f;

    for (size_t i = 0;
         i < Out->PaddedRowSize * Out->PaddedColumnSize * batchSize; ++i)
    {
        auto diff = __half2float(*(cudaOut->DenseMatHost + i)) -
                    *(Out->DenseMatHost + i);
        maxDiff = (maxDiff > diff) ? maxDiff : diff;
    }

    std::cout << "Maximum error : %f" << maxDiff;
}

}
