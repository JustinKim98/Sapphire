// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/compute/Compute.hpp>
#include <Motutapu/tensor/Shape.hpp>
#include <Motutapu/tensor/TensorData.hpp>
#include <Motutapu/util/Device.hpp>
#include "doctest.h"

namespace Motutapu::Test
{
void TestGemm()
{
    const auto M = 64;
    const auto N = 64;
    const auto K = 64;
    const Shape shapeA({ M, K });
    const Shape shapeB({ K, N });
    const Shape shapeC({ M, N });
    const Shape shapeOut({ M, N });

    const Device cuda(0, "device0");
    const Device host("host");

    const auto batchSize = 2;

    TensorUtil::TensorData A(shapeA, Type::Dense, host, batchSize);

    TensorUtil::TensorData B(shapeA, Type::Dense, host, batchSize);

    TensorUtil::TensorData C(shapeA, Type::Dense, host, batchSize);

    TensorUtil::TensorData Out(shapeA, Type::Dense, host, batchSize);

    //! TODO : Write initialize kernel

    Compute::Gemm(Out, A, B, C);

    float cpuGemmResult[Out.DenseTotalLength];

    for (size_t i = 0; i < Out.DenseTotalLength; ++i)
    {
        cpuGemmResult[i] = Out.DenseMatHost[i];
    }

    A.SendTo(cuda);
    B.SendTo(cuda);
    C.SendTo(cuda);
    Out.SendTo(cuda);

    Compute::Gemm(Out, A, B, C);

    Out.SendTo(host);

    for (size_t i = 0; i < Out.DenseTotalLength; ++i)
    {
        CHECK(static_cast<int>(cpuGemmResult[i]) ==
              static_cast<int>(Out.DenseMatHost[i]));
    }
}

}  // namespace Motutapu::Test
