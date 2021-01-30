// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include "TestCudaGemm.hpp"

#include <iostream>

namespace Motutapu::Test
{

void TensorGemmTest()
{
//    const auto M = 64;
//    const auto N = 64;
//    const auto K = 64;
//    const Shape shapeA({ M, K });
//    const Shape shapeB({ K, N });
//    const Shape shapeC({ M, N });
//    const Shape shapeOut({ M, N });
//
//    const Device cuda(0, "device0");
//    const Device host("host");
//
//    const auto batchSize = 2;
//
//    float* otherPtr;
//
//    Util::TensorData A(shapeA, Type::Dense, host, batchSize);
//
//    Util::TensorData B(shapeA, Type::Dense, host, batchSize);
//
//    Util::TensorData C(shapeA, Type::Dense, host, batchSize);
//
//    Util::TensorData Out(shapeA, Type::Dense, host, batchSize);
//
//    Compute::Gemm(Out, A, B, C);
//
//    A.SendTo(cuda);
//    B.SendTo(cuda);
//    C.SendTo(cuda);
//    Out.SendTo(cuda);
//
//    Compute::Gemm(Out, A, B, C);
}

}  // namespace Motutapu::Test
