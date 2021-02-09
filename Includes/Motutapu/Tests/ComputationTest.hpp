// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_COMPUTATIONTEST_HPP
#define MOTUTAPU_COMPUTATIONTEST_HPP

namespace Motutapu::Test
{
#ifdef WITH_CUDA
void TestGemm1();

void TestGemm2();

void TestGemmBroadcast();

#endif
}  // namespace Motutapu::Test

#endif  // MOTUTAPU_COMPUTATIONTEST_HPP
