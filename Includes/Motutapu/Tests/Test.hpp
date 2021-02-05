#ifndef MOTUTAPU_TEST_HPP
#define MOTUTAPU_TEST_HPP

namespace Motutapu::Test
{
int Add(int a, int b);

#ifdef WITH_CUDA
void TestGemm1();

void TestGemm2();

void TestGemmBroadcast();

#endif
}  // namespace Motutapu::Test

#endif