#ifndef MOTUTAPU_TEST_HPP
#define MOTUTAPU_TEST_HPP

namespace Motutapu::Test
{
int Add(int a, int b);

#ifdef WITH_CUDA
void PrintCudaVersion();

void MallocTest();

#endif
}  // namespace Motutapu::Test

#endif