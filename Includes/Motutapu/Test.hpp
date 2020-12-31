#ifndef MOTUTAPU_TEST_HPP
#define MOTUTAPU_TEST_HPP

int Add(int a, int b);

#ifdef WITH_CUDA
void PrintCudaVersion();
#endif

#endif 