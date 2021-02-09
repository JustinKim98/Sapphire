// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_CUDAFUNCTIONALITYTEST_CUH
#define MOTUTAPU_CUDAFUNCTIONALITYTEST_CUH

namespace Motutapu::Test
{
int PrintCudaVersion();

int MallocTest();

int CublasTest();
}  // namespace Motutapu::Test

#endif  // MOTUTAPU_CUDAFUNTIONALITYTEST_HPP
