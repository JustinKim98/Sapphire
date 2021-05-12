// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Sapphire_CUDAFUNCTIONALITYTEST_CUH
#define Sapphire_CUDAFUNCTIONALITYTEST_CUH

namespace Sapphire::Test
{
int PrintCudaVersion();

int MallocTest();

int CublasTest();
}  // namespace Sapphire::Test

#endif  // Sapphire_CUDAFUNTIONALITYTEST_HPP
