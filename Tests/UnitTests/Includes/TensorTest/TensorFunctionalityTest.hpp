// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_TEST_TENSOR_FUNCTIONALITY_TEST_HPP
#define SAPPHIRE_TEST_TENSOR_FUNCTIONALITY_TEST_HPP

namespace Sapphire::Test
{
//! Tests whether sending data between devices work well
void SendDataBetweenHostDevice(bool print);

//! Tests whether data is preserved when copying
void TensorDataCopyOnCuda(bool print);

//! Tests whether data is preserved when copying
void TensorDataCopyOnHost(bool print);

}

#endif
