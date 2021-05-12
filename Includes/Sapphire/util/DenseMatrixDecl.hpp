// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Sapphire_UTIL_DENSEMATRIX_DECL_HPP
#define Sapphire_UTIL_DENSEMATRIX_DECL_HPP

#include <Sapphire/util/SpanDecl.hpp>

namespace Sapphire::Util
{
template <typename T>
struct DenseMatrix
{
    T* Data;
    unsigned long Length;

    unsigned int NumRows;
    unsigned int NumCols;
    unsigned int PadSize;

   
};
}

#endif
