// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_UTIL_DENSEMATRIX_HPP
#define MOTUTAPU_UTIL_DENSEMATRIX_HPP

#include <Motutapu/util/SpanDecl.hpp>

namespace Motutapu::Util
{
template <typename T>
struct DenseMatrix
{
    T* Data;
    unsigned long Length;

    unsigned int NumRows;
    unsigned int NumCols;
    unsigned int PadSize;

    static DenseMatrix AllocateOnCPU();
    static void CopyToCPU(DenseMatrix& dest, const DenseMatrix& src);
    static void CopyCPU(DenseMatrix& dest, const DenseMatrix& src);

#ifdef WITH_CUDA
    static DenseMatrix AllocateOnGPU();
    static void CopyToGPU(DenseMatrix& dest, const DenseMatrix& src);
    static void CopyGPU(DenseMatrix& dest, const DenseMatrix& src);
#endif
};

}

#endif