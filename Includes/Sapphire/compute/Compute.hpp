// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef SAPPHIRE_COMPUTE_COMPUTE_DECL_HPP
#define SAPPHIRE_COMPUTE_COMPUTE_DECL_HPP

#include <Sapphire/tensor/TensorData.hpp>

namespace Sapphire::Compute
{
using namespace TensorUtil;

//! Warning
//! These operations does not check validity of the inputs
//! If input data condition does not meet, it will cause unhandled errors
//! All operations requires TensorData on the same device! (This should be checked previously before calling the function)

//! Performs y = a + b
void Add(TensorData& y, const TensorData& a, const TensorData& b);

//! Performs y = a - b
void Sub(TensorData& y, const TensorData& a, const TensorData& b);

//! Performs GEMM (y = a*b + c)
void Gemm(TensorData& y, const TensorData& a, const TensorData& b,
          const TensorData& c);

//! x, y, filter must have shape of C,H,W with Same batch size N
void Conv2DForward(TensorData& y, const TensorData& x,
                   const TensorData& filter, int strideRow, int strideCol,
                   int dilationRow, int dilationCol, int rowPadding,
                   int columnPadding);

void MaxPool2DForward(TensorData& y, const TensorData& x, int windowHeight,
                      int windowWidth, int strideRow, int strideCol,
                      int rowPadding, int columnPadding);

void AvgPool2DForward(TensorData& y, const TensorData& x, int windowHeight,
                      int windowWidth, int strideRow, int strideCol,
                      int rowPadding, int columnPadding);

//! Performs y = x*factor
void Scale(TensorData& y, const TensorData& x, float factor);

//! Performs y = TransposeKernel(x)
void Transpose(TensorData& y, const TensorData& x);

//! Performs Element-wise multiply
void Dot(TensorData& y, const TensorData& a, const TensorData& b);

//! Performs y = x^factor for each element
void Pow(TensorData& y, const TensorData& x, float factor);

void log(TensorData& y, const TensorData& x);

void log10(TensorData& y, const TensorData& x);

void exp(TensorData& y, const TensorData& x);

void Inverse(TensorData& y, const TensorData& x);

void Mean(TensorData& y, const TensorData& x);

void Mean(TensorData& y, const TensorData& x, int dim);

void InverseBackward(TensorData& dx, const TensorData& dy, const TensorData& x);

} // namespace Sapphire::Compute

#endif
