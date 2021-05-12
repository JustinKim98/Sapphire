// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Sapphire_INDEXINGINTERFACE_DECL_HPP
#define Sapphire_INDEXINGINTERFACE_DECL_HPP

#include <functional>
#include <Sapphire/tensor/Tensor.hpp>

namespace Sapphire
{
//! Returns a new tensor with all the dimensions of input of size 1 removed
//! \param tensor : input tensor to squeeze
//! \return : squeezed tensor. Returned tensor is independent from the input
Tensor Squeeze(const Tensor& tensor);

//! Returns a new tensor with all the dimensions of given dimension of size 1 removed
//! \param tensor : input tensor to squeeze
//! \param dim : dimension to squeeze
//! \return : squeezed tensor. Returned tensor is independent from the input
Tensor Squeeze(const Tensor& tensor, int dim);

//! Removes all dimensions with size 1
//! modifies the given tensor
//! \param tensor : input tensor to squeeze
void Squeeze(Tensor& tensor);

//! Removes given dimension if dimension is size 1
//! modifies the given tensor
//! \param tensor : input tensor to squeeze
//! \param dim : dimension to remove
void Squeeze(Tensor& tensor, int dim);

//! Returns a new tensor with a dimension of size one inserted at the specified position
//! \param tensor : input tensor to unSqueeze
//! \param dim : dimension to insert one
Tensor UnSqueeze(const Tensor& tensor, int dim);

//! Inserts dimension of size one at the given tensor
//! \param tensor : input tensor to unSqueeze
//! \param dim : dimension to insert one
void UnSqueeze(Tensor& tensor, int dim);

//! Concatenates the given sequence of seq tensors in the given dimension.
//! All tensors must either have the same shape(except in the concatenating dimension)
//! or be empty
//! \param tensors : Vector of tensors to concatenate
//! \param dim : The dimension over which the tensors are concatenated.
Tensor Cat(std::vector<Tensor> tensors, int dim);

//! Stacks tensors in sequence depthWise (3rd dimension wise)
//! Expands tensors into 3-dimension and concatenates them along 3rd dimension
//! \param tensors : Vector of tensors to concatenate
Tensor DStack(const std::vector<Tensor>& tensors);

//! Stacks tensors in sequence rowWise (2nd dimension wise)
//! Expands tensors into 3-dimension and concatenates them along 3rd dimension
//! \param tensors : Vector of tensors to concatenate
Tensor HStack(const std::vector<Tensor>& tensors);

//! Stacks tensors in sequence columnWise (1st dimension wise)
//! Expands tensors into 3-dimension and concatenates them along 3rd dimension
//! \param tensors : Vector of tensors to concatenate
Tensor VStack(const std::vector<Tensor>& tensors);

//! Creates new tensor expanded to given dimension
//! Tensors with more dimension than given dimension is returned as-is
//! \param tensor : tensor to expand from
//! \param dim : dimension to expand
//! \return : expanded tensor
Tensor Expand(const Tensor& tensor, int dim);

//! Expands tensor to given dimension
//! Tensors with more dimension than given dimension is returned as-is
//! \param dim : dimension to expand
//! \param tensor : tensor to expand from
void Expand(Tensor& tensor, int dim);

//! Creates new tensor converted into one dimension
//! Tensors with one dimension is returned as-is
//! \param tensor : tensor 
Tensor Flatten(const Tensor& tensor);

//! Converts tensor to one dimension
//! Tensors with one dimension is returned as-is
//! \param tensor : tensor to flatten
void Flatten(Tensor& tensor);

//! Splits tensor into a specific number of chunks.
//! \param tensor : tensor to split
//! \param dim : dimension along which to split the tensor
//! \return : vector of split by tensor
std::vector<Tensor> Chunk(const Tensor& tensor, int dim);

//! Gathers values along an axis specified by dim
//! \param tensor : input tensor
//! \param dim : the axis along which to index
//! \param index : the indices to 
Tensor Gather(const Tensor& tensor, int dim, const Tensor& index);

//! Returns a new tensor that is narrowed version of input tensor
//! The dimension dim is input from start to start + length
//! \param tensor : tensor to narrow
//! \param dim : dimension along which to narrow
//! \param start : starting dimension
//! \param length : distance to the ending dimension
//! \return : Narrowed tensor
Tensor Narrow(const Tensor& tensor, int dim, int start, int length);

//! Narrows the given tensor
//! The dimension dim is input from start to start + length
//! \param tensor : tensor to narrow
//! \param dim : dimension along which to narrow
//! \param start : starting dimension
//! \param length : distance to the ending dimension
void Narrow(Tensor& tensor, int dim, int start, int length);

//! Returns indices with nonzero value
//! \param tensor : tensor to search for nonzero
//! \return : tensor with nonzero indices from input tensor
Tensor Nonzero(const Tensor& tensor);

//! Splits given tensor with given size
//! \param tensor : tensor to split
//! \param splitSize : vector of split size
//! \param dim : dimension to split
//! \return : vector of split tensors
std::vector<Tensor> Split(const Tensor& tensor, std::vector<int> splitSize, int dim);

//! Returns a new tensor with the elements of input as the given indices
//! The input tensor is treated as if it were viewed as a 1-D tensor
//! The result takes the same shape as the indices
//! \param tensor : tensor to perform take
//! \param index : vector of indices to take
//! \return : output tensor
Tensor Take(const Tensor& tensor, std::vector<int> index);

//! Creates new tensor with transposed value from input tensor
//! First and second dimension is swapped
//! \param tensor : tensor to transpose
//! \return : transposed tensor
Tensor Transpose(const Tensor& tensor);

//! Transposes the given tensor
//! First and second dimension is swapped
//! \param tensor : tensor to transpose
void Transpose(Tensor& tensor);

//! Creates new tensor with given dimension removed
//! \param tensor : tensor given to remove dimension
//! \param dim : dimension to unbind
//! \return : tensor with given dimension removed
Tensor Unbind(const Tensor& tensor, int dim);

//! Removes dimension from the given tensor
//! \param tensor : tensor given to remove dimension
//! \param dim : dimension to unbind
void Unbind(Tensor& tensor, int dim);

//! Selects between tensor and value using the condition for each entry
//! When condition is true, tensor is selected. Otherwise, value is selected.
//! \param condition : condition to determine between tensor and given value
//! \param tensor : tensor to insert when condition is true
//! \param value : value to insert when condition is false
//! \return : tensor with value selected between two candidates
Tensor Where(std::function<bool(float)> condition, const Tensor& tensor,
                float value);

//! Selects between tensorA and tensorB using the condition for each entry
//! When condition is true, tensor is selected. Otherwise, value is selected.
//! tensorA and tensorB must be broadcastable
//! \param condition : condition to determine between tensor and given value
//! \param tensorA : tensor to insert when condition is true
//! \param tensorB : tensor to insert when condition is false
//! \return : tensor with value selected between two candidates

Tensor Where(std::function<bool(float, float)> condition, const Tensor& tensorA,
                const Tensor& tensorB);

//! Replaces value of each entry in tensor to corresponding
//! entry in alternative if condition is false
//! \param condition : Condition to select default(tensor) or alternative
//! \param tensor : tensor to filter
//! \param alternative : value to fill if condition is false
void Filter(std::function<bool(float)> condition, Tensor& tensor, float alternative);

//! Replaces value of each entry in tensor to corresponding
//! entry in alternative if condition is false
//! \param condition : Condition to select default(tensor) or alternative
//! \param tensor : tensor to filter
//! \param alternative : value to fill if condition is false
void Filter(std::function<bool(float)> condition, Tensor& tensor,
            const Tensor& alternative);
}

#endif
