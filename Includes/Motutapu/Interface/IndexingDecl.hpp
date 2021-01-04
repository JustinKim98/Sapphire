// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_INDEXING_DECL_HPP
#define MOTUTAPU_INDEXING_DECL_HPP

#include <functional>
#include <Motutapu/tensor/TensorDecl.hpp>

namespace Motutapu
{
//! Returns a new tensor with all the dimensions of input of size 1 removed
//! \tparam T : data type of the tensor
//! \param tensor : input tensor to squeeze
//! \return : squeezed tensor. Returned tensor is independent from the input
template <typename T>
Tensor<T> Squeeze(const Tensor<T>& tensor);

//! Returns a new tensor with all the dimensions of given dimension of size 1 removed
//! \tparam T : data type of the tensor
//! \param tensor : input tensor to squeeze
//! \param dim : dimension to squeeze
//! \return : squeezed tensor. Returned tensor is independent from the input
template <typename T>
Tensor<T> Squeeze(const Tensor<T>& tensor, int dim);

//! Removes all dimensions with size 1
//! modifies the given tensor
//! \tparam T : ata type of the tensor
//! \param tensor : input tensor to squeeze
template <typename T>
void Squeeze(Tensor<T>& tensor);

//! Removes given dimension if dimension is size 1
//! modifies the given tensor
//! \tparam T : data type of the tensor
//! \param tensor : input tensor to squeeze
//! \param dim : dimension to remove
template <typename T>
void Squeeze(Tensor<T>& tensor, int dim);

//! Returns a new tensor with a dimension of size one inserted at the specified position
//! \tparam T  : data type of the tensor
//! \param tensor : input tensor to unSqueeze
//! \param dim : dimension to insert one
template <typename T>
Tensor<T> UnSqueeze(const Tensor<T>& tensor, int dim);

//! Inserts dimension of size one at the given tensor
//! \tparam T : data type of the tensor
//! \param tensor : input tensor to unSqueeze
//! \param dim : dimension to insert one
template <typename T>
void UnSqueeze(Tensor<T>& tensor, int dim);

//! Concatenates the given sequence of seq tensors in the given dimension.
//! All tensors must either have the same shape(except in the concatenating dimension)
//! or be empty
//! \tparam T : Data type of tensor
//! \param tensors : Vector of tensors to concatenate
//! \param dim : The dimension over which the tensors are concatenated.
template <typename T>
Tensor<T> Cat(std::vector<Tensor<T>> tensors, int dim);

//! Stacks tensors in sequence depthWise (3rd dimension wise)
//! Expands tensors into 3-dimension and concatenates them along 3rd dimension
//! \tparam T : Data type of tensor
//! \param tensors : Vector of tensors to concatenate
template <typename T>
Tensor<T> DStack(const std::vector<Tensor<T>>& tensors);

//! Stacks tensors in sequence rowWise (2nd dimension wise)
//! Expands tensors into 3-dimension and concatenates them along 3rd dimension
//! \tparam T : Data type of tensor
//! \param tensors : Vector of tensors to concatenate
template <typename T>
Tensor<T> HStack(const std::vector<Tensor<T>>& tensors);

//! Stacks tensors in sequence columnWise (1st dimension wise)
//! Expands tensors into 3-dimension and concatenates them along 3rd dimension
//! \tparam T : Data type of tensor
//! \param tensors : Vector of tensors to concatenate
template <typename T>
Tensor<T> VStack(const std::vector<Tensor<T>>& tensors);

//! Creates new tensor expanded to given dimension
//! Tensors with more dimension than given dimension is returned as-is
//! \tparam T : Data type of tensor
//! \param tensor : tensor to expand from
//! \param dim : dimension to expand
//! \return : expanded tensor
template <typename T>
Tensor<T> Expand(const Tensor<T>& tensor, int dim);

//! Expands tensor to given dimension
//! Tensors with more dimension than given dimension is returned as-is
//! \tparam T : Data type of tensor
//! \param dim : dimension to expand
//! \param tensor : tensor to expand from
template <typename T>
void Expand(Tensor<T>& tensor, int dim);

//! Creates new tensor converted into one dimension
//! Tensors with one dimension is returned as-is
//! \tparam T : Data type of tensor
//! \param tensor : tensor 
template <typename T>
Tensor<T> Flatten(Tensor<T>& tensor);

//! Converts tensor to one dimension
//! Tensors with one dimension is returned as-is
//! \tparam T : Data type of tensor
//! \param tensor : tensor to flatten
template <typename T>
void Flatten(Tensor<T>& tensor);

//! Splits tensor into a specific number of chunks.
//! \tparam T : Data type of tensor
//! \param tensor : tensor to split
//! \param dim : dimension along which to split the tensor
//! \return : vector of split by tensor
template <typename T>
std::vector<Tensor<T>> Chunk(const Tensor<T>& tensor, int dim);

//! Gathers values along an axis specified by dim
//! \tparam T : Data type of tensor
//! \param tensor : input tensor
//! \param dim : the axis along which to index
//! \param index : the indices to 
template <typename T>
Tensor<T> Gather(const Tensor<T>& tensor, int dim, const Tensor<T>& index);

//! Returns a new tensor that is narrowed version of input tensor
//! The dimension dim is input from start to start + length
//! \tparam T : Data type of tensor
//! \param tensor : tensor to narrow
//! \param dim : dimension along which to narrow
//! \param start : starting dimension
//! \param length : distance to the ending dimension
//! \return : Narrowed tensor
template <typename T>
Tensor<T> Narrow(const Tensor<T>& tensor, int dim, int start, int length);

//! Narrows the given tensor
//! The dimension dim is input from start to start + length
//! \tparam T : Data type of tensor
//! \param tensor : tensor to narrow
//! \param dim : dimension along which to narrow
//! \param start : starting dimension
//! \param length : distance to the ending dimension
template<typename T>
void Narrow(Tensor<T>& tensor, int dim, int start, int length);

//! Returns indices with nonzero value
//! \tparam T : Data type of tensor
//! \param tensor : tensor to search for nonzero
//! \return : tensor with nonzero indices from input tensor
template <typename T>
Tensor<T> Nonzero(const Tensor<T>& tensor);

//! Splits given tensor with given size
//! \tparam T : Data type of tensor
//! \param tensor : tensor to split
//! \param splitSize : vector of split size
//! \param dim : dimension to split
//! \return : vector of split tensors
template <typename T>
std::vector<Tensor<T>> Split(const Tensor<T>& tensor, std::vector<int> splitSize, int dim);

//! Returns a new tensor with the elements of input as the given indices
//! The input tensor is treated as if it were viewed as a 1-D tensor
//! The result takes the same shape as the indices
//! \tparam T : Data type of tensor
//! \param tensor : tensor to perform take
//! \param index : vector of indices to take
//! \return : output tensor
template <typename T>
Tensor<T> Take(const Tensor<T>& tensor, std::vector<int> index);

//! Creates new tensor with transposed value from input tensor
//! First and second dimension is swapped
//! \param tensor : tensor to transpose
//! \return : transposed tensor
template <typename T>
Tensor<T> Transpose(const Tensor<T>& tensor);

//! Transposes the given tensor
//! First and second dimension is swapped
//! \param tensor : tensor to transpose
template <typename T>
void Transpose(Tensor<T>& tensor);

//! Creates new tensor with given dimension removed
//! \param tensor : tensor given to remove dimension
//! \param dim : dimension to unbind
//! \return : tensor with given dimension removed
template <typename T>
Tensor<T> Unbind(const Tensor<T>& tensor, int dim);

//! Removes dimension from the given tensor
//! \param tensor : tensor given to remove dimension
//! \param dim : dimension to unbind
template <typename T>
void Unbind(Tensor<T>& tensor, int dim);

//! Selects between tensor and value using the condition for each entry
//! When condition is true, tensor is selected. Otherwise, value is selected.
//! \param condition : condition to determine between tensor and given value
//! \param tensor : tensor to insert when condition is true
//! \param value : value to insert when condition is false
//! \return : tensor with value selected between two candidates
template <typename T>
Tensor<T> Where(std::function<bool(T)> condition, const Tensor<T>& tensor,
                T value);

//! Selects between tensorA and tensorB using the condition for each entry
//! When condition is true, tensor is selected. Otherwise, value is selected.
//! tensorA and tensorB must be broadcastable
//! \param condition : condition to determine between tensor and given value
//! \param tensorA : tensor to insert when condition is true
//! \param tensorB : tensor to insert when condition is false
//! \return : tensor with value selected between two candidates
template <typename T>
Tensor<T> Where(std::function<bool(T, T)> condition, const Tensor<T>& tensorA,
                const Tensor<T>& tensorB);

//! Replaces value of each entry in tensor to corresponding
//! entry in alternative if condition is false
//! \tparam T : Data type of tensor
//! \param condition : Condition to select default(tensor) or alternative
//! \param tensor : tensor to filter
//! \param alternative : value to fill if condition is false
template <typename T>
void Filter(std::function<bool(T)> condition, Tensor<T>& tensor, T alternative);

//! Replaces value of each entry in tensor to corresponding
//! entry in alternative if condition is false
//! \tparam T : Data type of tensor
//! \param condition : Condition to select default(tensor) or alternative
//! \param tensor : tensor to filter
//! \param alternative : value to fill if condition is false
template <typename T>
void Filter(std::function<bool(T)> condition, Tensor<T>& tensor,
            const Tensor<T>& alternative);
}

#endif
