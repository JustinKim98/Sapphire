// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/Model.hpp>
#include <Sapphire/compute/BasicOps.hpp>
#include <Sapphire/operations/Backward/LinearBackward.hpp>
#include <Sapphire/operations/Forward/Linear.hpp>
#include <Sapphire/compute/Initialize.hpp>
#include <Sapphire/operations/Unit.hpp>
#include <Sapphire/util/UnitUtils.hpp>
#include <Sapphire/tensor/TensorData.hpp>
#include <Sapphire/tensor/CreateTensor.hpp>

namespace Sapphire::NN
{
int Linear::m_unitIdCount = 0;

Linear::Linear(int inputFeatureSize, int outputFeatureSize,
               bool isSparse)
    : Unit(std::string("Linear") + std::to_string(m_unitIdCount++)),
      m_inputs(inputFeatureSize),
      m_outputs(outputFeatureSize),
      m_isSparse(isSparse)
{
    if (m_isSparse)
        throw std::invalid_argument(
            "NN::Linear - Sparse version not implemented");

    auto sd = 1.0f / static_cast<float>(std::sqrt(inputFeatureSize));
    const Tensor weight = MakeTensor(
        Shape({ inputFeatureSize, outputFeatureSize }),
        M<Initialize::Uniform>(-sd, sd), true);
    const Tensor bias = MakeTensor(Shape({ outputFeatureSize }),
                                   M<Initialize::Uniform>(-sd, sd), true);
    m_trainableTensorMap["weight"] = weight;
    m_trainableTensorMap["bias"] = bias;
}

Linear::Linear(std::string name, int inputFeatureSize, int outputFeatureSize,
               bool isSparse)
    : Unit(std::move(name)),
      m_inputs(inputFeatureSize),
      m_outputs(outputFeatureSize),
      m_isSparse(isSparse)
{
    if (m_isSparse)
        throw std::invalid_argument(
            "NN::Linear - Sparse version not implemented");

    auto sd = 1.0f / static_cast<float>(std::sqrt(inputFeatureSize));
    const Tensor weight = MakeTensor(
        Shape({ inputFeatureSize, outputFeatureSize }),
        M<Initialize::Uniform>(-sd, sd), true);
    const Tensor bias = MakeTensor(Shape({ outputFeatureSize }),
                                   M<Initialize::Uniform>(-sd, sd), true);

    m_trainableTensorMap["weight"] = weight;
    m_trainableTensorMap["bias"] = bias;
}

Tensor Linear::operator()(Tensor& x)
{
    const Tensor weight = m_trainableTensorMap.at("weight");
    const Tensor bias = m_trainableTensorMap.at("bias");

    weight.SetDevice(x.GetDevice());
    bias.SetDevice(x.GetDevice());
    if (weight.Mode() != x.Mode())
    {
        if (x.Mode() == ComputeMode::Cuda)
            weight.ToCuda();
        else
            weight.ToHost();
    }

    if (bias.Mode() != x.Mode())
    {
        if (x.Mode() == ComputeMode::Cuda)
            bias.ToCuda();
        else
            bias.ToHost();
    }

    return this->operator()(x, weight, bias);
}

Tensor Linear::operator()(Tensor& x, Tensor weight, Tensor bias)
{
    auto mode = x.Mode();
    if (!Util::CheckModeEquality(mode, weight, bias))
        throw std::invalid_argument("NN::Linear - Device mode inequality");
    auto& model = ModelManager::CurModel();

    auto& xDesc =
        model.GetDescriptor(x.TensorDescriptorKey());
    m_checkArguments({ &xDesc });
    auto& weightDesc = model.GetDescriptor(weight.TensorDescriptorKey());
    auto& biasDesc = model.GetDescriptor(bias.TensorDescriptorKey());
    const auto yKey = m_registerOutputTensor(xDesc);
    auto& yDesc = model.GetDescriptor(yKey);
    yDesc.SetMode(mode);

    auto weightData = weightDesc.GetForwardData();
    auto biasData = biasDesc.GetForwardData();
    auto xData = xDesc.GetForwardData();
    auto dxData = xDesc.GetBackwardData();
    auto yData = yDesc.GetForwardData();
    auto dyData = yDesc.GetBackwardData();

    const auto batchSize = x.GetShape().GetNumUnits(1);
    auto transposedOnes = TensorUtil::TensorData(
        Shape({ batchSize, 1 }), Type::Dense,
        bias.GetDevice());
    transposedOnes.SetMode(bias.Mode());
    Compute::Initialize::Ones(transposedOnes);

    //! Change the dimension of the data to match the requirements
    Util::ChangeTensorDataDimension(2, xData, dxData, yData, dyData, biasData);

    auto expandedBias = TensorUtil::TensorData(
        yData.GetShape(), Type::Dense, bias.GetDevice());
    expandedBias.SetMode(bias.Mode());

    const auto shape0 = expandedBias.GetShape();
    const auto shape1 = transposedOnes.GetShape(); // check
    const auto shape2 = biasData.GetShape();
    const auto shape3 = yData.GetShape();
    const auto shape4 = weightData.GetShape();

    Compute::Initialize::Zeros(expandedBias);
    Compute::Gemm(yData, transposedOnes,
                  biasData);
    Compute::Gemm(yData, xData, weightData);

    auto* backPropWrapper =
        new BackProp::LinearBackProp(m_name,
                                     dxData, dyData, weightData, biasData,
                                     xData, batchSize);
    Util::SaveHistory(backPropWrapper, std::make_tuple(&xDesc),
                      std::make_tuple(&yDesc));

    return Tensor(yKey);
}

Tensor Linear::GetWeight() const
{
    return m_trainableTensorMap.at("weight");
}

Tensor Linear::GetBias() const
{
    return m_trainableTensorMap.at("bias");
}

int Linear::m_registerOutputTensor(
    const TensorUtil::TensorDescriptor& xDesc) const
{
    auto& model = ModelManager::CurModel();
    const auto x = xDesc.GetForwardData();
    const Shape xShape = xDesc.GetShape();
    Shape yShape = xShape;
    yShape[yShape.Dim() - 1] = m_outputs;
    const auto yKey = model.RegisterTensorDescriptor(
        yShape, xDesc.GetType(), xDesc.GetDevice());
    return yKey;
}

void Linear::m_checkArguments(
    std::vector<TensorUtil::TensorDescriptor*> arguments) const
{
    const auto input = arguments.at(0);
    if (input->GetShape().Cols() != m_inputs)
        throw std::invalid_argument("NN::Linear - Shape mismatch input: (" +
                                    std::to_string(input->GetShape().Cols()) +
                                    ") expected : (" + std::to_string(m_inputs)
                                    +
                                    ")");
}
} // namespace Sapphire::NN
