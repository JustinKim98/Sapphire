// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Sapphire_BACKPROP_MATHBACKWARD_DECL_HPP
#define Sapphire_BACKPROP_MATHBACKWARD_DECL_HPP

#include <Sapphire/operations/Backward/BackPropWrapper.hpp>

namespace Sapphire::BackProp
{
class MulBackProp : public BackPropWrapper
{
public:
    explicit MulBackProp(std::string name, const TensorUtil::TensorData& a,
                         TensorUtil::TensorData da,
                         const TensorUtil::TensorData& b,
                         TensorUtil::TensorData db,
                         TensorUtil::TensorData dy);


private:
    void m_runBackProp() override;
};

class AddBackProp : public BackPropWrapper
{
public:
    explicit AddBackProp(std::string name, TensorUtil::TensorData da,
                         TensorUtil::TensorData db,
                         TensorUtil::TensorData dy);

private:
    void m_runBackProp() override;
};

class SubBackProp : public BackPropWrapper
{
public:
    explicit SubBackProp(std::string name, TensorUtil::TensorData da,
                         TensorUtil::TensorData db, TensorUtil::TensorData dy);

private:
    void m_runBackProp() override;
};

class DotBackProp : public BackPropWrapper
{
public:
    explicit DotBackProp(std::string name, const TensorUtil::TensorData& a,
                         const TensorUtil::TensorData& b, TensorUtil::TensorData da,
                         TensorUtil::TensorData db, TensorUtil::TensorData dy);

private:
    void m_runBackProp() override;
};

class MeanBackProp : public BackPropWrapper
{
public:
    explicit MeanBackProp(std::string name, TensorUtil::TensorData dx,
                          TensorUtil::TensorData x,
                          TensorUtil::TensorData dy, int dim);

private:
    void m_runBackProp() override;
    int m_dim;
};
} // namespace Sapphire::BackProp

#endif
