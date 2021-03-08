// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef MOTUTAPU_DEVICE_HPP
#define MOTUTAPU_DEVICE_HPP

#include <cuda_runtime.h>
#include <Motutapu/tensor/Shape.hpp>
#include <stdexcept>
#include <string>

namespace Motutapu
{
enum class DeviceType
{
    HOST,
    CUDA,
};

class Device
{
 public:
    Device() = default;

    explicit Device(std::string name);
    Device(int id, std::string name);
    ~Device() = default;

    Device(const Device& device) = default;
    Device(Device&& device) noexcept = default;
    Device& operator=(const Device& device) = default;
    Device& operator=(Device&& device) noexcept = default;

    bool operator==(const Device& device) const;
    bool operator!=(const Device& device) const;

    [[nodiscard]] DeviceType Type() const
    {
        return m_type;
    }

    [[nodiscard]] std::string Name() const
    {
        return m_name;
    }

    [[nodiscard]] int GetID() const
    {
        return m_id;
    }

    [[nodiscard]] int GetCudaCapability() const
    {
        if (m_type != DeviceType::CUDA)
        {
            throw std::runtime_error(
                "GetCudaCapability - Device is not set as CUDA");
        }
        return m_cudaCapability;
    }

    static int GetAvailableCudaDeviceCount()
    {
        int count;
        cudaGetDeviceCount(&count);
        return count;
    }

 private:
    int m_id = -1;
    DeviceType m_type = DeviceType::HOST;
    std::string m_name = "Undefined";
    std::size_t m_padByteSize = 0;
    int m_cudaCapability = 0;
};
}  // namespace Motutapu

#endif
