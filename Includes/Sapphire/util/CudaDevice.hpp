// Copyright (c) 2021, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Sapphire_DEVICE_HPP
#define Sapphire_DEVICE_HPP

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace Sapphire
{
enum class DeviceType
{
    Host,
    Cuda,
};

class CudaDevice
{
public:
    CudaDevice() = default;

    CudaDevice(int id, std::string name);
    ~CudaDevice() = default;

    CudaDevice(const CudaDevice& device) = default;
    CudaDevice(CudaDevice&& device) noexcept = default;
    CudaDevice& operator=(const CudaDevice& device) = default;
    CudaDevice& operator=(CudaDevice&& device) noexcept = default;

    bool operator==(const CudaDevice& device) const;
    bool operator!=(const CudaDevice& device) const;

    [[nodiscard]] DeviceType Type() const
    {
        if (m_id == -1)
            return DeviceType::Host;
        return DeviceType::Cuda;
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
    std::string m_name = "Undefined";
    int m_cudaCapability = 0;
};
} // namespace Sapphire

#endif
