// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Sapphire/util/CudaDevice.hpp>
#include <stdexcept>

namespace Sapphire
{
CudaDevice::CudaDevice(int id, std::string name)
    : m_id(id),
      m_name(std::move(name))
{
    if (id >= GetAvailableCudaDeviceCount())
    {
        throw std::runtime_error("Cuda device has not been detected");
    }

    int majorCapability;
    int minorCapability;
    cudaDeviceGetAttribute(&majorCapability, cudaDevAttrComputeCapabilityMajor,
                           m_id);
    cudaDeviceGetAttribute(&minorCapability, cudaDevAttrComputeCapabilityMinor,
                           m_id);
    m_cudaCapability = majorCapability * 10 + minorCapability;
}

bool CudaDevice::operator==(const CudaDevice& device) const
{
    return m_id == device.m_id && m_name == device.m_name;
}

bool CudaDevice::operator!=(const CudaDevice& device) const
{
    return !(*this == device);
}
} // namespace Sapphire
