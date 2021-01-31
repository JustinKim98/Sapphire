// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Motutapu/util/Device.hpp>
#include <stdexcept>

namespace Motutapu
{
Device::Device(std::string name)
    : m_id(-1),
      m_type(DeviceType::CPU),
      m_name(std::move(name)),
      m_padByteSize(32)
{
    //! Todo : change padByteSize according to hardware support
}

Device::Device(int id, std::string name)
    : m_id(id),
      m_type(DeviceType::CUDA),
      m_name(std::move(name)),
      m_padByteSize(32)
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

bool Device::operator==(const Device& device) const
{
    return m_id == device.m_id && m_type == device.m_type &&
           m_name == device.m_name && m_padByteSize == device.m_padByteSize;
}

bool Device::operator!=(const Device& device) const
{
    return !(*this == device);
}
}  // namespace Motutapu
