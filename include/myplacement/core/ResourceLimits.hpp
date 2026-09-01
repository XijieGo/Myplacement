#pragma once

#include <cstddef>

namespace myplacement {

// A single placement run keeps both a DensityMap and spectral work buffers.  A
// conservative bin-count cap prevents an accidental command-line grid from
// consuming the machine before the optimizer can report a useful error.
inline constexpr std::size_t kMaximumDensityBinCount = 1U << 20U;

}  // namespace myplacement
