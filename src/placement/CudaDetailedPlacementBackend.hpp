#pragma once

#include <string>

#include "myplacement/placement/DetailedPlacer.hpp"

namespace myplacement::detail {

[[nodiscard]] bool cudaDetailedPlacementBackendCompiled();

// Returns true only when the CUDA pass completed and populated result.  A
// recoverable availability failure is returned as false with a reason, so
// --detailed-backend auto can safely fall back to the CPU reference path.
[[nodiscard]] bool tryRunCudaDetailedPlacement(PlacementDatabase& database,
                                                const DetailedPlacementOptions& options,
                                                DetailedPlacementResult& result,
                                                std::string& reason);

}  // namespace myplacement::detail
