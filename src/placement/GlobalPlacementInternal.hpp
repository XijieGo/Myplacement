#pragma once

#include "myplacement/placement/GlobalPlacer.hpp"

namespace myplacement::detail {

// These entry points intentionally remain private to the placement module.
// GlobalPlacer is the only public facade and chooses the implementation.
GlobalPlacementResult runAdaptiveGlobalPlacement(PlacementDatabase& database,
                                                 const GlobalPlacementOptions& options);
GlobalPlacementResult runLegacyGlobalPlacement(PlacementDatabase& database,
                                               const GlobalPlacementOptions& options);

}  // namespace myplacement::detail
