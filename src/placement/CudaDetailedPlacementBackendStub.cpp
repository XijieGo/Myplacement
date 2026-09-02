#include "CudaDetailedPlacementBackend.hpp"

namespace myplacement::detail {

bool cudaDetailedPlacementBackendCompiled() {
    return false;
}

bool tryRunCudaDetailedPlacement(PlacementDatabase&, const DetailedPlacementOptions&,
                                 DetailedPlacementResult&, std::string& reason) {
    reason = "This MyPlacement build does not include CUDA. Reconfigure with -DMYPLACEMENT_ENABLE_CUDA=ON.";
    return false;
}

}  // namespace myplacement::detail
