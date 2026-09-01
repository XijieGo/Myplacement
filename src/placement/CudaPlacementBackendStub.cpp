#include "CudaPlacementBackend.hpp"

namespace myplacement::detail {

bool cudaPlacementBackendCompiled() {
    return false;
}

std::unique_ptr<CudaPlacementBackend> tryCreateCudaPlacementBackend(
    const PlacementDatabase&, const std::vector<ModuleId>&, const std::vector<DensityFiller>&,
    const DensityMap&, const GlobalPlacementOptions&, std::string& reason) {
    reason = "This MyPlacement build does not include CUDA. Reconfigure with -DMYPLACEMENT_ENABLE_CUDA=ON.";
    return nullptr;
}

}  // namespace myplacement::detail
