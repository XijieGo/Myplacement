#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "GlobalPlacementSupport.hpp"

namespace myplacement::detail {

// Numerical values returned by the CUDA path. The adaptive controller remains
// deliberately CPU-owned: it is responsible for backtracking, checkpointing,
// and updating the PlacementDatabase. The high-volume math is device-side.
struct CudaPlacementEvaluation {
    double hpwl = 0.0;
    double smooth_wirelength = 0.0;
    double electrostatic_energy = 0.0;
    DensityMetrics optimizer_density;
    DensityMetrics design_density;
    std::vector<Vec2> wire_gradient;
    std::vector<Vec2> density_gradient;
};

class CudaPlacementBackend {
public:
    virtual ~CudaPlacementBackend() = default;

    CudaPlacementBackend(const CudaPlacementBackend&) = delete;
    CudaPlacementBackend& operator=(const CudaPlacementBackend&) = delete;

    virtual CudaPlacementEvaluation evaluate(const std::vector<Vec2>& particle_positions,
                                             double smoothing, bool calculate_gradient) = 0;
    [[nodiscard]] virtual int device() const = 0;
    [[nodiscard]] virtual std::size_t reservedBytes() const = 0;

protected:
    CudaPlacementBackend() = default;
};

// A nullptr result means that CUDA is unavailable for a recoverable reason;
// the reason is always populated. Explicit --compute-backend cuda turns that
// condition into a user-visible error, while auto can safely choose CPU.
[[nodiscard]] bool cudaPlacementBackendCompiled();
[[nodiscard]] std::unique_ptr<CudaPlacementBackend> tryCreateCudaPlacementBackend(
    const PlacementDatabase& database, const std::vector<ModuleId>& movable,
    const std::vector<DensityFiller>& fillers, const DensityMap& static_density,
    const GlobalPlacementOptions& options, std::string& reason);

}  // namespace myplacement::detail
