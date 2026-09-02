#include "myplacement/placement/GlobalPlacer.hpp"

#include "myplacement/placement/CudaDevicePolicy.hpp"

#include "GlobalPlacementInternal.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace myplacement {
namespace {

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

bool isKnownGlobalOptimizer(GlobalOptimizer optimizer) {
    switch (optimizer) {
        case GlobalOptimizer::Legacy:
        case GlobalOptimizer::Adaptive: return true;
    }
    return false;
}

bool isKnownComputeBackend(ComputeBackend backend) {
    switch (backend) {
        case ComputeBackend::Cpu:
        case ComputeBackend::Cuda:
        case ComputeBackend::Auto: return true;
    }
    return false;
}

bool isKnownDensityFieldBoundary(DensityFieldBoundary boundary) {
    switch (boundary) {
        case DensityFieldBoundary::Periodic:
        case DensityFieldBoundary::Neumann: return true;
    }
    return false;
}

bool isKnownRudyPenaltyModel(RudyPenaltyModel model) {
    switch (model) {
        case RudyPenaltyModel::Disabled:
        case RudyPenaltyModel::HingeL2:
        case RudyPenaltyModel::SoftplusL2:
        case RudyPenaltyModel::HingeL4: return true;
    }
    return false;
}

bool allFinite(std::initializer_list<double> values) {
    return std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); });
}

void validatePublicGlobalOptions(const GlobalPlacementOptions& options) {
    if (!isKnownGlobalOptimizer(options.optimizer) || !isKnownComputeBackend(options.compute_backend) ||
        !isKnownDensityFieldBoundary(options.density_field_boundary) ||
        !isKnownRudyPenaltyModel(options.rudy_options.penalty_model) ||
        !allFinite({options.target_density,
                    options.maximum_bin_overflow,
                    options.initial_smoothing,
                    options.final_smoothing,
                    options.maximum_movement_in_bins,
                    options.penalty_growth,
                    options.backtracking_ratio,
                    options.armijo_coefficient,
                    options.maximum_momentum,
                    options.initial_movement_in_bins,
                    options.objective_increase_for_density,
                    options.hpwl_growth_restart_threshold,
                    options.density_stall_threshold,
                    options.density_penalty_increase,
                    options.density_penalty_decrease,
                    options.minimum_density_penalty,
                    options.maximum_density_penalty,
                    options.smoothing_overflow_exponent,
                    options.rudy_options.minimum_span_in_bins,
                    options.rudy_options.capacity_factor,
                    options.rudy_options.softplus_temperature,
                    options.rudy_validation_capacity_factor,
                    options.routability_start_overflow,
                    options.routability_weight_scale})) {
        throw std::invalid_argument("Global placement options contain an unknown mode or non-finite value.");
    }
    if (options.compute_backend != ComputeBackend::Cpu &&
        (!isPermittedCudaDevice(options.cuda_device) || options.maximum_cuda_memory_bytes == 0U)) {
        throw std::invalid_argument(
            "CUDA global placement requires device 1 through 4 or 7 and a non-zero memory budget.");
    }
}

}  // namespace

std::string toString(GlobalOptimizer optimizer) {
    return optimizer == GlobalOptimizer::Adaptive ? "adaptive" : "legacy";
}

GlobalOptimizer parseGlobalOptimizer(const std::string& value) {
    const std::string normalized = lowercase(value);
    if (normalized == "adaptive" || normalized == "closed-loop" || normalized == "closed_loop") {
        return GlobalOptimizer::Adaptive;
    }
    if (normalized == "legacy" || normalized == "open-loop" || normalized == "open_loop") {
        return GlobalOptimizer::Legacy;
    }
    throw std::invalid_argument("Unknown global optimizer: " + value + ". Use adaptive or legacy.");
}

std::string toString(ComputeBackend backend) {
    switch (backend) {
        case ComputeBackend::Cpu: return "cpu";
        case ComputeBackend::Cuda: return "cuda";
        case ComputeBackend::Auto: return "auto";
    }
    return "cpu";
}

ComputeBackend parseComputeBackend(const std::string& value) {
    const std::string normalized = lowercase(value);
    if (normalized == "cpu") return ComputeBackend::Cpu;
    if (normalized == "cuda" || normalized == "gpu") return ComputeBackend::Cuda;
    if (normalized == "auto") return ComputeBackend::Auto;
    throw std::invalid_argument("Unknown compute backend: " + value + ". Use cpu, cuda, or auto.");
}

GlobalPlacementResult GlobalPlacer::run(PlacementDatabase& database, const GlobalPlacementOptions& options) const {
    validatePublicGlobalOptions(options);
    if (options.optimizer == GlobalOptimizer::Legacy && options.compute_backend == ComputeBackend::Cuda) {
        throw std::invalid_argument("The CUDA backend currently supports the adaptive optimizer only.");
    }
    if (options.optimizer == GlobalOptimizer::Legacy &&
        options.rudy_options.penalty_model != RudyPenaltyModel::Disabled) {
        throw std::invalid_argument("RUDY routability optimization currently supports the adaptive optimizer only.");
    }
    return options.optimizer == GlobalOptimizer::Adaptive
               ? detail::runAdaptiveGlobalPlacement(database, options)
               : detail::runLegacyGlobalPlacement(database, options);
}

}  // namespace myplacement
