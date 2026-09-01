#include "myplacement/placement/GlobalPlacer.hpp"

#include "GlobalPlacementInternal.hpp"

#include <algorithm>
#include <cctype>
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
    if (options.optimizer == GlobalOptimizer::Legacy && options.compute_backend == ComputeBackend::Cuda) {
        throw std::invalid_argument("The CUDA backend currently supports the adaptive optimizer only.");
    }
    return options.optimizer == GlobalOptimizer::Adaptive
               ? detail::runAdaptiveGlobalPlacement(database, options)
               : detail::runLegacyGlobalPlacement(database, options);
}

}  // namespace myplacement
