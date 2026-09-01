#include "GlobalPlacementInternal.hpp"
#include "GlobalPlacementSupport.hpp"
#include "CudaPlacementBackend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace myplacement::detail {
namespace {

[[nodiscard]] double dot(Vec2 left, Vec2 right) {
    return left.x * right.x + left.y * right.y;
}

struct SmoothWirelengthEvaluation {
    double smooth_wirelength = 0.0;
    double hpwl = 0.0;
    std::vector<Vec2> gradient;
};

SmoothWirelengthEvaluation evaluateSmoothWirelength(const PlacementDatabase& database, double smoothing,
                                                     bool calculate_gradient) {
    SmoothWirelengthEvaluation result;
    if (calculate_gradient) result.gradient.resize(database.modules.size());
    const double gamma = std::max(smoothing, kEpsilon);
    for (const Net& net : database.nets) {
        if (net.pins.size() < 2U) continue;
        double maximum_x = -std::numeric_limits<double>::infinity();
        double minimum_x = std::numeric_limits<double>::infinity();
        double maximum_y = -std::numeric_limits<double>::infinity();
        double minimum_y = std::numeric_limits<double>::infinity();
        for (const PinId pin_id : net.pins) {
            const Vec2 position = database.pinPosition(pin_id);
            maximum_x = std::max(maximum_x, position.x);
            minimum_x = std::min(minimum_x, position.x);
            maximum_y = std::max(maximum_y, position.y);
            minimum_y = std::min(minimum_y, position.y);
        }

        double positive_x_sum = 0.0;
        double negative_x_sum = 0.0;
        double positive_y_sum = 0.0;
        double negative_y_sum = 0.0;
        for (const PinId pin_id : net.pins) {
            const Vec2 position = database.pinPosition(pin_id);
            positive_x_sum += std::exp((position.x - maximum_x) / gamma);
            negative_x_sum += std::exp((minimum_x - position.x) / gamma);
            positive_y_sum += std::exp((position.y - maximum_y) / gamma);
            negative_y_sum += std::exp((minimum_y - position.y) / gamma);
        }
        result.smooth_wirelength += net.weight *
                                    (maximum_x + gamma * std::log(positive_x_sum) - minimum_x +
                                     gamma * std::log(negative_x_sum) + maximum_y +
                                     gamma * std::log(positive_y_sum) - minimum_y +
                                     gamma * std::log(negative_y_sum));
        result.hpwl += net.weight * ((maximum_x - minimum_x) + (maximum_y - minimum_y));

        if (!calculate_gradient) continue;
        for (const PinId pin_id : net.pins) {
            const Pin& pin = database.pins[pin_id];
            if (database.modules[pin.module].is_fixed) continue;
            const Vec2 position = database.pinPosition(pin_id);
            const double gradient_x = std::exp((position.x - maximum_x) / gamma) / positive_x_sum -
                                      std::exp((minimum_x - position.x) / gamma) / negative_x_sum;
            const double gradient_y = std::exp((position.y - maximum_y) / gamma) / positive_y_sum -
                                      std::exp((minimum_y - position.y) / gamma) / negative_y_sum;
            result.gradient[pin.module] += {net.weight * gradient_x, net.weight * gradient_y};
        }
    }
    if (!std::isfinite(result.smooth_wirelength) || !std::isfinite(result.hpwl)) {
        throw std::runtime_error("Smooth wirelength evaluation produced a non-finite value.");
    }
    return result;
}

struct DensityEvaluation {
    DensityMetrics optimizer_density;
    DensityMetrics design_density;
    double electrostatic_energy = 0.0;
};

struct ObjectiveMetrics {
    double hpwl = 0.0;
    double smooth_wirelength = 0.0;
    double density_energy = 0.0;
    double objective = 0.0;
    DensityMetrics optimizer_density;
    DensityMetrics design_density;
};

struct StateEvaluation {
    ObjectiveMetrics metrics;
    std::vector<Vec2> raw_gradient;
    std::vector<Vec2> preconditioned_gradient;
    double raw_gradient_norm = 0.0;
    double preconditioned_gradient_norm = 0.0;
    double wire_gradient_l1 = 0.0;
    double density_gradient_l1 = 0.0;
};

class GlobalPlacementSession {
public:
    GlobalPlacementSession(PlacementDatabase& database, const GlobalPlacementOptions& options)
        : database_(database), options_(options), density_map_(database.core_region, options.bins_x, options.bins_y,
                                                                options.target_density),
          movable_(database.movableModules()) {
        initializeStaticDensity(density_map_, database_);
        fillers_ = createDensityFillers(database_, density_map_, options_);
        initializePreconditioner();
        initializeComputeBackend();
        if (!cuda_backend_) {
            density_field_ = std::make_unique<ElectrostaticField>(
                options_.bins_x, options_.bins_y, database_.core_region, options_.density_field_boundary);
        }
    }

    [[nodiscard]] std::size_t particleCount() const { return movable_.size() + fillers_.size(); }
    [[nodiscard]] double minimumBinLength() const {
        return std::min(density_map_.binWidth(), density_map_.binHeight());
    }
    [[nodiscard]] double maximumBinLength() const {
        return std::max(density_map_.binWidth(), density_map_.binHeight());
    }
    [[nodiscard]] ComputeBackend computeBackend() const {
        return cuda_backend_ ? ComputeBackend::Cuda : ComputeBackend::Cpu;
    }
    [[nodiscard]] int cudaDevice() const { return cuda_backend_ ? cuda_backend_->device() : -1; }
    [[nodiscard]] std::size_t cudaReservedBytes() const {
        return cuda_backend_ ? cuda_backend_->reservedBytes() : 0U;
    }

    [[nodiscard]] std::vector<Vec2> capturePositions() const {
        std::vector<Vec2> positions(particleCount());
        for (std::size_t index = 0; index < movable_.size(); ++index) {
            positions[index] = database_.modules[movable_[index]].center;
        }
        for (std::size_t index = 0; index < fillers_.size(); ++index) {
            positions[movable_.size() + index] = fillers_[index].center;
        }
        return positions;
    }

    void applyPositions(const std::vector<Vec2>& positions) {
        if (positions.size() != particleCount()) {
            throw std::invalid_argument("Adaptive placement state does not match the particle count.");
        }
        for (std::size_t index = 0; index < movable_.size(); ++index) {
            Module& module = database_.modules[movable_[index]];
            module.center = clampCenterToRegion(positions[index], module.width, module.height, database_.core_region);
        }
        for (std::size_t index = 0; index < fillers_.size(); ++index) {
            DensityFiller& filler = fillers_[index];
            filler.center = clampCenterToRegion(positions[movable_.size() + index], filler.width, filler.height,
                                                 database_.core_region);
        }
    }

    [[nodiscard]] StateEvaluation evaluate(double smoothing, double penalty, double reference_penalty,
                                           bool calculate_gradient) {
        if (cuda_backend_) return evaluateCuda(smoothing, penalty, reference_penalty, calculate_gradient);

        const DensityEvaluation density = evaluateDensity();
        const SmoothWirelengthEvaluation wire =
            evaluateSmoothWirelength(database_, smoothing, calculate_gradient);

        StateEvaluation result;
        result.metrics.hpwl = wire.hpwl;
        result.metrics.smooth_wirelength = wire.smooth_wirelength;
        result.metrics.density_energy = density.electrostatic_energy;
        result.metrics.optimizer_density = density.optimizer_density;
        result.metrics.design_density = density.design_density;
        result.metrics.objective = wire.smooth_wirelength + penalty * density.electrostatic_energy;
        if (!std::isfinite(result.metrics.objective)) {
            throw std::runtime_error("Adaptive placement objective produced a non-finite value.");
        }
        if (!calculate_gradient) return result;

        result.raw_gradient.resize(particleCount());
        result.preconditioned_gradient.resize(particleCount());
        for (const Vec2 value : wire.gradient) {
            result.wire_gradient_l1 += std::abs(value.x) + std::abs(value.y);
        }
        for (std::size_t index = 0; index < movable_.size(); ++index) {
            const Module& module = database_.modules[movable_[index]];
            const double charge_area = densityChargeArea(module, options_.target_density);
            const Vec2 density_gradient = density_field_->sampleField(module.center) * (-charge_area);
            result.density_gradient_l1 += std::abs(density_gradient.x) + std::abs(density_gradient.y);
            result.raw_gradient[index] = wire.gradient[movable_[index]] + density_gradient * penalty;
        }
        for (std::size_t index = 0; index < fillers_.size(); ++index) {
            const std::size_t particle = movable_.size() + index;
            const Vec2 density_gradient =
                density_field_->sampleField(fillers_[index].center) *
                (-densityChargeArea(fillers_[index], options_.target_density));
            result.density_gradient_l1 += std::abs(density_gradient.x) + std::abs(density_gradient.y);
            result.raw_gradient[particle] = density_gradient * penalty;
        }

        const double density_strength =
            std::sqrt(1.0 + std::max(0.0, penalty / std::max(reference_penalty, kEpsilon)));
        double raw_gradient_squared_norm = 0.0;
        double preconditioned_gradient_squared_norm = 0.0;
        for (std::size_t index = 0; index < particleCount(); ++index) {
            result.preconditioned_gradient[index] =
                result.raw_gradient[index] * (preconditioner_base_[index] / density_strength);
            raw_gradient_squared_norm += squaredNorm(result.raw_gradient[index]);
            preconditioned_gradient_squared_norm += squaredNorm(result.preconditioned_gradient[index]);
        }
        result.raw_gradient_norm = std::sqrt(raw_gradient_squared_norm);
        result.preconditioned_gradient_norm = std::sqrt(preconditioned_gradient_squared_norm);
        return result;
    }

private:
    void initializeComputeBackend() {
        if (options_.compute_backend == ComputeBackend::Cpu) return;
        std::string reason;
        cuda_backend_ = tryCreateCudaPlacementBackend(database_, movable_, fillers_, density_map_, options_, reason);
        if (!cuda_backend_ && options_.compute_backend == ComputeBackend::Cuda) {
            throw std::invalid_argument("CUDA global-placement backend is unavailable: " + reason);
        }
    }

    [[nodiscard]] StateEvaluation evaluateCuda(double smoothing, double penalty, double reference_penalty,
                                               bool calculate_gradient) {
        const CudaPlacementEvaluation cuda_evaluation =
            cuda_backend_->evaluate(capturePositions(), smoothing, calculate_gradient);

        StateEvaluation result;
        result.metrics.hpwl = cuda_evaluation.hpwl;
        result.metrics.smooth_wirelength = cuda_evaluation.smooth_wirelength;
        result.metrics.density_energy = cuda_evaluation.electrostatic_energy;
        result.metrics.optimizer_density = cuda_evaluation.optimizer_density;
        result.metrics.design_density = cuda_evaluation.design_density;
        result.metrics.objective = result.metrics.smooth_wirelength + penalty * result.metrics.density_energy;
        if (!std::isfinite(result.metrics.objective)) {
            throw std::runtime_error("CUDA adaptive placement objective produced a non-finite value.");
        }
        if (!calculate_gradient) return result;
        if (cuda_evaluation.wire_gradient.size() != particleCount() ||
            cuda_evaluation.density_gradient.size() != particleCount()) {
            throw std::runtime_error("CUDA placement gradients do not match the particle count.");
        }

        result.raw_gradient.resize(particleCount());
        result.preconditioned_gradient.resize(particleCount());
        const double density_strength =
            std::sqrt(1.0 + std::max(0.0, penalty / std::max(reference_penalty, kEpsilon)));
        double raw_gradient_squared_norm = 0.0;
        double preconditioned_gradient_squared_norm = 0.0;
        for (std::size_t index = 0; index < particleCount(); ++index) {
            const Vec2 wire_gradient = cuda_evaluation.wire_gradient[index];
            const Vec2 density_gradient = cuda_evaluation.density_gradient[index];
            result.wire_gradient_l1 += std::abs(wire_gradient.x) + std::abs(wire_gradient.y);
            result.density_gradient_l1 += std::abs(density_gradient.x) + std::abs(density_gradient.y);
            result.raw_gradient[index] = wire_gradient + density_gradient * penalty;
            result.preconditioned_gradient[index] =
                result.raw_gradient[index] * (preconditioner_base_[index] / density_strength);
            raw_gradient_squared_norm += squaredNorm(result.raw_gradient[index]);
            preconditioned_gradient_squared_norm += squaredNorm(result.preconditioned_gradient[index]);
        }
        result.raw_gradient_norm = std::sqrt(raw_gradient_squared_norm);
        result.preconditioned_gradient_norm = std::sqrt(preconditioned_gradient_squared_norm);
        return result;
    }

    [[nodiscard]] DensityEvaluation evaluateDensity() {
        depositDynamicDensity(density_map_, database_, fillers_);
        buildDensityDeviation(density_map_, density_deviation_);
        density_field_->solve(density_deviation_);

        DensityEvaluation result;
        result.optimizer_density = density_map_.metrics(true);
        result.design_density = density_map_.metrics(false);
        const double mean = std::accumulate(density_deviation_.begin(), density_deviation_.end(), 0.0) /
                            static_cast<double>(std::max<std::size_t>(1U, density_deviation_.size()));
        const std::vector<DensityBin>& bins = density_map_.bins();
        for (std::size_t index = 0; index < bins.size(); ++index) {
            const int column = static_cast<int>(index % static_cast<std::size_t>(density_map_.columns()));
            const int row = static_cast<int>(index / static_cast<std::size_t>(density_map_.columns()));
            const double source = density_deviation_[index] - mean;
            result.electrostatic_energy +=
                0.5 * source * density_field_->potentialAt(column, row) * bins[index].region.area();
        }
        if (!std::isfinite(result.electrostatic_energy)) {
            throw std::runtime_error("Electrostatic density energy produced a non-finite value.");
        }
        result.electrostatic_energy = std::max(0.0, result.electrostatic_energy);
        return result;
    }

    void initializePreconditioner() {
        double total_movable_area = 0.0;
        for (const ModuleId id : movable_) total_movable_area += database_.modules[id].area();
        const double average_movable_area =
            total_movable_area / static_cast<double>(std::max<std::size_t>(1U, movable_.size()));
        preconditioner_base_.reserve(particleCount());
        const auto add_scale = [&](double area, std::size_t pin_count) {
            const double area_ratio = std::max(area / std::max(average_movable_area, kEpsilon), 0.1);
            const double pin_factor = std::sqrt(1.0 + static_cast<double>(pin_count));
            const double area_factor = 1.0 + std::sqrt(area_ratio);
            preconditioner_base_.push_back(1.0 / (pin_factor * area_factor));
        };
        for (const ModuleId id : movable_) {
            const Module& module = database_.modules[id];
            add_scale(module.area(), module.pins.size());
        }
        for (const DensityFiller& filler : fillers_) add_scale(filler.area(), 0U);
    }

    PlacementDatabase& database_;
    const GlobalPlacementOptions& options_;
    DensityMap density_map_;
    std::unique_ptr<ElectrostaticField> density_field_;
    std::unique_ptr<CudaPlacementBackend> cuda_backend_;
    const std::vector<ModuleId>& movable_;
    std::vector<DensityFiller> fillers_;
    std::vector<double> density_deviation_;
    std::vector<double> preconditioner_base_;
};

struct StepEstimate {
    double step_size = 0.0;
    double curvature = 0.0;
};

struct StepProposal {
    std::vector<Vec2> positions;
    double predicted_reduction = 0.0;
    double maximum_displacement = 0.0;
};

struct Checkpoint {
    std::vector<Vec2> positions;
    int iteration = 0;
    double hpwl = std::numeric_limits<double>::infinity();
    double overflow = std::numeric_limits<double>::infinity();
    bool found = false;
    bool feasible = false;
};

void validateAdaptiveOptions(const PlacementDatabase& database, const GlobalPlacementOptions& options) {
    if (database.movableModules().empty()) return;
    if (options.iterations <= 0 || options.bins_x < 2 || options.bins_y < 2 || options.target_density <= 0.0 ||
        options.target_density > 1.0 || options.maximum_bin_overflow < 0.0 ||
        options.initial_smoothing <= 0.0 || options.final_smoothing <= 0.0 ||
        options.maximum_movement_in_bins <= 0.0 || options.initial_movement_in_bins <= 0.0 ||
        options.maximum_backtracks < 0 || options.backtracking_ratio <= 0.0 || options.backtracking_ratio >= 1.0 ||
        options.armijo_coefficient <= 0.0 || options.armijo_coefficient >= 1.0 || options.maximum_momentum < 0.0 ||
        options.maximum_momentum >= 1.0 || options.density_penalty_increase <= 1.0 ||
        options.density_penalty_decrease <= 0.0 || options.density_penalty_decrease >= 1.0 ||
        options.minimum_density_penalty <= 0.0 || options.maximum_density_penalty < options.minimum_density_penalty ||
        options.smoothing_overflow_exponent <= 0.0 || options.feasible_refinement_iterations <= 0) {
        throw std::invalid_argument("Adaptive global placement options are outside their valid range.");
    }
    const std::size_t columns = static_cast<std::size_t>(options.bins_x);
    const std::size_t rows = static_cast<std::size_t>(options.bins_y);
    if (columns > ElectrostaticField::kMaximumBinCount / rows) {
        throw std::invalid_argument("Density grid exceeds the safe one-million-bin workspace limit.");
    }
}

[[nodiscard]] double smoothingFromOverflow(const GlobalPlacementOptions& options, double maximum_bin_length,
                                            double overflow, double initial_overflow) {
    const double ratio = clamp(overflow / std::max(initial_overflow, kEpsilon), 0.0, 1.0);
    const double annealing = std::pow(ratio, options.smoothing_overflow_exponent);
    return (options.final_smoothing + (options.initial_smoothing - options.final_smoothing) * annealing) *
           maximum_bin_length;
}

[[nodiscard]] double initialStepSize(const StateEvaluation& evaluation, double desired_movement) {
    const double count = static_cast<double>(std::max<std::size_t>(1U, evaluation.preconditioned_gradient.size()));
    const double root_mean_square_gradient = evaluation.preconditioned_gradient_norm / std::sqrt(count);
    return desired_movement / std::max(root_mean_square_gradient, kEpsilon);
}

[[nodiscard]] StepEstimate estimateStep(const std::vector<Vec2>& positions, const StateEvaluation& evaluation,
                                         const std::vector<Vec2>& previous_positions,
                                         const std::vector<Vec2>& previous_gradient, bool has_previous_gradient,
                                         double fallback_step) {
    StepEstimate result;
    result.step_size = fallback_step;
    if (!has_previous_gradient || positions.size() != previous_positions.size() ||
        evaluation.preconditioned_gradient.size() != previous_gradient.size()) {
        return result;
    }
    double position_delta_squared_norm = 0.0;
    double gradient_delta_squared_norm = 0.0;
    for (std::size_t index = 0; index < positions.size(); ++index) {
        position_delta_squared_norm += squaredNorm(positions[index] - previous_positions[index]);
        gradient_delta_squared_norm +=
            squaredNorm(evaluation.preconditioned_gradient[index] - previous_gradient[index]);
    }
    if (position_delta_squared_norm <= kEpsilon || gradient_delta_squared_norm <= kEpsilon) return result;
    result.curvature = std::sqrt(gradient_delta_squared_norm / position_delta_squared_norm);
    const double curvature_step = 1.0 / std::max(result.curvature, kEpsilon);
    result.step_size = clamp(curvature_step, fallback_step * 0.25, fallback_step * 4.0);
    return result;
}

[[nodiscard]] StepProposal proposeStep(GlobalPlacementSession& session, const std::vector<Vec2>& base_positions,
                                       const StateEvaluation& base_evaluation, double step_size,
                                       double maximum_displacement) {
    if (base_positions.size() != base_evaluation.preconditioned_gradient.size()) {
        throw std::invalid_argument("Adaptive gradient does not match the current placement state.");
    }
    std::vector<Vec2> proposed_positions = base_positions;
    for (std::size_t index = 0; index < proposed_positions.size(); ++index) {
        Vec2 displacement = base_evaluation.preconditioned_gradient[index] * (-step_size);
        const double displacement_norm = norm(displacement);
        if (displacement_norm > maximum_displacement) {
            displacement = displacement * (maximum_displacement / displacement_norm);
        }
        proposed_positions[index] += displacement;
    }
    session.applyPositions(proposed_positions);

    StepProposal result;
    result.positions = session.capturePositions();
    for (std::size_t index = 0; index < result.positions.size(); ++index) {
        const Vec2 actual_descent = base_positions[index] - result.positions[index];
        result.predicted_reduction += dot(base_evaluation.raw_gradient[index], actual_descent);
        result.maximum_displacement = std::max(result.maximum_displacement, norm(actual_descent));
    }
    return result;
}

[[nodiscard]] bool acceptsCandidate(const ObjectiveMetrics& base, const ObjectiveMetrics& candidate,
                                    double predicted_reduction, const GlobalPlacementOptions& options) {
    const double objective_tolerance = std::max(1e-8, std::abs(base.objective) * 1e-10);
    const double armijo_target = base.objective - options.armijo_coefficient * std::max(0.0, predicted_reduction);
    const bool armijo = candidate.objective <= armijo_target + objective_tolerance;
    const bool stable_objective = candidate.objective <= base.objective + objective_tolerance &&
                                  candidate.optimizer_density.normalized_overflow <=
                                      base.optimizer_density.normalized_overflow * 1.005;
    const double meaningful_density_gain = std::max(1e-4, options.density_stall_threshold * 0.25);
    const bool density_trade = candidate.optimizer_density.normalized_overflow <=
                                   base.optimizer_density.normalized_overflow - meaningful_density_gain &&
                               candidate.objective <= base.objective * (1.0 + options.objective_increase_for_density);
    return armijo || stable_objective || density_trade;
}

[[nodiscard]] double updateDensityPenalty(double penalty, double reference_penalty, const ObjectiveMetrics& before,
                                          const ObjectiveMetrics& after, const GlobalPlacementOptions& options) {
    const double hpwl_growth = (after.hpwl - before.hpwl) / std::max(before.hpwl, 1.0);
    const double density_improvement =
        before.optimizer_density.normalized_overflow - after.optimizer_density.normalized_overflow;
    const bool density_is_critical =
        after.optimizer_density.normalized_overflow > std::max(2.0 * options.maximum_bin_overflow, 0.02);
    if (density_is_critical && density_improvement < options.density_stall_threshold) {
        penalty *= options.density_penalty_increase;
    } else if (!density_is_critical && hpwl_growth > options.hpwl_growth_restart_threshold &&
               density_improvement < options.density_stall_threshold) {
        penalty *= options.density_penalty_decrease;
    } else if (after.optimizer_density.normalized_overflow > options.maximum_bin_overflow &&
               density_improvement < options.density_stall_threshold) {
        penalty *= options.density_penalty_increase;
    } else if (after.design_density.normalized_overflow <= options.maximum_bin_overflow) {
        penalty *= options.density_penalty_decrease;
    }
    const double minimum_penalty = std::max(options.minimum_density_penalty, reference_penalty * 0.02);
    const double maximum_penalty = std::max(options.maximum_density_penalty, minimum_penalty * 10.0);
    return clamp(penalty, minimum_penalty, maximum_penalty);
}

void considerCheckpoint(Checkpoint& best_feasible, Checkpoint& best_fallback, const std::vector<Vec2>& positions,
                        const ObjectiveMetrics& metrics, int iteration, const GlobalPlacementOptions& options,
                        bool& is_new_best) {
    is_new_best = false;
    const double overflow = metrics.design_density.normalized_overflow;
    const bool feasible = overflow <= options.maximum_bin_overflow;
    if (feasible && (!best_feasible.found || metrics.hpwl < best_feasible.hpwl)) {
        best_feasible.positions = positions;
        best_feasible.iteration = iteration;
        best_feasible.hpwl = metrics.hpwl;
        best_feasible.overflow = overflow;
        best_feasible.found = true;
        best_feasible.feasible = true;
        is_new_best = true;
    }
    if (!best_feasible.found &&
        (!best_fallback.found || overflow < best_fallback.overflow - 1e-12 ||
         (std::abs(overflow - best_fallback.overflow) <= 1e-12 && metrics.hpwl < best_fallback.hpwl))) {
        best_fallback.positions = positions;
        best_fallback.iteration = iteration;
        best_fallback.hpwl = metrics.hpwl;
        best_fallback.overflow = overflow;
        best_fallback.found = true;
        best_fallback.feasible = false;
        is_new_best = true;
    }
}

[[nodiscard]] double momentumAlignment(const std::vector<Vec2>& velocity,
                                        const std::vector<Vec2>& preconditioned_gradient) {
    double alignment = 0.0;
    for (std::size_t index = 0; index < velocity.size(); ++index) {
        alignment -= dot(velocity[index], preconditioned_gradient[index]);
    }
    return alignment;
}

GlobalPlacementIteration makeHistoryRow(int iteration, const ObjectiveMetrics& metrics, double penalty,
                                        double smoothing, double step_size, double maximum_displacement,
                                        double gradient_norm, double curvature, int backtracks, bool restarted,
                                        bool accepted, bool best_checkpoint) {
    GlobalPlacementIteration row;
    row.iteration = iteration;
    row.hpwl = metrics.hpwl;
    row.overflow = metrics.optimizer_density.normalized_overflow;
    row.design_overflow = metrics.design_density.normalized_overflow;
    row.smooth_wirelength = metrics.smooth_wirelength;
    row.density_energy = metrics.density_energy;
    row.objective = metrics.objective;
    row.penalty = penalty;
    row.smoothing = smoothing;
    row.step_size = step_size;
    row.maximum_displacement = maximum_displacement;
    row.gradient_norm = gradient_norm;
    row.curvature = curvature;
    row.backtracks = backtracks;
    row.momentum_restarted = restarted;
    row.accepted = accepted;
    row.best_checkpoint = best_checkpoint;
    return row;
}

}  // namespace

GlobalPlacementResult runAdaptiveGlobalPlacement(PlacementDatabase& database, const GlobalPlacementOptions& options) {
    if (database.movableModules().empty()) return {};
    validateAdaptiveOptions(database, options);

    const auto started = std::chrono::steady_clock::now();
    GlobalPlacementResult result;
    result.hpwl_before = calculateHpwl(database).hpwl;
    result.overflow_before = calculateDensity(database, options.bins_x, options.bins_y, options.target_density)
                                 .normalized_overflow;

    GlobalPlacementSession session(database, options);
    result.compute_backend_used = session.computeBackend();
    result.cuda_device_used = session.cudaDevice();
    result.cuda_reserved_memory_bytes = session.cudaReservedBytes();
    std::vector<Vec2> current_positions = session.capturePositions();
    std::vector<Vec2> velocity(current_positions.size());
    std::vector<Vec2> previous_gradient;
    std::vector<Vec2> previous_gradient_positions;
    bool has_previous_gradient = false;

    double smoothing = options.initial_smoothing * session.maximumBinLength();
    StateEvaluation initial_evaluation = session.evaluate(smoothing, 1.0, 1.0, true);
    const double initial_overflow = std::max(initial_evaluation.metrics.optimizer_density.normalized_overflow, kEpsilon);
    smoothing = smoothingFromOverflow(options, session.maximumBinLength(),
                                      initial_evaluation.metrics.optimizer_density.normalized_overflow, initial_overflow);
    const double raw_penalty = initial_evaluation.density_gradient_l1 > kEpsilon
                                   ? initial_evaluation.wire_gradient_l1 / initial_evaluation.density_gradient_l1
                                   : 1.0;
    double penalty = clamp(raw_penalty, options.minimum_density_penalty, options.maximum_density_penalty);
    const double reference_penalty = penalty;
    initial_evaluation = session.evaluate(smoothing, penalty, reference_penalty, true);
    double last_optimizer_overflow = initial_evaluation.metrics.optimizer_density.normalized_overflow;
    double last_step_size = initialStepSize(initial_evaluation,
                                            options.initial_movement_in_bins * session.minimumBinLength());

    Checkpoint best_feasible;
    Checkpoint best_fallback;
    bool initial_is_best = false;
    considerCheckpoint(best_feasible, best_fallback, current_positions, initial_evaluation.metrics, 0, options,
                       initial_is_best);

    int momentum_age = 0;
    int objective_worse_streak = 0;
    int feasible_refinement_iterations =
        initial_evaluation.metrics.design_density.normalized_overflow <= options.maximum_bin_overflow ? 1 : 0;

    for (int iteration = 0; iteration < options.iterations; ++iteration) {
        StateEvaluation current_evaluation;
        if (iteration == 0) {
            current_evaluation = std::move(initial_evaluation);
        } else {
            smoothing = smoothingFromOverflow(options, session.maximumBinLength(), last_optimizer_overflow,
                                              initial_overflow);
            session.applyPositions(current_positions);
            current_evaluation = session.evaluate(smoothing, penalty, reference_penalty, true);
        }

        bool restarted = objective_worse_streak >= 2;
        bool momentum_used = false;
        std::vector<Vec2> base_positions = current_positions;
        const StateEvaluation* base_evaluation = &current_evaluation;
        StateEvaluation lookahead_evaluation;
        if (!restarted && momentum_age > 0) {
            const double momentum = std::min(options.maximum_momentum,
                                             static_cast<double>(momentum_age) /
                                                 static_cast<double>(momentum_age + 3));
            if (momentumAlignment(velocity, current_evaluation.preconditioned_gradient) <= 0.0) {
                restarted = true;
            } else {
                for (std::size_t index = 0; index < base_positions.size(); ++index) {
                    base_positions[index] += velocity[index] * momentum;
                }
                session.applyPositions(base_positions);
                base_positions = session.capturePositions();
                lookahead_evaluation = session.evaluate(smoothing, penalty, reference_penalty, true);
                if (lookahead_evaluation.metrics.objective >
                    current_evaluation.metrics.objective * (1.0 + options.hpwl_growth_restart_threshold)) {
                    restarted = true;
                } else {
                    base_evaluation = &lookahead_evaluation;
                    momentum_used = true;
                }
            }
        }
        if (restarted) {
            session.applyPositions(current_positions);
            base_positions = current_positions;
            base_evaluation = &current_evaluation;
            momentum_age = 0;
            objective_worse_streak = 0;
            ++result.momentum_restarts;
        }

        bool accepted = false;
        bool new_best_checkpoint = false;
        int backtracks = 0;
        double used_step_size = 0.0;
        double used_curvature = 0.0;
        double used_maximum_displacement = 0.0;
        StateEvaluation accepted_evaluation;
        std::vector<Vec2> accepted_positions;

        for (int restart_pass = 0; restart_pass < 2 && !accepted; ++restart_pass) {
            const StepEstimate estimate =
                estimateStep(base_positions, *base_evaluation, previous_gradient_positions, previous_gradient,
                             has_previous_gradient, std::max(last_step_size, kEpsilon));
            double candidate_step_size = estimate.step_size;
            const double overflow_ratio = clamp(base_evaluation->metrics.optimizer_density.normalized_overflow /
                                                    std::max(initial_overflow, kEpsilon),
                                                0.0, 1.0);
            const double maximum_displacement = options.maximum_movement_in_bins * session.minimumBinLength() *
                                                (0.35 + 0.65 * overflow_ratio);
            for (int attempt = 0; attempt <= options.maximum_backtracks; ++attempt) {
                StepProposal proposal = proposeStep(session, base_positions, *base_evaluation, candidate_step_size,
                                                    maximum_displacement);
                if (proposal.predicted_reduction <= kEpsilon) {
                    session.applyPositions(base_positions);
                    break;
                }
                StateEvaluation candidate_evaluation =
                    session.evaluate(smoothing, penalty, reference_penalty, false);
                if (acceptsCandidate(base_evaluation->metrics, candidate_evaluation.metrics,
                                     proposal.predicted_reduction, options)) {
                    accepted = true;
                    used_step_size = candidate_step_size;
                    used_curvature = estimate.curvature;
                    used_maximum_displacement = proposal.maximum_displacement;
                    accepted_positions = std::move(proposal.positions);
                    accepted_evaluation = std::move(candidate_evaluation);
                    break;
                }
                ++backtracks;
                ++result.rejected_candidates;
                candidate_step_size *= options.backtracking_ratio;
                session.applyPositions(base_positions);
            }
            if (accepted || restarted || !momentum_used) break;

            restarted = true;
            ++result.momentum_restarts;
            momentum_age = 0;
            objective_worse_streak = 0;
            session.applyPositions(current_positions);
            base_positions = current_positions;
            base_evaluation = &current_evaluation;
            momentum_used = false;
        }

        const double penalty_used = penalty;
        if (accepted) {
            const std::vector<Vec2> old_current_positions = current_positions;
            current_positions = std::move(accepted_positions);
            velocity.resize(current_positions.size());
            for (std::size_t index = 0; index < current_positions.size(); ++index) {
                velocity[index] = current_positions[index] - old_current_positions[index];
            }
            last_step_size = used_step_size;
            if (restarted) {
                momentum_age = 0;
            } else {
                ++momentum_age;
            }

            const double hpwl_growth = (accepted_evaluation.metrics.hpwl - current_evaluation.metrics.hpwl) /
                                       std::max(current_evaluation.metrics.hpwl, 1.0);
            objective_worse_streak = hpwl_growth > options.hpwl_growth_restart_threshold
                                         ? objective_worse_streak + 1
                                         : 0;
            const double next_penalty = updateDensityPenalty(penalty, reference_penalty,
                                                              current_evaluation.metrics,
                                                              accepted_evaluation.metrics, options);
            const bool penalty_changed = std::abs(std::log(next_penalty / std::max(penalty, kEpsilon))) > 0.01;
            penalty = next_penalty;
            if (penalty_changed) {
                previous_gradient.clear();
                previous_gradient_positions.clear();
                has_previous_gradient = false;
            } else {
                previous_gradient_positions = base_positions;
                previous_gradient = base_evaluation->preconditioned_gradient;
                has_previous_gradient = true;
            }
            last_optimizer_overflow = accepted_evaluation.metrics.optimizer_density.normalized_overflow;
            considerCheckpoint(best_feasible, best_fallback, current_positions, accepted_evaluation.metrics,
                               iteration + 1, options, new_best_checkpoint);
            ++result.accepted_iterations;
            if (accepted_evaluation.metrics.design_density.normalized_overflow <= options.maximum_bin_overflow) {
                ++feasible_refinement_iterations;
            } else {
                feasible_refinement_iterations = 0;
            }
            result.history.push_back(makeHistoryRow(iteration + 1, accepted_evaluation.metrics, penalty_used,
                                                    smoothing, used_step_size, used_maximum_displacement,
                                                    base_evaluation->preconditioned_gradient_norm, used_curvature,
                                                    backtracks, restarted, true, new_best_checkpoint));
        } else {
            session.applyPositions(current_positions);
            std::fill(velocity.begin(), velocity.end(), Vec2{});
            momentum_age = 0;
            objective_worse_streak = 0;
            if (current_evaluation.metrics.optimizer_density.normalized_overflow > options.maximum_bin_overflow) {
                penalty = clamp(penalty * options.density_penalty_increase,
                                std::max(options.minimum_density_penalty, reference_penalty * 0.02),
                                options.maximum_density_penalty);
            }
            last_optimizer_overflow = current_evaluation.metrics.optimizer_density.normalized_overflow;
            if (current_evaluation.metrics.design_density.normalized_overflow <= options.maximum_bin_overflow) {
                ++feasible_refinement_iterations;
            } else {
                feasible_refinement_iterations = 0;
            }
            result.history.push_back(makeHistoryRow(iteration + 1, current_evaluation.metrics, penalty_used,
                                                    smoothing, 0.0, 0.0,
                                                    current_evaluation.preconditioned_gradient_norm, 0.0, backtracks,
                                                    restarted, false, false));
        }
        result.completed_iterations = iteration + 1;
        if (iteration >= 20 && feasible_refinement_iterations >= options.feasible_refinement_iterations) break;
    }

    const Checkpoint& chosen = best_feasible.found ? best_feasible : best_fallback;
    if (chosen.found) {
        session.applyPositions(chosen.positions);
        result.restored_best_checkpoint = true;
        result.best_checkpoint_iteration = chosen.iteration;
        result.best_checkpoint_hpwl = chosen.hpwl;
        result.best_checkpoint_overflow = chosen.overflow;
    }
    result.hpwl_after = calculateHpwl(database).hpwl;
    result.overflow_after = calculateDensity(database, options.bins_x, options.bins_y, options.target_density)
                                .normalized_overflow;
    result.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return result;
}

}  // namespace myplacement::detail
