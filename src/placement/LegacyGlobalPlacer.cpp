#include "GlobalPlacementInternal.hpp"
#include "GlobalPlacementSupport.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace myplacement::detail {
namespace {

std::vector<Vec2> smoothWirelengthGradient(const PlacementDatabase& database, double smoothing) {
    std::vector<Vec2> gradient(database.modules.size());
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
        for (const PinId pin_id : net.pins) {
            const Pin& pin = database.pins[pin_id];
            if (database.modules[pin.module].is_fixed) continue;
            const Vec2 position = database.pinPosition(pin_id);
            const double gradient_x = std::exp((position.x - maximum_x) / gamma) / positive_x_sum -
                                      std::exp((minimum_x - position.x) / gamma) / negative_x_sum;
            const double gradient_y = std::exp((position.y - maximum_y) / gamma) / positive_y_sum -
                                      std::exp((minimum_y - position.y) / gamma) / negative_y_sum;
            gradient[pin.module] += {net.weight * gradient_x, net.weight * gradient_y};
        }
    }
    return gradient;
}

double l1Norm(const std::vector<Vec2>& values) {
    double result = 0.0;
    for (const Vec2 value : values) result += std::abs(value.x) + std::abs(value.y);
    return result;
}

}  // namespace

GlobalPlacementResult runLegacyGlobalPlacement(PlacementDatabase& database, const GlobalPlacementOptions& options) {
    if (database.movableModules().empty()) return {};
    if (options.iterations <= 0 || options.bins_x < 2 || options.bins_y < 2 || options.target_density <= 0.0 ||
        options.target_density > 1.0) {
        throw std::invalid_argument(
            "Global placement requires positive iterations, a grid of at least 2 x 2, and target density in (0, 1].");
    }
    const std::size_t columns = static_cast<std::size_t>(options.bins_x);
    const std::size_t rows = static_cast<std::size_t>(options.bins_y);
    if (columns > ElectrostaticField::kMaximumBinCount / rows) {
        throw std::invalid_argument("Density grid exceeds the safe one-million-bin workspace limit.");
    }

    const auto started = std::chrono::steady_clock::now();
    GlobalPlacementResult result;
    result.hpwl_before = calculateHpwl(database).hpwl;
    result.overflow_before = calculateDensity(database, options.bins_x, options.bins_y, options.target_density)
                                 .normalized_overflow;

    DensityMap density_map(database.core_region, options.bins_x, options.bins_y, options.target_density);
    initializeStaticDensity(density_map, database);
    std::vector<DensityFiller> fillers = createDensityFillers(database, density_map, options);
    ElectrostaticField density_field(options.bins_x, options.bins_y, database.core_region,
                                    options.density_field_boundary);
    std::vector<double> density_deviation;

    const std::vector<ModuleId>& movable = database.movableModules();
    const std::size_t particle_count = movable.size() + fillers.size();
    std::vector<Vec2> current(particle_count);
    std::vector<Vec2> previous(particle_count);
    for (std::size_t index = 0; index < movable.size(); ++index) current[index] = database.modules[movable[index]].center;
    for (std::size_t index = 0; index < fillers.size(); ++index) current[movable.size() + index] = fillers[index].center;
    previous = current;

    double penalty = 1.0;
    const double base_movement = options.maximum_movement_in_bins *
                                 std::min(density_map.binWidth(), density_map.binHeight());
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
        const double fraction = static_cast<double>(iteration) /
                                static_cast<double>(std::max(1, options.iterations - 1));
        const double momentum = std::min(0.92, static_cast<double>(iteration) /
                                                  static_cast<double>(iteration + 3));
        std::vector<Vec2> lookahead(particle_count);
        for (std::size_t index = 0; index < particle_count; ++index) {
            lookahead[index] = current[index] + (current[index] - previous[index]) * momentum;
        }
        for (std::size_t index = 0; index < movable.size(); ++index) {
            Module& module = database.modules[movable[index]];
            module.center = clampCenterToRegion(lookahead[index], module.width, module.height, database.core_region);
            lookahead[index] = module.center;
        }
        for (std::size_t index = 0; index < fillers.size(); ++index) {
            DensityFiller& filler = fillers[index];
            const std::size_t particle = movable.size() + index;
            filler.center = clampCenterToRegion(lookahead[particle], filler.width, filler.height,
                                                database.core_region);
            lookahead[particle] = filler.center;
        }

        depositDynamicDensity(density_map, database, fillers);
        buildDensityDeviation(density_map, density_deviation);
        density_field.solve(density_deviation);
        const double smoothing = (options.initial_smoothing * (1.0 - fraction) +
                                  options.final_smoothing * fraction) *
                                 std::max(density_map.binWidth(), density_map.binHeight());
        const std::vector<Vec2> wire_gradient = smoothWirelengthGradient(database, smoothing);
        std::vector<Vec2> density_gradient(particle_count);
        for (std::size_t index = 0; index < movable.size(); ++index) {
            const Module& module = database.modules[movable[index]];
            density_gradient[index] = density_field.sampleField(module.center) *
                                      (-densityChargeArea(module, options.target_density));
        }
        for (std::size_t index = 0; index < fillers.size(); ++index) {
            density_gradient[movable.size() + index] = density_field.sampleField(fillers[index].center) *
                                                        (-densityChargeArea(fillers[index], options.target_density));
        }

        if (iteration == 0) {
            const double wire_norm = l1Norm(wire_gradient);
            const double density_norm = l1Norm(density_gradient);
            penalty = density_norm > kEpsilon ? wire_norm / density_norm : 1.0;
            penalty = clamp(penalty, 1e-4, 1e6);
        }

        std::vector<Vec2> total_gradient(particle_count);
        double maximum_gradient = 0.0;
        for (std::size_t index = 0; index < movable.size(); ++index) {
            total_gradient[index] = wire_gradient[movable[index]] + density_gradient[index] * penalty;
            maximum_gradient = std::max(maximum_gradient, norm(total_gradient[index]));
        }
        for (std::size_t index = 0; index < fillers.size(); ++index) {
            const std::size_t particle = movable.size() + index;
            total_gradient[particle] = density_gradient[particle] * penalty;
            maximum_gradient = std::max(maximum_gradient, norm(total_gradient[particle]));
        }

        std::vector<Vec2> next = lookahead;
        double applied_scale = 0.0;
        double applied_movement = 0.0;
        if (maximum_gradient > kEpsilon) {
            applied_movement = base_movement * (1.0 - 0.55 * fraction);
            applied_scale = applied_movement / maximum_gradient;
            for (std::size_t index = 0; index < movable.size(); ++index) {
                Module& module = database.modules[movable[index]];
                module.center = clampCenterToRegion(lookahead[index] - total_gradient[index] * applied_scale,
                                                    module.width, module.height, database.core_region);
                next[index] = module.center;
            }
            for (std::size_t index = 0; index < fillers.size(); ++index) {
                DensityFiller& filler = fillers[index];
                const std::size_t particle = movable.size() + index;
                filler.center = clampCenterToRegion(lookahead[particle] - total_gradient[particle] * applied_scale,
                                                    filler.width, filler.height, database.core_region);
                next[particle] = filler.center;
            }
        }

        previous = current;
        current = next;
        depositDynamicDensity(density_map, database, fillers);
        const DensityMetrics density_metrics = density_map.metrics();
        if (iteration % std::max(1, options.report_interval) == 0 || iteration + 1 == options.iterations) {
            GlobalPlacementIteration history;
            history.iteration = iteration + 1;
            history.hpwl = calculateHpwl(database).hpwl;
            history.overflow = density_metrics.normalized_overflow;
            history.design_overflow = density_map.metrics(false).normalized_overflow;
            history.penalty = penalty;
            history.smoothing = smoothing;
            history.step_size = applied_scale;
            history.maximum_displacement = applied_movement;
            history.gradient_norm = maximum_gradient;
            history.accepted = maximum_gradient > kEpsilon;
            result.history.push_back(history);
        }
        result.completed_iterations = iteration + 1;
        if (maximum_gradient > kEpsilon) ++result.accepted_iterations;
        if (iteration > 20 && density_metrics.normalized_overflow <= options.maximum_bin_overflow) break;
        penalty = std::min(penalty * options.penalty_growth, 1e9);
    }

    result.hpwl_after = calculateHpwl(database).hpwl;
    result.overflow_after = calculateDensity(database, options.bins_x, options.bins_y, options.target_density)
                                .normalized_overflow;
    result.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return result;
}

}  // namespace myplacement::detail
