#include "GlobalPlacementSupport.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>

namespace myplacement::detail {

Vec2 clampCenterToRegion(Vec2 center, double width, double height, const Rect& region) {
    const double min_x = region.ll.x + width * 0.5;
    const double max_x = region.ur.x - width * 0.5;
    const double min_y = region.ll.y + height * 0.5;
    const double max_y = region.ur.y - height * 0.5;
    center.x = min_x <= max_x ? clamp(center.x, min_x, max_x) : region.center().x;
    center.y = min_y <= max_y ? clamp(center.y, min_y, max_y) : region.center().y;
    return center;
}

void initializeStaticDensity(DensityMap& density_map, const PlacementDatabase& database) {
    std::vector<Rect> placeable_regions;
    placeable_regions.reserve(database.rows.size());
    for (const SiteRow& row : database.rows) placeable_regions.push_back(row.rect());
    if (!placeable_regions.empty()) density_map.setPlaceableRegions(placeable_regions);

    for (const Module& module : database.modules) {
        if (module.is_fixed) density_map.addRectangle(module.rect(), DensityLayer::Fixed);
    }
}

void depositDynamicDensity(DensityMap& density_map, const PlacementDatabase& database,
                           const std::vector<DensityFiller>& fillers) {
    density_map.clearDynamic();
    for (const ModuleId id : database.movableModules()) {
        const Module& module = database.modules[id];
        density_map.addRectangle(module.rect(), module.is_macro ? DensityLayer::Macro : DensityLayer::Movable);
    }
    for (const DensityFiller& filler : fillers) density_map.addRectangle(filler.rect(), DensityLayer::Filler);
}

void buildDensityDeviation(const DensityMap& density_map, std::vector<double>& output) {
    const std::vector<DensityBin>& bins = density_map.bins();
    output.resize(bins.size());
    for (std::size_t index = 0; index < bins.size(); ++index) {
        output[index] = bins[index].chargeArea(density_map.targetDensity()) /
                            std::max(bins[index].region.area(), kEpsilon) -
                        density_map.targetDensity();
    }
}

std::vector<DensityFiller> createDensityFillers(const PlacementDatabase& database,
                                                const DensityMap& static_density,
                                                const GlobalPlacementOptions& options) {
    double standard_cell_area = 0.0;
    double macro_area = 0.0;
    std::vector<double> areas;
    areas.reserve(database.movableModules().size());
    for (const ModuleId id : database.movableModules()) {
        const Module& module = database.modules[id];
        const double area = module.rect().intersection(database.core_region).area();
        if (module.is_macro) {
            macro_area += area;
        } else {
            standard_cell_area += area;
        }
        if (area > kEpsilon) areas.push_back(area);
    }

    const double static_charge_area = static_density.metrics(false).total_charge_area;
    const double movable_charge_area = standard_cell_area + options.target_density * macro_area;
    const double required_filler_charge_area = std::max(
        0.0, options.target_density * database.core_region.area() - static_charge_area - movable_charge_area);
    // Filler charge is target-density-scaled, so recover its physical area.
    const double filler_area = required_filler_charge_area / std::max(options.target_density, kEpsilon);
    if (filler_area <= kEpsilon || areas.empty() || options.maximum_fillers == 0U) return {};

    std::sort(areas.begin(), areas.end());
    const std::size_t lower = areas.size() / 10U;
    const std::size_t upper = std::max(lower + 1U, areas.size() - areas.size() / 10U);
    const double representative_area = std::accumulate(areas.begin() + static_cast<std::ptrdiff_t>(lower),
                                                        areas.begin() + static_cast<std::ptrdiff_t>(upper), 0.0) /
                                       static_cast<double>(upper - lower);
    const std::size_t count = std::min(options.maximum_fillers,
                                       std::max<std::size_t>(1U, static_cast<std::size_t>(std::ceil(
                                           filler_area / std::max(representative_area, kEpsilon)))));
    const double area_per_filler = filler_area / static_cast<double>(count);
    const double aspect = std::max(database.core_region.width() / database.core_region.height(), 0.1);
    const double width = std::sqrt(area_per_filler * aspect);
    const double height = area_per_filler / std::max(width, kEpsilon);

    std::mt19937_64 generator(options.seed ^ 0x9e3779b97f4a7c15ULL);
    std::uniform_real_distribution<double> x_distribution(database.core_region.ll.x + width * 0.5,
                                                           database.core_region.ur.x - width * 0.5);
    std::uniform_real_distribution<double> y_distribution(database.core_region.ll.y + height * 0.5,
                                                           database.core_region.ur.y - height * 0.5);
    std::vector<DensityFiller> fillers;
    fillers.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        fillers.push_back({{x_distribution(generator), y_distribution(generator)}, width, height});
    }
    return fillers;
}

double densityChargeArea(const Module& module, double target_density) {
    return module.is_macro ? target_density * module.area() : module.area();
}

double densityChargeArea(const DensityFiller& filler, double target_density) {
    return target_density * filler.area();
}

}  // namespace myplacement::detail
