#pragma once

#include <vector>

#include "myplacement/metrics/Metrics.hpp"
#include "myplacement/model/PlacementDatabase.hpp"
#include "myplacement/placement/GlobalPlacer.hpp"

namespace myplacement::detail {

// Virtual density-only object. It never enters PlacementDatabase, legalization,
// or export; both global placers share this exact representation.
struct DensityFiller {
    Vec2 center;
    double width = 0.0;
    double height = 0.0;

    [[nodiscard]] double area() const { return width * height; }
    [[nodiscard]] Rect rect() const {
        return {{center.x - width * 0.5, center.y - height * 0.5},
                {center.x + width * 0.5, center.y + height * 0.5}};
    }
};

Vec2 clampCenterToRegion(Vec2 center, double width, double height, const Rect& region);

// The course density definition lives in one shared implementation so adaptive
// and legacy runs cannot drift in their row mask, filler, or density source.
void initializeStaticDensity(DensityMap& density_map, const PlacementDatabase& database);
void depositDynamicDensity(DensityMap& density_map, const PlacementDatabase& database,
                           const std::vector<DensityFiller>& fillers);
void buildDensityDeviation(const DensityMap& density_map, std::vector<double>& output);
std::vector<DensityFiller> createDensityFillers(const PlacementDatabase& database,
                                                const DensityMap& static_density,
                                                const GlobalPlacementOptions& options);

[[nodiscard]] double densityChargeArea(const Module& module, double target_density);
[[nodiscard]] double densityChargeArea(const DensityFiller& filler, double target_density);

}  // namespace myplacement::detail
