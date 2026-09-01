#pragma once

#include <cstddef>
#include <vector>

#include "myplacement/core/ResourceLimits.hpp"
#include "myplacement/model/PlacementDatabase.hpp"

namespace myplacement {

// Areas are kept in physical units. DensityMap applies the target-density
// scaling prescribed by the course ePlace model when it evaluates a bin:
// standard cells contribute at full area, while macros, fixed terminals,
// fillers, and unusable (dark) area consume target-density-scaled capacity.
enum class DensityLayer { Fixed, Movable, Macro, Filler, Dark };

struct DensityBin {
    Rect region;
    double fixed_area = 0.0;
    double movable_area = 0.0;
    double macro_area = 0.0;
    double filler_area = 0.0;
    double dark_area = 0.0;

    [[nodiscard]] double chargeArea(double target_density, bool include_fillers = true) const {
        const double scaled_area = fixed_area + macro_area + dark_area +
                                   (include_fillers ? filler_area : 0.0);
        return movable_area + target_density * scaled_area;
    }

    // This is the denominator of the course overflow definition. Fixed and
    // dark area affect congestion, but must never dilute the normalized result.
    [[nodiscard]] double normalizationArea(double target_density) const {
        return movable_area + target_density * macro_area;
    }
};

struct DensityMetrics {
    double total_overflow_area = 0.0;
    double normalized_overflow = 0.0;
    double maximum_density = 0.0;
    double average_density = 0.0;
    double total_charge_area = 0.0;
    double normalization_area = 0.0;
    double placeable_area = 0.0;
    double dark_area = 0.0;
};

class DensityMap {
public:
    DensityMap(Rect region, int columns, int rows, double target_density);

    // Configures the legal placement mask from SiteRow rectangles. Regions not
    // covered by a row become dark density and therefore cannot absorb cells.
    void setPlaceableRegions(const std::vector<Rect>& regions);
    void clearDynamic();
    void addRectangle(const Rect& rect, DensityLayer layer);
    [[nodiscard]] DensityMetrics metrics(bool include_fillers = true) const;
    [[nodiscard]] int columns() const { return columns_; }
    [[nodiscard]] int rows() const { return rows_; }
    [[nodiscard]] double binWidth() const { return bin_width_; }
    [[nodiscard]] double binHeight() const { return bin_height_; }
    [[nodiscard]] double targetDensity() const { return target_density_; }
    [[nodiscard]] const Rect& region() const { return region_; }
    [[nodiscard]] const DensityBin& at(int column, int row) const;
    [[nodiscard]] const std::vector<DensityBin>& bins() const { return bins_; }

private:
    Rect region_;
    int columns_ = 0;
    int rows_ = 0;
    double target_density_ = 0.0;
    double bin_width_ = 0.0;
    double bin_height_ = 0.0;
    std::vector<DensityBin> bins_;
};

struct WirelengthMetrics {
    double hpwl = 0.0;
    std::size_t counted_nets = 0;
};

WirelengthMetrics calculateHpwl(const PlacementDatabase& database);
DensityMetrics calculateDensity(const PlacementDatabase& database, int columns, int rows,
                                double target_density);

}  // namespace myplacement
