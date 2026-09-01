#include "myplacement/metrics/Metrics.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace myplacement {

DensityMap::DensityMap(Rect region, int columns, int rows, double target_density)
    : region_(region), columns_(columns), rows_(rows), target_density_(target_density) {
    if (!region_.valid() || region_.area() <= kEpsilon || columns_ <= 0 || rows_ <= 0 ||
        target_density_ <= 0.0) {
        throw std::invalid_argument("DensityMap requires a valid region, positive dimensions, and target density.");
    }
    const std::size_t columns_size = static_cast<std::size_t>(columns_);
    const std::size_t rows_size = static_cast<std::size_t>(rows_);
    if (columns_size > kMaximumDensityBinCount / rows_size) {
        throw std::invalid_argument("Density grid exceeds the safe one-million-bin workspace limit.");
    }
    bin_width_ = region_.width() / static_cast<double>(columns_);
    bin_height_ = region_.height() / static_cast<double>(rows_);
    bins_.resize(columns_size * rows_size);
    for (int row = 0; row < rows_; ++row) {
        for (int column = 0; column < columns_; ++column) {
            const double left = region_.ll.x + static_cast<double>(column) * bin_width_;
            const double bottom = region_.ll.y + static_cast<double>(row) * bin_height_;
            bins_[static_cast<std::size_t>(row * columns_ + column)].region = {
                {left, bottom}, {left + bin_width_, bottom + bin_height_}};
        }
    }
}

void DensityMap::clearDynamic() {
    for (DensityBin& bin : bins_) {
        bin.movable_area = 0.0;
        bin.macro_area = 0.0;
        bin.filler_area = 0.0;
    }
}

void DensityMap::setPlaceableRegions(const std::vector<Rect>& regions) {
    // BookShelf SiteRows describe disjoint legal placement rectangles. Begin
    // from a completely dark core and subtract every legal row contribution.
    for (DensityBin& bin : bins_) bin.dark_area = bin.region.area();
    for (const Rect& region : regions) {
        const Rect clipped = region.intersection(region_);
        if (clipped.area() <= kEpsilon) continue;

        const int start_column = std::max(0, std::min(columns_ - 1,
            static_cast<int>(std::floor((clipped.ll.x - region_.ll.x) / bin_width_))));
        const int end_column = std::max(0, std::min(columns_ - 1,
            static_cast<int>(std::ceil((clipped.ur.x - region_.ll.x) / bin_width_) - 1.0)));
        const int start_row = std::max(0, std::min(rows_ - 1,
            static_cast<int>(std::floor((clipped.ll.y - region_.ll.y) / bin_height_))));
        const int end_row = std::max(0, std::min(rows_ - 1,
            static_cast<int>(std::ceil((clipped.ur.y - region_.ll.y) / bin_height_) - 1.0)));
        for (int row = start_row; row <= end_row; ++row) {
            for (int column = start_column; column <= end_column; ++column) {
                DensityBin& bin = bins_[static_cast<std::size_t>(row * columns_ + column)];
                bin.dark_area -= clipped.overlapArea(bin.region);
            }
        }
    }
    for (DensityBin& bin : bins_) {
        const double tolerance = std::max(1.0, bin.region.area()) * 1e-10;
        if (bin.dark_area < -tolerance) {
            throw std::invalid_argument("Overlapping SiteRows cannot define a density placeable-area mask.");
        }
        bin.dark_area = std::max(0.0, bin.dark_area);
    }
}

void DensityMap::addRectangle(const Rect& rect, DensityLayer layer) {
    const Rect clipped = rect.intersection(region_);
    if (clipped.area() <= kEpsilon) return;

    const int start_column = std::max(0, std::min(columns_ - 1,
        static_cast<int>(std::floor((clipped.ll.x - region_.ll.x) / bin_width_))));
    const int end_column = std::max(0, std::min(columns_ - 1,
        static_cast<int>(std::ceil((clipped.ur.x - region_.ll.x) / bin_width_) - 1.0)));
    const int start_row = std::max(0, std::min(rows_ - 1,
        static_cast<int>(std::floor((clipped.ll.y - region_.ll.y) / bin_height_))));
    const int end_row = std::max(0, std::min(rows_ - 1,
        static_cast<int>(std::ceil((clipped.ur.y - region_.ll.y) / bin_height_) - 1.0)));

    for (int row = start_row; row <= end_row; ++row) {
        for (int column = start_column; column <= end_column; ++column) {
            DensityBin& bin = bins_[static_cast<std::size_t>(row * columns_ + column)];
            const double area = clipped.overlapArea(bin.region);
            if (area <= kEpsilon) continue;
            if (layer == DensityLayer::Fixed) bin.fixed_area += area;
            if (layer == DensityLayer::Movable) bin.movable_area += area;
            if (layer == DensityLayer::Macro) bin.macro_area += area;
            if (layer == DensityLayer::Filler) bin.filler_area += area;
            if (layer == DensityLayer::Dark) bin.dark_area += area;
        }
    }
}

DensityMetrics DensityMap::metrics(bool include_fillers) const {
    DensityMetrics result;
    double total_density = 0.0;
    for (const DensityBin& bin : bins_) {
        const double bin_area = bin.region.area();
        const double charge_area = bin.chargeArea(target_density_, include_fillers);
        const double density = charge_area / std::max(bin_area, kEpsilon);
        result.maximum_density = std::max(result.maximum_density, density);
        total_density += density;
        result.total_overflow_area += std::max(0.0, charge_area - target_density_ * bin_area);
        result.total_charge_area += charge_area;
        result.normalization_area += bin.normalizationArea(target_density_);
        result.dark_area += bin.dark_area;
        result.placeable_area += bin_area - bin.dark_area;
    }
    result.average_density = total_density / static_cast<double>(std::max<std::size_t>(1U, bins_.size()));
    result.normalized_overflow = result.total_overflow_area / std::max(result.normalization_area, kEpsilon);
    return result;
}

const DensityBin& DensityMap::at(int column, int row) const {
    if (column < 0 || column >= columns_ || row < 0 || row >= rows_) {
        throw std::out_of_range("Density bin index out of range.");
    }
    return bins_.at(static_cast<std::size_t>(row * columns_ + column));
}

WirelengthMetrics calculateHpwl(const PlacementDatabase& database) {
    WirelengthMetrics result;
    for (const Net& net : database.nets) {
        if (net.pins.size() < 2U) continue;
        double min_x = std::numeric_limits<double>::infinity();
        double max_x = -std::numeric_limits<double>::infinity();
        double min_y = std::numeric_limits<double>::infinity();
        double max_y = -std::numeric_limits<double>::infinity();
        for (const PinId pin_id : net.pins) {
            const Vec2 position = database.pinPosition(pin_id);
            min_x = std::min(min_x, position.x);
            max_x = std::max(max_x, position.x);
            min_y = std::min(min_y, position.y);
            max_y = std::max(max_y, position.y);
        }
        result.hpwl += net.weight * ((max_x - min_x) + (max_y - min_y));
        ++result.counted_nets;
    }
    return result;
}

DensityMetrics calculateDensity(const PlacementDatabase& database, int columns, int rows,
                                double target_density) {
    DensityMap map(database.core_region, columns, rows, target_density);
    std::vector<Rect> placeable_regions;
    placeable_regions.reserve(database.rows.size());
    for (const SiteRow& row : database.rows) placeable_regions.push_back(row.rect());
    if (!placeable_regions.empty()) map.setPlaceableRegions(placeable_regions);
    for (const Module& module : database.modules) {
        if (module.is_fixed) {
            map.addRectangle(module.rect(), DensityLayer::Fixed);
        } else {
            map.addRectangle(module.rect(), module.is_macro ? DensityLayer::Macro : DensityLayer::Movable);
        }
    }
    return map.metrics();
}

}  // namespace myplacement
