#include "myplacement/placement/Legalizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

namespace myplacement {
namespace {

bool isStandardCell(const Module& module) {
    return !module.is_fixed && !module.is_macro;
}

Vec2 clampCenter(Vec2 center, double width, double height, const Rect& region) {
    const double min_x = region.ll.x + width * 0.5;
    const double max_x = region.ur.x - width * 0.5;
    const double min_y = region.ll.y + height * 0.5;
    const double max_y = region.ur.y - height * 0.5;
    center.x = min_x <= max_x ? clamp(center.x, min_x, max_x) : region.center().x;
    center.y = min_y <= max_y ? clamp(center.y, min_y, max_y) : region.center().y;
    return center;
}

double alignUp(double value, double origin, double step) {
    if (step <= kEpsilon) return value;
    return origin + std::ceil((value - origin - kEpsilon) / step) * step;
}

double alignDown(double value, double origin, double step) {
    if (step <= kEpsilon) return value;
    return origin + std::floor((value - origin + kEpsilon) / step) * step;
}

double alignNearest(double value, double origin, double step) {
    if (step <= kEpsilon) return value;
    return origin + std::round((value - origin) / step) * step;
}

struct BlockingInterval {
    double left = 0.0;
    double right = 0.0;
};

struct RowSlot {
    std::size_t row_index = 0;
    double left = 0.0;
    double right = 0.0;
    double remaining = 0.0;
    std::vector<ModuleId> modules;
};

struct IsotonicBlock {
    std::size_t begin = 0;
    std::size_t end = 0;
    double total_weight = 0.0;
    double weighted_sum = 0.0;

    [[nodiscard]] double value() const { return weighted_sum / std::max(total_weight, kEpsilon); }
};

std::vector<Rect> fixedAndMacroObstacles(const PlacementDatabase& database, bool include_movable_macros) {
    std::vector<Rect> obstacles;
    for (const Module& module : database.modules) {
        if (module.is_fixed || (include_movable_macros && module.is_macro)) obstacles.push_back(module.rect());
    }
    return obstacles;
}

bool fitsWithoutOverlap(const Rect& candidate, const Rect& core, const std::vector<Rect>& obstacles) {
    if (candidate.ll.x < core.ll.x - kEpsilon || candidate.ur.x > core.ur.x + kEpsilon ||
        candidate.ll.y < core.ll.y - kEpsilon || candidate.ur.y > core.ur.y + kEpsilon) {
        return false;
    }
    return std::none_of(obstacles.begin(), obstacles.end(), [&](const Rect& obstacle) {
        return candidate.overlaps(obstacle);
    });
}

void legalizeMacros(PlacementDatabase& database, const LegalizationOptions& options) {
    std::vector<ModuleId> macros;
    for (ModuleId id = 0; id < database.modules.size(); ++id) {
        if (!database.modules[id].is_fixed && database.modules[id].is_macro) macros.push_back(id);
    }
    std::sort(macros.begin(), macros.end(), [&](ModuleId left, ModuleId right) {
        return database.modules[left].area() > database.modules[right].area();
    });
    std::vector<Rect> obstacles = fixedAndMacroObstacles(database, false);
    const double row_height = std::max(database.nominalRowHeight(), 1.0);

    for (const ModuleId id : macros) {
        Module& macro = database.modules[id];
        const Vec2 ideal = macro.center;
        const double step = std::max(row_height, std::min(macro.width, macro.height) * 0.25);
        Vec2 best = clampCenter(ideal, macro.width, macro.height, database.core_region);
        bool found = false;
        double best_distance = std::numeric_limits<double>::infinity();
        for (int radius = 0; radius <= options.macro_search_radius; ++radius) {
            const std::vector<std::pair<int, int>> offsets = radius == 0
                ? std::vector<std::pair<int, int>>{{0, 0}}
                : std::vector<std::pair<int, int>>{{-radius, -radius}, {-radius, 0}, {-radius, radius},
                                                   {0, -radius},                    {0, radius},
                                                   {radius, -radius},  {radius, 0},  {radius, radius}};
            for (const auto& [offset_x, offset_y] : offsets) {
                const Vec2 candidate_center = clampCenter(
                    {ideal.x + static_cast<double>(offset_x) * step, ideal.y + static_cast<double>(offset_y) * step},
                    macro.width, macro.height, database.core_region);
                const Rect candidate{{candidate_center.x - macro.width * 0.5, candidate_center.y - macro.height * 0.5},
                                     {candidate_center.x + macro.width * 0.5, candidate_center.y + macro.height * 0.5}};
                if (!fitsWithoutOverlap(candidate, database.core_region, obstacles)) continue;
                const double distance = squaredNorm(candidate_center - ideal);
                if (distance < best_distance) {
                    best = candidate_center;
                    best_distance = distance;
                    found = true;
                }
            }
            if (found) break;
        }
        macro.center = best;
        obstacles.push_back(macro.rect());
    }
}

std::vector<RowSlot> buildRowSlots(const PlacementDatabase& database) {
    const std::vector<Rect> obstacles = fixedAndMacroObstacles(database, true);
    std::vector<RowSlot> slots;
    for (std::size_t row_index = 0; row_index < database.rows.size(); ++row_index) {
        const SiteRow& row = database.rows[row_index];
        std::vector<BlockingInterval> blocked;
        for (const Rect& obstacle : obstacles) {
            if (obstacle.ur.y <= row.bottom + kEpsilon || obstacle.ll.y >= row.bottom + row.height - kEpsilon) {
                continue;
            }
            const double left = std::max(obstacle.ll.x, row.x_start);
            const double right = std::min(obstacle.ur.x, row.xEnd());
            if (right - left > kEpsilon) blocked.push_back({left, right});
        }
        std::sort(blocked.begin(), blocked.end(), [](const BlockingInterval& left, const BlockingInterval& right) {
            return left.left < right.left;
        });
        std::vector<BlockingInterval> merged;
        for (const BlockingInterval interval : blocked) {
            if (merged.empty() || interval.left > merged.back().right + kEpsilon) {
                merged.push_back(interval);
            } else {
                merged.back().right = std::max(merged.back().right, interval.right);
            }
        }
        double cursor = row.x_start;
        const auto addSlot = [&](double left, double right) {
            const double aligned_left = alignUp(left, row.x_start, row.site_spacing);
            const double aligned_right = alignDown(right, row.x_start, row.site_spacing);
            if (aligned_right - aligned_left + kEpsilon >= row.site_spacing) {
                slots.push_back({row_index, aligned_left, aligned_right, aligned_right - aligned_left, {}});
            }
        };
        for (const BlockingInterval interval : merged) {
            addSlot(cursor, interval.left);
            cursor = std::max(cursor, interval.right);
        }
        addSlot(cursor, row.xEnd());
    }
    return slots;
}

std::vector<std::size_t> sortedRowIndices(const PlacementDatabase& database) {
    std::vector<std::size_t> indices(database.rows.size());
    std::iota(indices.begin(), indices.end(), 0U);
    std::sort(indices.begin(), indices.end(), [&](std::size_t left, std::size_t right) {
        return database.rows[left].bottom < database.rows[right].bottom;
    });
    return indices;
}

std::size_t nearestSortedRow(const PlacementDatabase& database, const std::vector<std::size_t>& sorted_rows,
                             double coordinate) {
    const auto iterator = std::lower_bound(sorted_rows.begin(), sorted_rows.end(), coordinate,
        [&](std::size_t row_index, double value) {
            const SiteRow& row = database.rows[row_index];
            return row.bottom + row.height * 0.5 < value;
        });
    if (iterator == sorted_rows.begin()) return 0;
    if (iterator == sorted_rows.end()) return sorted_rows.size() - 1U;
    const std::size_t right = static_cast<std::size_t>(iterator - sorted_rows.begin());
    const std::size_t left = right - 1U;
    const double left_distance = std::abs(database.rows[sorted_rows[left]].bottom +
                                          database.rows[sorted_rows[left]].height * 0.5 - coordinate);
    const double right_distance = std::abs(database.rows[sorted_rows[right]].bottom +
                                           database.rows[sorted_rows[right]].height * 0.5 - coordinate);
    return left_distance <= right_distance ? left : right;
}

double roundedWidth(const Module& module, const SiteRow& row) {
    return alignUp(module.width, 0.0, row.site_spacing);
}

void packRowSlot(PlacementDatabase& database, RowSlot& slot) {
    if (slot.modules.empty()) return;
    const SiteRow& row = database.rows[slot.row_index];
    std::sort(slot.modules.begin(), slot.modules.end(), [&](ModuleId left, ModuleId right) {
        return database.modules[left].lowerLeft().x < database.modules[right].lowerLeft().x;
    });
    std::vector<double> widths(slot.modules.size());
    std::vector<double> prefix(slot.modules.size(), 0.0);
    double total_width = 0.0;
    for (std::size_t index = 0; index < slot.modules.size(); ++index) {
        widths[index] = roundedWidth(database.modules[slot.modules[index]], row);
        prefix[index] = total_width;
        total_width += widths[index];
    }
    if (total_width > slot.right - slot.left + kEpsilon) return;

    std::vector<IsotonicBlock> blocks;
    for (std::size_t index = 0; index < slot.modules.size(); ++index) {
        const Module& module = database.modules[slot.modules[index]];
        const double weight = std::max(module.area(), 1.0);
        const double desired = module.lowerLeft().x - prefix[index];
        blocks.push_back({index, index + 1U, weight, weight * desired});
        while (blocks.size() >= 2U && blocks[blocks.size() - 2U].value() > blocks.back().value()) {
            IsotonicBlock right = blocks.back();
            blocks.pop_back();
            IsotonicBlock& left = blocks.back();
            left.end = right.end;
            left.total_weight += right.total_weight;
            left.weighted_sum += right.weighted_sum;
        }
    }

    const double lower = slot.left;
    const double upper = slot.right - total_width;
    for (const IsotonicBlock& block : blocks) {
        const double offset = alignNearest(clamp(block.value(), lower, upper), row.x_start, row.site_spacing);
        for (std::size_t index = block.begin; index < block.end; ++index) {
            Module& module = database.modules[slot.modules[index]];
            const double lower_left_x = offset + prefix[index];
            module.setLowerLeft({lower_left_x, row.bottom});
            if (row.orientation != Orientation::Unknown) module.orientation = row.orientation;
        }
    }
}

}  // namespace

LegalityReport Legalizer::legalize(PlacementDatabase& database, const LegalizationOptions& options) const {
    LegalizationOptions normalized_options = options;
    normalized_options.candidate_rows = std::max(1, normalized_options.candidate_rows);
    legalizeMacros(database, normalized_options);
    std::vector<RowSlot> slots = buildRowSlots(database);
    const std::vector<std::size_t> sorted_rows = sortedRowIndices(database);
    std::vector<std::vector<std::size_t>> slots_by_row(database.rows.size());
    for (std::size_t slot_index = 0; slot_index < slots.size(); ++slot_index) {
        slots_by_row[slots[slot_index].row_index].push_back(slot_index);
    }

    std::vector<ModuleId> standard_cells;
    for (ModuleId id = 0; id < database.modules.size(); ++id) {
        if (isStandardCell(database.modules[id])) standard_cells.push_back(id);
    }
    std::sort(standard_cells.begin(), standard_cells.end(), [&](ModuleId left, ModuleId right) {
        return database.modules[left].center.y < database.modules[right].center.y;
    });

    LegalityReport result;
    for (const ModuleId id : standard_cells) {
        Module& module = database.modules[id];
        const double ideal_x = module.lowerLeft().x;
        const std::size_t nearest = nearestSortedRow(database, sorted_rows, module.center.y);
        std::size_t best_slot = slots.size();
        double best_cost = std::numeric_limits<double>::infinity();
        const int lower = std::max(0, static_cast<int>(nearest) - normalized_options.candidate_rows);
        const int upper = std::min(static_cast<int>(sorted_rows.size()) - 1,
                                   static_cast<int>(nearest) + normalized_options.candidate_rows);
        const auto considerRow = [&](std::size_t row_index) {
            const SiteRow& row = database.rows[row_index];
            const double width = roundedWidth(module, row);
            for (const std::size_t slot_index : slots_by_row[row_index]) {
                const RowSlot& slot = slots[slot_index];
                if (slot.remaining + kEpsilon < width) continue;
                const double projected_x = clamp(ideal_x, slot.left, slot.right - width);
                const double cost = std::abs(module.center.y - (row.bottom + row.height * 0.5)) +
                                    0.15 * std::abs(ideal_x - projected_x);
                if (cost < best_cost) {
                    best_cost = cost;
                    best_slot = slot_index;
                }
            }
        };
        for (int sorted_index = lower; sorted_index <= upper; ++sorted_index) {
            considerRow(sorted_rows[static_cast<std::size_t>(sorted_index)]);
        }
        if (best_slot == slots.size()) {
            for (std::size_t row_index = 0; row_index < database.rows.size(); ++row_index) considerRow(row_index);
        }
        if (best_slot == slots.size()) {
            ++result.unplaced_standard_cells;
            continue;
        }
        const SiteRow& target_row = database.rows[slots[best_slot].row_index];
        slots[best_slot].remaining -= roundedWidth(module, target_row);
        slots[best_slot].modules.push_back(id);
        ++result.placed_standard_cells;
    }
    for (RowSlot& slot : slots) packRowSlot(database, slot);

    LegalityReport checked = check(database, normalized_options.epsilon);
    checked.placed_standard_cells = result.placed_standard_cells;
    checked.unplaced_standard_cells = result.unplaced_standard_cells;
    checked.legal = checked.legal && checked.unplaced_standard_cells == 0U;
    return checked;
}

LegalityReport Legalizer::check(const PlacementDatabase& database, double epsilon) const {
    LegalityReport result;
    const std::vector<std::size_t> sorted_rows = sortedRowIndices(database);
    std::vector<std::vector<ModuleId>> cells_in_row(database.rows.size());
    std::vector<std::pair<ModuleId, Rect>> obstacles;
    for (ModuleId id = 0; id < database.modules.size(); ++id) {
        const Module& module = database.modules[id];
        if (module.is_fixed || module.is_macro) obstacles.emplace_back(id, module.rect());
    }
    std::sort(obstacles.begin(), obstacles.end(), [](const auto& left, const auto& right) {
        return left.second.ll.y < right.second.ll.y;
    });
    const auto countObstacleOverlaps = [&](ModuleId module_id, const Rect& rectangle) {
        const auto end = std::upper_bound(obstacles.begin(), obstacles.end(), rectangle.ur.y + epsilon,
            [](double upper_y, const auto& obstacle) { return upper_y < obstacle.second.ll.y; });
        for (auto iterator = obstacles.begin(); iterator != end; ++iterator) {
            const auto& [obstacle_id, obstacle] = *iterator;
            if (obstacle_id != module_id && obstacle.ur.y > rectangle.ll.y + epsilon &&
                rectangle.overlaps(obstacle)) {
                ++result.overlap_pairs;
            }
        }
    };

    for (ModuleId id = 0; id < database.modules.size(); ++id) {
        const Module& module = database.modules[id];
        if (module.is_fixed) continue;
        const Rect rect = module.rect();
        if (rect.ll.x < database.core_region.ll.x - epsilon || rect.ur.x > database.core_region.ur.x + epsilon ||
            rect.ll.y < database.core_region.ll.y - epsilon || rect.ur.y > database.core_region.ur.y + epsilon) {
            ++result.out_of_core_modules;
        }
        if (module.is_macro) {
            countObstacleOverlaps(id, rect);
            continue;
        }
        const std::size_t row_position = nearestSortedRow(database, sorted_rows, rect.ll.y + module.height * 0.5);
        const std::size_t row_index = sorted_rows[row_position];
        const SiteRow& row = database.rows[row_index];
        const double x_remainder = std::abs(std::remainder(rect.ll.x - row.x_start, row.site_spacing));
        const bool x_is_misaligned = std::min(x_remainder, row.site_spacing - x_remainder) > epsilon;
        if (std::abs(rect.ll.y - row.bottom) > epsilon || x_is_misaligned || rect.ur.x > row.xEnd() + epsilon) {
            ++result.off_row_modules;
        }
        cells_in_row[row_index].push_back(id);
        countObstacleOverlaps(id, rect);
    }

    for (std::vector<ModuleId>& row_cells : cells_in_row) {
        std::sort(row_cells.begin(), row_cells.end(), [&](ModuleId left, ModuleId right) {
            return database.modules[left].lowerLeft().x < database.modules[right].lowerLeft().x;
        });
        for (std::size_t index = 1; index < row_cells.size(); ++index) {
            if (database.modules[row_cells[index - 1U]].rect().overlaps(database.modules[row_cells[index]].rect())) {
                ++result.overlap_pairs;
            }
        }
    }
    result.legal = result.overlap_pairs == 0U && result.out_of_core_modules == 0U &&
                   result.off_row_modules == 0U;
    return result;
}

}  // namespace myplacement
