#include "myplacement/placement/Legalizer.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace myplacement {
namespace {

bool isStandardCell(const Module& module) {
    return !module.is_fixed && !module.is_macro;
}

bool isKnownLegalizationStrategy(LegalizationStrategy strategy) {
    switch (strategy) {
        case LegalizationStrategy::Abacus:
        case LegalizationStrategy::GreedyIsotonic: return true;
    }
    return false;
}

void validateLegalityTolerance(double epsilon) {
    if (!std::isfinite(epsilon) || epsilon < 0.0) {
        throw std::invalid_argument("Legality epsilon must be finite and non-negative.");
    }
}

void validateLegalizationOptions(const LegalizationOptions& options) {
    if (!isKnownLegalizationStrategy(options.strategy) || options.candidate_rows <= 0 ||
        options.macro_search_radius < 0) {
        throw std::invalid_argument("Legalization options are outside their valid range.");
    }
    validateLegalityTolerance(options.epsilon);
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

// A RowSlot is a macro/fixed-cell-free part of a physical row.  It is shared
// by the retained greedy baseline and the Abacus implementation below.
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

struct PlacementOutcome {
    std::size_t placed_standard_cells = 0;
    std::size_t unplaced_standard_cells = 0;
};

struct MovementMetrics {
    double total_displacement = 0.0;
    double total_squared_displacement = 0.0;
    double weighted_squared_displacement = 0.0;
    double maximum_displacement = 0.0;
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

std::vector<ModuleId> standardCells(const PlacementDatabase& database) {
    std::vector<ModuleId> result;
    result.reserve(database.modules.size());
    for (ModuleId id = 0; id < database.modules.size(); ++id) {
        if (isStandardCell(database.modules[id])) result.push_back(id);
    }
    return result;
}

std::vector<Vec2> lowerLeftTargets(const PlacementDatabase& database) {
    std::vector<Vec2> targets;
    targets.reserve(database.modules.size());
    for (const Module& module : database.modules) targets.push_back(module.lowerLeft());
    return targets;
}

MovementMetrics calculateStandardCellMovement(const PlacementDatabase& database,
                                              const std::vector<ModuleId>& standard_cells,
                                              const std::vector<Vec2>& targets) {
    MovementMetrics result;
    for (const ModuleId id : standard_cells) {
        const Module& module = database.modules[id];
        const Vec2 delta = module.lowerLeft() - targets[id];
        const double squared_displacement = squaredNorm(delta);
        const double displacement = std::sqrt(squared_displacement);
        result.total_displacement += displacement;
        result.total_squared_displacement += squared_displacement;
        result.weighted_squared_displacement += std::max(module.area(), 1.0) * squared_displacement;
        result.maximum_displacement = std::max(result.maximum_displacement, displacement);
    }
    return result;
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

PlacementOutcome legalizeGreedyIsotonic(PlacementDatabase& database, const LegalizationOptions& options,
                                        const std::vector<ModuleId>& standard_cells,
                                        const std::vector<Vec2>& targets) {
    std::vector<RowSlot> slots = buildRowSlots(database);
    const std::vector<std::size_t> sorted_rows = sortedRowIndices(database);
    std::vector<std::vector<std::size_t>> slots_by_row(database.rows.size());
    for (std::size_t slot_index = 0; slot_index < slots.size(); ++slot_index) {
        slots_by_row[slots[slot_index].row_index].push_back(slot_index);
    }

    std::vector<ModuleId> cells = standard_cells;
    std::sort(cells.begin(), cells.end(), [&](ModuleId left, ModuleId right) {
        const double left_y = targets[left].y + database.modules[left].height * 0.5;
        const double right_y = targets[right].y + database.modules[right].height * 0.5;
        if (std::abs(left_y - right_y) > kEpsilon) return left_y < right_y;
        return left < right;
    });

    PlacementOutcome result;
    if (sorted_rows.empty()) {
        result.unplaced_standard_cells = cells.size();
        return result;
    }
    for (const ModuleId id : cells) {
        Module& module = database.modules[id];
        const Vec2 ideal = targets[id];
        const std::size_t nearest = nearestSortedRow(database, sorted_rows, ideal.y + module.height * 0.5);
        std::size_t best_slot = slots.size();
        double best_cost = std::numeric_limits<double>::infinity();
        const int lower = std::max(0, static_cast<int>(nearest) - options.candidate_rows);
        const int upper = std::min(static_cast<int>(sorted_rows.size()) - 1,
                                   static_cast<int>(nearest) + options.candidate_rows);
        const auto considerRow = [&](std::size_t row_index) {
            const SiteRow& row = database.rows[row_index];
            const double width = roundedWidth(module, row);
            for (const std::size_t slot_index : slots_by_row[row_index]) {
                const RowSlot& slot = slots[slot_index];
                if (slot.remaining + kEpsilon < width) continue;
                const double projected_x = clamp(ideal.x, slot.left, slot.right - width);
                const double cost = std::abs(ideal.y + module.height * 0.5 - (row.bottom + row.height * 0.5)) +
                                    0.15 * std::abs(ideal.x - projected_x);
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
    return result;
}

// The following data structures directly encode Algorithm 2 of Spindler et
// al.'s Abacus paper.  A trial only reconstructs the suffix of clusters that
// would collapse, rather than copying and re-placing a whole row.
struct AbacusEntry {
    ModuleId module_id = 0;
    double desired_x = 0.0;
    double desired_y = 0.0;
    double width = 0.0;
    double weight = 1.0;
};

struct AbacusCluster {
    std::size_t first_entry = 0;
    double total_weight = 0.0;
    double total_width = 0.0;
    double q = 0.0;
    double x = 0.0;
};

struct AbacusSlot {
    std::size_t row_index = 0;
    double left = 0.0;
    double right = 0.0;
    double grid_origin = 0.0;
    double site_spacing = 1.0;
    bool mirrored = false;
    double used_width = 0.0;
    std::vector<AbacusEntry> entries;
    std::vector<AbacusCluster> clusters;
};

struct AbacusTrial {
    bool feasible = false;
    std::size_t replace_from = 0;
    AbacusCluster cluster;
    double candidate_x = 0.0;
};

std::vector<AbacusSlot> buildAbacusSlots(const PlacementDatabase& database, bool reverse_x) {
    const std::vector<RowSlot> row_slots = buildRowSlots(database);
    std::vector<AbacusSlot> result;
    result.reserve(row_slots.size());
    for (const RowSlot& row_slot : row_slots) {
        const SiteRow& row = database.rows[row_slot.row_index];
        if (reverse_x) {
            // Reflect lower-left coordinates: u = -(x + width).  This turns
            // a decreasing-x pass into the same append-only dynamic program.
            result.push_back({row_slot.row_index, -row_slot.right, -row_slot.left, -row.xEnd(), row.site_spacing,
                              true, 0.0, {}, {}});
        } else {
            result.push_back({row_slot.row_index, row_slot.left, row_slot.right, row.x_start, row.site_spacing,
                              false, 0.0, {}, {}});
        }
    }
    return result;
}

double projectClusterStart(const AbacusSlot& slot, const AbacusCluster& cluster) {
    const double upper = slot.right - cluster.total_width;
    const double continuous = clamp(cluster.q / std::max(cluster.total_weight, kEpsilon), slot.left, upper);
    return clamp(alignNearest(continuous, slot.grid_origin, slot.site_spacing), slot.left, upper);
}

AbacusCluster mergeAbacusClusters(const AbacusCluster& left, const AbacusCluster& right,
                                  const AbacusSlot& slot) {
    AbacusCluster merged;
    merged.first_entry = left.first_entry;
    merged.total_weight = left.total_weight + right.total_weight;
    merged.total_width = left.total_width + right.total_width;
    merged.q = left.q + right.q - right.total_weight * left.total_width;
    merged.x = projectClusterStart(slot, merged);
    return merged;
}

AbacusTrial tryAppendAbacusEntry(const AbacusSlot& slot, const AbacusEntry& entry) {
    AbacusTrial result;
    if (entry.width > slot.right - slot.left - slot.used_width + kEpsilon) return result;

    AbacusCluster trial_cluster;
    trial_cluster.first_entry = slot.entries.size();
    trial_cluster.total_weight = entry.weight;
    trial_cluster.total_width = entry.width;
    trial_cluster.q = entry.weight * entry.desired_x;
    trial_cluster.x = projectClusterStart(slot, trial_cluster);

    std::size_t replace_from = slot.clusters.size();
    while (replace_from > 0U &&
           slot.clusters[replace_from - 1U].x + slot.clusters[replace_from - 1U].total_width >
               trial_cluster.x + kEpsilon) {
        trial_cluster = mergeAbacusClusters(slot.clusters[replace_from - 1U], trial_cluster, slot);
        --replace_from;
    }

    result.feasible = true;
    result.replace_from = replace_from;
    result.cluster = trial_cluster;
    result.candidate_x = trial_cluster.x + trial_cluster.total_width - entry.width;
    return result;
}

void commitAbacusTrial(AbacusSlot& slot, const AbacusEntry& entry, const AbacusTrial& trial) {
    slot.entries.push_back(entry);
    slot.clusters.resize(trial.replace_from);
    slot.clusters.push_back(trial.cluster);
    slot.used_width += entry.width;
}

void materializeAbacusSlots(PlacementDatabase& database, const std::vector<AbacusSlot>& slots) {
    for (const AbacusSlot& slot : slots) {
        const SiteRow& row = database.rows[slot.row_index];
        for (std::size_t cluster_index = 0; cluster_index < slot.clusters.size(); ++cluster_index) {
            const AbacusCluster& cluster = slot.clusters[cluster_index];
            const std::size_t end = cluster_index + 1U < slot.clusters.size()
                ? slot.clusters[cluster_index + 1U].first_entry
                : slot.entries.size();
            double x = cluster.x;
            for (std::size_t entry_index = cluster.first_entry; entry_index < end; ++entry_index) {
                const AbacusEntry& entry = slot.entries[entry_index];
                Module& module = database.modules[entry.module_id];
                const double lower_left_x = slot.mirrored ? -x - entry.width : x;
                module.setLowerLeft({lower_left_x, row.bottom});
                if (row.orientation != Orientation::Unknown) module.orientation = row.orientation;
                x += entry.width;
            }
        }
    }
}

AbacusEntry makeAbacusEntry(const Module& module, ModuleId id, const Vec2& target, const SiteRow& row,
                            bool reverse_x) {
    const double width = roundedWidth(module, row);
    return {id, reverse_x ? -(target.x + width) : target.x, target.y, width, std::max(module.area(), 1.0)};
}

PlacementOutcome legalizeAbacusPass(PlacementDatabase& database, const LegalizationOptions& options,
                                    const std::vector<ModuleId>& standard_cells,
                                    const std::vector<Vec2>& targets, bool reverse_x) {
    std::vector<AbacusSlot> slots = buildAbacusSlots(database, reverse_x);
    const std::vector<std::size_t> sorted_rows = sortedRowIndices(database);
    std::vector<std::vector<std::size_t>> slots_by_row(database.rows.size());
    for (std::size_t slot_index = 0; slot_index < slots.size(); ++slot_index) {
        slots_by_row[slots[slot_index].row_index].push_back(slot_index);
    }

    std::vector<ModuleId> cells = standard_cells;
    std::sort(cells.begin(), cells.end(), [&](ModuleId left, ModuleId right) {
        const double left_x = targets[left].x;
        const double right_x = targets[right].x;
        if (std::abs(left_x - right_x) > kEpsilon) return reverse_x ? left_x > right_x : left_x < right_x;
        return left < right;
    });

    PlacementOutcome result;
    if (sorted_rows.empty()) {
        result.unplaced_standard_cells = cells.size();
        return result;
    }
    for (const ModuleId id : cells) {
        const Module& module = database.modules[id];
        const Vec2 target = targets[id];
        const std::size_t nearest = nearestSortedRow(database, sorted_rows, target.y + module.height * 0.5);
        std::size_t best_slot = slots.size();
        AbacusEntry best_entry;
        AbacusTrial best_trial;
        double best_cost = std::numeric_limits<double>::infinity();

        const auto considerRow = [&](std::size_t row_index) {
            const SiteRow& row = database.rows[row_index];
            const AbacusEntry entry = makeAbacusEntry(module, id, target, row, reverse_x);
            for (const std::size_t slot_index : slots_by_row[row_index]) {
                const AbacusTrial trial = tryAppendAbacusEntry(slots[slot_index], entry);
                if (!trial.feasible) continue;
                const double dx = trial.candidate_x - entry.desired_x;
                const double dy = row.bottom - entry.desired_y;
                const double cost = entry.weight * (dx * dx + dy * dy);
                if (cost < best_cost - kEpsilon ||
                    (std::abs(cost - best_cost) <= kEpsilon && slot_index < best_slot)) {
                    best_cost = cost;
                    best_slot = slot_index;
                    best_entry = entry;
                    best_trial = trial;
                }
            }
        };

        // Abacus first tries the nearest row, then expands upward and downward.
        // The vertical-only cost is a lower bound because the horizontal term is
        // non-negative, so a direction can stop once it cannot beat best_cost.
        considerRow(sorted_rows[nearest]);
        bool search_lower = true;
        bool search_upper = true;
        for (int distance = 1; distance <= options.candidate_rows && (search_lower || search_upper); ++distance) {
            const int lower_position = static_cast<int>(nearest) - distance;
            if (search_lower) {
                if (lower_position < 0) {
                    search_lower = false;
                } else {
                    const std::size_t row_index = sorted_rows[static_cast<std::size_t>(lower_position)];
                    const double dy = database.rows[row_index].bottom - target.y;
                    const double vertical_lower_bound = std::max(module.area(), 1.0) * dy * dy;
                    if (best_slot != slots.size() && vertical_lower_bound > best_cost + kEpsilon) {
                        search_lower = false;
                    } else {
                        considerRow(row_index);
                    }
                }
            }

            const int upper_position = static_cast<int>(nearest) + distance;
            if (search_upper) {
                if (upper_position >= static_cast<int>(sorted_rows.size())) {
                    search_upper = false;
                } else {
                    const std::size_t row_index = sorted_rows[static_cast<std::size_t>(upper_position)];
                    const double dy = database.rows[row_index].bottom - target.y;
                    const double vertical_lower_bound = std::max(module.area(), 1.0) * dy * dy;
                    if (best_slot != slots.size() && vertical_lower_bound > best_cost + kEpsilon) {
                        search_upper = false;
                    } else {
                        considerRow(row_index);
                    }
                }
            }
        }

        // If the local search found no capacity, retain the old legalizer's
        // full-row fallback so an otherwise placeable design is not abandoned.
        if (best_slot == slots.size()) {
            for (const std::size_t row_index : sorted_rows) considerRow(row_index);
        }
        if (best_slot == slots.size()) {
            ++result.unplaced_standard_cells;
            continue;
        }
        commitAbacusTrial(slots[best_slot], best_entry, best_trial);
        ++result.placed_standard_cells;
    }
    materializeAbacusSlots(database, slots);
    return result;
}

struct StandardCellState {
    Vec2 center;
    Orientation orientation = Orientation::Unknown;
};

std::vector<StandardCellState> captureStandardCellState(const PlacementDatabase& database,
                                                        const std::vector<ModuleId>& standard_cells) {
    std::vector<StandardCellState> result;
    result.reserve(standard_cells.size());
    for (const ModuleId id : standard_cells) {
        result.push_back({database.modules[id].center, database.modules[id].orientation});
    }
    return result;
}

void restoreStandardCellState(PlacementDatabase& database, const std::vector<ModuleId>& standard_cells,
                              const std::vector<StandardCellState>& state) {
    for (std::size_t index = 0; index < standard_cells.size(); ++index) {
        Module& module = database.modules[standard_cells[index]];
        module.center = state[index].center;
        module.orientation = state[index].orientation;
    }
}

bool preferAbacusOutcome(const PlacementOutcome& candidate, const MovementMetrics& candidate_metrics,
                         const PlacementOutcome& incumbent, const MovementMetrics& incumbent_metrics) {
    if (candidate.unplaced_standard_cells != incumbent.unplaced_standard_cells) {
        return candidate.unplaced_standard_cells < incumbent.unplaced_standard_cells;
    }
    if (candidate_metrics.total_displacement < incumbent_metrics.total_displacement - kEpsilon) return true;
    if (std::abs(candidate_metrics.total_displacement - incumbent_metrics.total_displacement) <= kEpsilon) {
        return candidate_metrics.weighted_squared_displacement <
            incumbent_metrics.weighted_squared_displacement - kEpsilon;
    }
    return false;
}

PlacementOutcome legalizeAbacus(PlacementDatabase& database, const LegalizationOptions& options,
                                const std::vector<ModuleId>& standard_cells,
                                const std::vector<Vec2>& targets, bool& reverse_pass_selected) {
    const std::vector<StandardCellState> initial_state = captureStandardCellState(database, standard_cells);
    PlacementOutcome forward = legalizeAbacusPass(database, options, standard_cells, targets, false);
    const MovementMetrics forward_metrics = calculateStandardCellMovement(database, standard_cells, targets);
    const std::vector<StandardCellState> forward_state = captureStandardCellState(database, standard_cells);

    reverse_pass_selected = false;
    if (!options.abacus_bidirectional || standard_cells.empty()) return forward;

    restoreStandardCellState(database, standard_cells, initial_state);
    const PlacementOutcome reverse = legalizeAbacusPass(database, options, standard_cells, targets, true);
    const MovementMetrics reverse_metrics = calculateStandardCellMovement(database, standard_cells, targets);
    if (preferAbacusOutcome(reverse, reverse_metrics, forward, forward_metrics)) {
        reverse_pass_selected = true;
        return reverse;
    }
    restoreStandardCellState(database, standard_cells, forward_state);
    return forward;
}

}  // namespace

std::string toString(LegalizationStrategy strategy) {
    switch (strategy) {
        case LegalizationStrategy::Abacus: return "abacus";
        case LegalizationStrategy::GreedyIsotonic: return "greedy";
    }
    return "abacus";
}

LegalizationStrategy parseLegalizationStrategy(const std::string& text) {
    std::string normalized = text;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    if (normalized == "abacus") return LegalizationStrategy::Abacus;
    if (normalized == "greedy" || normalized == "legacy" || normalized == "isotonic") {
        return LegalizationStrategy::GreedyIsotonic;
    }
    throw std::invalid_argument("Unknown legalization strategy: " + text);
}

LegalityReport Legalizer::legalize(PlacementDatabase& database, const LegalizationOptions& options) const {
    validateLegalizationOptions(options);
    const auto started = std::chrono::steady_clock::now();
    legalizeMacros(database, options);

    const std::vector<ModuleId> standard_cells = standardCells(database);
    // Targets are captured after macro handling and before either standard-cell
    // pass.  They remain immutable while Abacus repeatedly moves row members.
    const std::vector<Vec2> targets = lowerLeftTargets(database);
    bool reverse_pass_selected = false;
    PlacementOutcome outcome;
    if (options.strategy == LegalizationStrategy::Abacus) {
        outcome = legalizeAbacus(database, options, standard_cells, targets, reverse_pass_selected);
    } else {
        outcome = legalizeGreedyIsotonic(database, options, standard_cells, targets);
    }

    LegalityReport checked = check(database, options.epsilon);
    const MovementMetrics movement = calculateStandardCellMovement(database, standard_cells, targets);
    checked.strategy = options.strategy;
    checked.abacus_reverse_pass_selected = reverse_pass_selected;
    checked.placed_standard_cells = outcome.placed_standard_cells;
    checked.unplaced_standard_cells = outcome.unplaced_standard_cells;
    checked.standard_cell_total_displacement = movement.total_displacement;
    checked.standard_cell_total_squared_displacement = movement.total_squared_displacement;
    checked.standard_cell_weighted_squared_displacement = movement.weighted_squared_displacement;
    checked.standard_cell_maximum_displacement = movement.maximum_displacement;
    checked.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    checked.legal = checked.legal && checked.unplaced_standard_cells == 0U;
    return checked;
}

LegalityReport Legalizer::check(const PlacementDatabase& database, double epsilon) const {
    validateLegalityTolerance(epsilon);
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
        if (sorted_rows.empty()) {
            ++result.off_row_modules;
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
