#include "myplacement/placement/DetailedPlacer.hpp"

#include "CudaDetailedPlacementBackend.hpp"
#include "myplacement/metrics/Metrics.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace myplacement {
namespace {

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

bool isStandardCell(const Module& module) {
    return !module.is_fixed && !module.is_macro;
}

bool isKnownDetailedPlacementMethod(DetailedPlacementMethod method) {
    switch (method) {
        case DetailedPlacementMethod::None:
        case DetailedPlacementMethod::AdjacentSwap:
        case DetailedPlacementMethod::WindowReorder: return true;
    }
    return false;
}

bool isKnownDetailedPlacementBackend(DetailedPlacementBackend backend) {
    switch (backend) {
        case DetailedPlacementBackend::Cpu:
        case DetailedPlacementBackend::Cuda:
        case DetailedPlacementBackend::Auto: return true;
    }
    return false;
}

void validateDetailedPlacementOptions(const DetailedPlacementOptions& options) {
    if (!isKnownDetailedPlacementMethod(options.method) || !isKnownDetailedPlacementBackend(options.compute_backend) ||
        !std::isfinite(options.improvement_epsilon) || options.improvement_epsilon < 0.0) {
        throw std::invalid_argument("Detailed placement options are outside their valid range.");
    }
    if (options.method == DetailedPlacementMethod::None) return;
    if (options.passes <= 0 || options.maximum_net_degree < 2U) {
        throw std::invalid_argument("Detailed placement options are outside their valid range.");
    }
    const int requested_window =
        options.method == DetailedPlacementMethod::AdjacentSwap ? 2 : options.window_size;
    if (requested_window < 2 || requested_window > 6) {
        throw std::invalid_argument("Detailed placement window size must be between 2 and 6.");
    }
    if (options.compute_backend != DetailedPlacementBackend::Cpu &&
        (options.cuda_device < 1 || options.cuda_device > 4 || options.maximum_cuda_memory_bytes == 0U)) {
        throw std::invalid_argument("Detailed CUDA placement requires a device in [1, 4] and a non-zero memory budget.");
    }
}

double alignUp(double value, double origin, double step) {
    if (step <= kEpsilon) return value;
    return origin + std::ceil((value - origin - kEpsilon) / step) * step;
}

double roundedWidth(const Module& module, const SiteRow& row) {
    return alignUp(module.width, 0.0, row.site_spacing);
}

struct RowSegment {
    std::size_t row_index = 0;
    std::vector<ModuleId> modules;
};

std::vector<std::size_t> sortedRows(const PlacementDatabase& database) {
    std::vector<std::size_t> result(database.rows.size());
    for (std::size_t index = 0; index < result.size(); ++index) result[index] = index;
    std::sort(result.begin(), result.end(), [&](std::size_t left, std::size_t right) {
        return database.rows[left].bottom < database.rows[right].bottom;
    });
    return result;
}

std::size_t nearestRow(const PlacementDatabase& database, const std::vector<std::size_t>& sorted_rows,
                       double coordinate) {
    const auto iterator = std::lower_bound(sorted_rows.begin(), sorted_rows.end(), coordinate,
        [&](std::size_t row_index, double value) {
            const SiteRow& row = database.rows[row_index];
            return row.bottom + row.height * 0.5 < value;
        });
    if (iterator == sorted_rows.begin()) return *iterator;
    if (iterator == sorted_rows.end()) return sorted_rows.back();
    const std::size_t right = *iterator;
    const std::size_t left = *(iterator - 1);
    const double left_distance = std::abs(database.rows[left].bottom + database.rows[left].height * 0.5 - coordinate);
    const double right_distance =
        std::abs(database.rows[right].bottom + database.rows[right].height * 0.5 - coordinate);
    return left_distance <= right_distance ? left : right;
}

std::vector<RowSegment> buildRowSegments(const PlacementDatabase& database, double epsilon) {
    if (database.rows.empty()) return {};
    const std::vector<std::size_t> sorted_rows = sortedRows(database);
    std::vector<std::vector<ModuleId>> cells_by_row(database.rows.size());
    for (ModuleId id = 0; id < database.modules.size(); ++id) {
        const Module& module = database.modules[id];
        if (!isStandardCell(module)) continue;
        const std::size_t row_index = nearestRow(database, sorted_rows, module.center.y);
        const SiteRow& row = database.rows[row_index];
        if (std::abs(module.lowerLeft().y - row.bottom) > epsilon) continue;
        cells_by_row[row_index].push_back(id);
    }

    std::vector<RowSegment> segments;
    for (std::size_t row_index = 0; row_index < cells_by_row.size(); ++row_index) {
        std::vector<ModuleId>& cells = cells_by_row[row_index];
        if (cells.size() < 2U) continue;
        const SiteRow& row = database.rows[row_index];
        std::sort(cells.begin(), cells.end(), [&](ModuleId left, ModuleId right) {
            return database.modules[left].lowerLeft().x < database.modules[right].lowerLeft().x;
        });

        RowSegment current;
        current.row_index = row_index;
        for (const ModuleId id : cells) {
            if (!current.modules.empty()) {
                const Module& previous = database.modules[current.modules.back()];
                const double expected_left = previous.lowerLeft().x + roundedWidth(previous, row);
                const double actual_left = database.modules[id].lowerLeft().x;
                if (std::abs(actual_left - expected_left) > epsilon) {
                    if (current.modules.size() >= 2U) segments.push_back(std::move(current));
                    current = {};
                    current.row_index = row_index;
                }
            }
            current.modules.push_back(id);
        }
        if (current.modules.size() >= 2U) segments.push_back(std::move(current));
    }
    return segments;
}

Vec2 pinPositionWithOverrides(const PlacementDatabase& database, PinId pin_id,
                              const std::vector<ModuleId>& window_modules,
                              const std::vector<Vec2>& window_centers) {
    const Pin& pin = database.pins[pin_id];
    Vec2 center = database.modules[pin.module].center;
    for (std::size_t index = 0; index < window_modules.size(); ++index) {
        if (window_modules[index] == pin.module) {
            center = window_centers[index];
            break;
        }
    }
    return center + transformOffset(pin.offset, database.modules[pin.module].orientation);
}

double netHpwlWithOverrides(const PlacementDatabase& database, const Net& net,
                            const std::vector<ModuleId>& window_modules,
                            const std::vector<Vec2>& window_centers) {
    double minimum_x = std::numeric_limits<double>::infinity();
    double maximum_x = -std::numeric_limits<double>::infinity();
    double minimum_y = std::numeric_limits<double>::infinity();
    double maximum_y = -std::numeric_limits<double>::infinity();
    for (const PinId pin_id : net.pins) {
        const Vec2 position = pinPositionWithOverrides(database, pin_id, window_modules, window_centers);
        minimum_x = std::min(minimum_x, position.x);
        maximum_x = std::max(maximum_x, position.x);
        minimum_y = std::min(minimum_y, position.y);
        maximum_y = std::max(maximum_y, position.y);
    }
    return net.weight * ((maximum_x - minimum_x) + (maximum_y - minimum_y));
}

struct WindowResult {
    bool improved = false;
    std::vector<std::size_t> best_order;
    std::size_t evaluated_permutations = 0;
};

WindowResult optimizeWindow(PlacementDatabase& database, std::size_t row_index,
                            const std::vector<ModuleId>& modules, const DetailedPlacementOptions& options) {
    const SiteRow& row = database.rows[row_index];
    std::vector<NetId> affected_nets;
    for (const ModuleId id : modules) {
        for (const PinId pin_id : database.modules[id].pins) affected_nets.push_back(database.pins[pin_id].net);
    }
    std::sort(affected_nets.begin(), affected_nets.end());
    affected_nets.erase(std::unique(affected_nets.begin(), affected_nets.end()), affected_nets.end());
    if (affected_nets.empty()) return {};
    for (const NetId net_id : affected_nets) {
        if (database.nets[net_id].pins.size() > options.maximum_net_degree) return {};
    }

    std::vector<Vec2> original_centers;
    original_centers.reserve(modules.size());
    for (const ModuleId id : modules) original_centers.push_back(database.modules[id].center);
    double baseline = 0.0;
    for (const NetId net_id : affected_nets) {
        baseline += netHpwlWithOverrides(database, database.nets[net_id], modules, original_centers);
    }

    const double start_x = database.modules[modules.front()].lowerLeft().x;
    std::vector<std::size_t> order(modules.size());
    for (std::size_t index = 0; index < order.size(); ++index) order[index] = index;
    std::vector<Vec2> candidate_centers(modules.size());
    double best_value = baseline;
    WindowResult result;
    do {
        double cursor_x = start_x;
        for (const std::size_t member_index : order) {
            const Module& module = database.modules[modules[member_index]];
            candidate_centers[member_index] = {cursor_x + module.width * 0.5,
                                                row.bottom + module.height * 0.5};
            cursor_x += roundedWidth(module, row);
        }
        double value = 0.0;
        for (const NetId net_id : affected_nets) {
            value += netHpwlWithOverrides(database, database.nets[net_id], modules, candidate_centers);
        }
        ++result.evaluated_permutations;
        if (value < best_value - options.improvement_epsilon) {
            best_value = value;
            result.best_order = order;
            result.improved = true;
        }
    } while (std::next_permutation(order.begin(), order.end()));
    return result;
}

void applyOrder(PlacementDatabase& database, std::size_t row_index, const std::vector<ModuleId>& modules,
                const std::vector<std::size_t>& order) {
    const SiteRow& row = database.rows[row_index];
    double cursor_x = database.modules[modules.front()].lowerLeft().x;
    for (const std::size_t member_index : order) {
        Module& module = database.modules[modules[member_index]];
        module.setLowerLeft({cursor_x, row.bottom});
        if (row.orientation != Orientation::Unknown) module.orientation = row.orientation;
        cursor_x += roundedWidth(module, row);
    }
}

}  // namespace

std::string toString(DetailedPlacementMethod method) {
    switch (method) {
        case DetailedPlacementMethod::None: return "none";
        case DetailedPlacementMethod::AdjacentSwap: return "swap";
        case DetailedPlacementMethod::WindowReorder: return "window";
    }
    return "none";
}

DetailedPlacementMethod parseDetailedPlacementMethod(const std::string& value) {
    const std::string normalized = lowercase(value);
    if (normalized == "none" || normalized == "off") return DetailedPlacementMethod::None;
    if (normalized == "swap" || normalized == "adjacent" || normalized == "adjacent-swap") {
        return DetailedPlacementMethod::AdjacentSwap;
    }
    if (normalized == "window" || normalized == "reorder" || normalized == "window-reorder") {
        return DetailedPlacementMethod::WindowReorder;
    }
    throw std::invalid_argument("Unknown detailed placement method: " + value + ". Use none, swap, or window.");
}

std::string toString(DetailedPlacementBackend backend) {
    switch (backend) {
        case DetailedPlacementBackend::Cpu: return "cpu";
        case DetailedPlacementBackend::Cuda: return "cuda";
        case DetailedPlacementBackend::Auto: return "auto";
    }
    return "cpu";
}

DetailedPlacementBackend parseDetailedPlacementBackend(const std::string& value) {
    const std::string normalized = lowercase(value);
    if (normalized == "cpu") return DetailedPlacementBackend::Cpu;
    if (normalized == "cuda" || normalized == "gpu") return DetailedPlacementBackend::Cuda;
    if (normalized == "auto") return DetailedPlacementBackend::Auto;
    throw std::invalid_argument("Unknown detailed placement backend: " + value + ". Use cpu, cuda, or auto.");
}

DetailedPlacementResult DetailedPlacer::run(PlacementDatabase& database,
                                            const DetailedPlacementOptions& options) const {
    validateDetailedPlacementOptions(options);
    DetailedPlacementResult result;
    result.hpwl_before = calculateHpwl(database).hpwl;
    if (options.method == DetailedPlacementMethod::None || database.rows.empty()) {
        result.hpwl_after = result.hpwl_before;
        return result;
    }
    const int requested_window = options.method == DetailedPlacementMethod::AdjacentSwap ? 2 : options.window_size;

    if (options.compute_backend != DetailedPlacementBackend::Cpu) {
        std::string reason;
        DetailedPlacementResult cuda_result;
        if (detail::tryRunCudaDetailedPlacement(database, options, cuda_result, reason)) return cuda_result;
        if (options.compute_backend == DetailedPlacementBackend::Cuda) {
            throw std::invalid_argument("CUDA detailed-placement backend is unavailable: " + reason);
        }
    }

    const auto started = std::chrono::steady_clock::now();
    for (int pass = 0; pass < options.passes; ++pass) {
        const std::vector<RowSegment> segments = buildRowSegments(database, 1e-6);
        const std::size_t phase =
            requested_window > 2 ? static_cast<std::size_t>((pass % 2) * (requested_window / 2)) : 0U;
        for (const RowSegment& segment : segments) {
            for (std::size_t start = phase; start + 1U < segment.modules.size();) {
                const std::size_t count = std::min<std::size_t>(static_cast<std::size_t>(requested_window),
                                                                 segment.modules.size() - start);
                if (count < 2U) break;
                std::vector<ModuleId> window(segment.modules.begin() + static_cast<std::ptrdiff_t>(start),
                                             segment.modules.begin() + static_cast<std::ptrdiff_t>(start + count));
                WindowResult window_result = optimizeWindow(database, segment.row_index, window, options);
                ++result.evaluated_windows;
                result.evaluated_permutations += window_result.evaluated_permutations;
                if (window_result.improved) {
                    applyOrder(database, segment.row_index, window, window_result.best_order);
                    ++result.accepted_operations;
                }
                start += static_cast<std::size_t>(requested_window);
            }
        }
        result.completed_passes = pass + 1;
    }
    result.hpwl_after = calculateHpwl(database).hpwl;
    result.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return result;
}

}  // namespace myplacement
