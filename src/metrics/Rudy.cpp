#include "myplacement/metrics/Rudy.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace myplacement {
namespace {

struct NetBox {
    double left = 0.0;
    double right = 0.0;
    double bottom = 0.0;
    double top = 0.0;
    double raw_left = 0.0;
    double raw_right = 0.0;
    double raw_bottom = 0.0;
    double raw_top = 0.0;
    double x_span = 0.0;
    double y_span = 0.0;
    bool x_is_physical = true;
    bool y_is_physical = true;
};

struct PenaltyValue {
    double value = 0.0;
    double derivative = 0.0;
};

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

bool isKnownPenaltyModel(RudyPenaltyModel model) {
    switch (model) {
        case RudyPenaltyModel::Disabled:
        case RudyPenaltyModel::HingeL2:
        case RudyPenaltyModel::SoftplusL2:
        case RudyPenaltyModel::HingeL4: return true;
    }
    return false;
}

void validateRudyOptions(const RudyOptions& options) {
    if (options.columns <= 0 || options.rows <= 0 || !std::isfinite(options.minimum_span_in_bins) ||
        options.minimum_span_in_bins <= 0.0 || !std::isfinite(options.capacity_factor) ||
        options.capacity_factor <= 0.0 || !std::isfinite(options.softplus_temperature) ||
        options.softplus_temperature <= 0.0 || !isKnownPenaltyModel(options.penalty_model)) {
        throw std::invalid_argument("RUDY evaluation requires a positive finite grid configuration and penalty model.");
    }
}

void validateRudyCapacity(const RudyCapacity& capacity) {
    const bool no_capacity_requested = capacity.horizontal == 0.0 && capacity.vertical == 0.0;
    if (!no_capacity_requested && !capacity.valid()) {
        throw std::invalid_argument("RUDY capacity must be omitted or contain two finite positive values.");
    }
}

double stableSoftplus(double value) {
    if (value > 30.0) return value;
    if (value < -30.0) return std::exp(value);
    return std::log1p(std::exp(value));
}

double stableSigmoid(double value) {
    if (value >= 0.0) {
        const double inverse = std::exp(-value);
        return 1.0 / (1.0 + inverse);
    }
    const double exponent = std::exp(value);
    return exponent / (1.0 + exponent);
}

PenaltyValue evaluatePenalty(double utilization, const RudyOptions& options) {
    const double overflow = std::max(0.0, utilization - 1.0);
    switch (options.penalty_model) {
        case RudyPenaltyModel::Disabled: return {};
        case RudyPenaltyModel::HingeL2: return {0.5 * overflow * overflow, overflow};
        case RudyPenaltyModel::HingeL4: {
            const double square = overflow * overflow;
            return {0.25 * square * square, overflow * square};
        }
        case RudyPenaltyModel::SoftplusL2: {
            const double temperature = std::max(options.softplus_temperature, kEpsilon);
            const double scaled = (utilization - 1.0) / temperature;
            const double softened = temperature * stableSoftplus(scaled);
            return {0.5 * softened * softened, softened * stableSigmoid(scaled)};
        }
    }
    return {};
}

std::size_t flatIndex(int column, int row, int columns) {
    return static_cast<std::size_t>(row * columns + column);
}

int startBin(double coordinate, double lower, double bin_length, int count) {
    return std::max(0, std::min(count - 1, static_cast<int>(std::floor((coordinate - lower) / bin_length))));
}

int endBin(double coordinate, double lower, double bin_length, int count) {
    return std::max(0, std::min(count - 1,
                                static_cast<int>(std::ceil((coordinate - lower) / bin_length) - 1.0)));
}

NetBox calculateNetBox(const PlacementDatabase& database, const Net& net, double minimum_x_span,
                       double minimum_y_span) {
    NetBox result;
    result.left = std::numeric_limits<double>::infinity();
    result.right = -std::numeric_limits<double>::infinity();
    result.bottom = std::numeric_limits<double>::infinity();
    result.top = -std::numeric_limits<double>::infinity();
    for (const PinId pin_id : net.pins) {
        const Vec2 position = database.pinPosition(pin_id);
        result.left = std::min(result.left, position.x);
        result.right = std::max(result.right, position.x);
        result.bottom = std::min(result.bottom, position.y);
        result.top = std::max(result.top, position.y);
    }
    const double raw_x_span = result.right - result.left;
    const double raw_y_span = result.top - result.bottom;
    result.raw_left = result.left;
    result.raw_right = result.right;
    result.raw_bottom = result.bottom;
    result.raw_top = result.top;
    if (raw_x_span < minimum_x_span) {
        const double center = (result.left + result.right) * 0.5;
        result.left = center - minimum_x_span * 0.5;
        result.right = center + minimum_x_span * 0.5;
        result.x_is_physical = false;
    }
    if (raw_y_span < minimum_y_span) {
        const double center = (result.bottom + result.top) * 0.5;
        result.bottom = center - minimum_y_span * 0.5;
        result.top = center + minimum_y_span * 0.5;
        result.y_is_physical = false;
    }
    result.x_span = std::max(result.right - result.left, kEpsilon);
    result.y_span = std::max(result.top - result.bottom, kEpsilon);
    return result;
}

double overlapLength(double lower_a, double upper_a, double lower_b, double upper_b) {
    return std::max(0.0, std::min(upper_a, upper_b) - std::max(lower_a, lower_b));
}

bool approximatelyEqual(double left, double right) {
    return std::abs(left - right) <= 1e-10 * std::max({1.0, std::abs(left), std::abs(right)});
}

double upperOverlapDerivative(double upper, double bin_lower, double bin_upper, double overlap) {
    if (upper > bin_lower + kEpsilon && upper < bin_upper - kEpsilon && overlap > kEpsilon) {
        return 1.0;
    }
    // At a grid boundary the rectangular-overlap map is non-smooth. Split a
    // valid Clarke subgradient over its two touching bins rather than
    // returning zero and pinning a net exactly on a RUDY grid line.
    if (approximatelyEqual(upper, bin_lower) || approximatelyEqual(upper, bin_upper)) return 0.5;
    return 0.0;
}

double lowerOverlapDerivative(double lower, double bin_lower, double bin_upper, double overlap) {
    if (lower > bin_lower + kEpsilon && lower < bin_upper - kEpsilon && overlap > kEpsilon) {
        return -1.0;
    }
    if (approximatelyEqual(lower, bin_lower) || approximatelyEqual(lower, bin_upper)) return -0.5;
    return 0.0;
}

bool liesOnInteriorGridBoundary(double coordinate, double lower, double bin_length, int count) {
    const double relative = (coordinate - lower) / bin_length;
    const double nearest = std::round(relative);
    return nearest > 0.0 && nearest < static_cast<double>(count) && approximatelyEqual(relative, nearest);
}

}  // namespace

std::string toString(RudyPenaltyModel model) {
    switch (model) {
        case RudyPenaltyModel::Disabled: return "disabled";
        case RudyPenaltyModel::HingeL2: return "rudy_hinge_l2";
        case RudyPenaltyModel::SoftplusL2: return "rudy_softplus_l2";
        case RudyPenaltyModel::HingeL4: return "rudy_hinge_l4";
    }
    return "disabled";
}

RudyPenaltyModel parseRudyPenaltyModel(const std::string& value) {
    const std::string normalized = lower(value);
    if (normalized == "disabled" || normalized == "none" || normalized == "off") {
        return RudyPenaltyModel::Disabled;
    }
    if (normalized == "rudy_hinge_l2" || normalized == "hinge_l2" || normalized == "rudy") {
        return RudyPenaltyModel::HingeL2;
    }
    if (normalized == "rudy_softplus_l2" || normalized == "softplus_l2") {
        return RudyPenaltyModel::SoftplusL2;
    }
    if (normalized == "rudy_hinge_l4" || normalized == "hinge_l4") {
        return RudyPenaltyModel::HingeL4;
    }
    throw std::invalid_argument("Unknown RUDY penalty model: " + value +
                                ". Use disabled, rudy_hinge_l2, rudy_softplus_l2, or rudy_hinge_l4.");
}

RudyCapacity calibrateRudyCapacity(const RudyMetrics& metrics, double capacity_factor) {
    if (!std::isfinite(capacity_factor) || capacity_factor <= 0.0) {
        throw std::invalid_argument("RUDY capacity factor must be finite and positive.");
    }
    if (!std::isfinite(metrics.horizontal_mean_demand) || !std::isfinite(metrics.vertical_mean_demand) ||
        metrics.horizontal_mean_demand < 0.0 || metrics.vertical_mean_demand < 0.0) {
        throw std::invalid_argument("RUDY mean demand must be finite and non-negative.");
    }
    return {std::max(kEpsilon, capacity_factor * metrics.horizontal_mean_demand),
            std::max(kEpsilon, capacity_factor * metrics.vertical_mean_demand)};
}

RudyEvaluation evaluateRudy(const PlacementDatabase& database, const RudyOptions& options,
                            const RudyCapacity& capacity, bool calculate_gradient) {
    validateRudyOptions(options);
    validateRudyCapacity(capacity);
    if (!database.core_region.valid() || database.core_region.area() <= kEpsilon) {
        throw std::invalid_argument("RUDY evaluation requires a valid core region.");
    }
    const std::size_t columns = static_cast<std::size_t>(options.columns);
    const std::size_t rows = static_cast<std::size_t>(options.rows);
    if (columns > kMaximumDensityBinCount / rows) {
        throw std::invalid_argument("RUDY grid exceeds the safe one-million-bin workspace limit.");
    }

    const std::size_t bin_count = columns * rows;
    const double bin_width = database.core_region.width() / static_cast<double>(options.columns);
    const double bin_height = database.core_region.height() / static_cast<double>(options.rows);
    const double minimum_x_span = options.minimum_span_in_bins * bin_width;
    const double minimum_y_span = options.minimum_span_in_bins * bin_height;
    std::vector<double> horizontal(bin_count, 0.0);
    std::vector<double> vertical(bin_count, 0.0);
    std::vector<NetBox> boxes(database.nets.size());
    std::vector<bool> counted(database.nets.size(), false);

    for (NetId net_id = 0; net_id < database.nets.size(); ++net_id) {
        const Net& net = database.nets[net_id];
        if (net.pins.size() < 2U || net.weight <= 0.0) continue;
        const NetBox box = calculateNetBox(database, net, minimum_x_span, minimum_y_span);
        boxes[net_id] = box;
        const Rect clipped{{std::max(box.left, database.core_region.ll.x),
                            std::max(box.bottom, database.core_region.ll.y)},
                           {std::min(box.right, database.core_region.ur.x),
                            std::min(box.top, database.core_region.ur.y)}};
        if (clipped.area() <= kEpsilon) continue;
        const int first_column = startBin(clipped.ll.x, database.core_region.ll.x, bin_width, options.columns);
        const int last_column = endBin(clipped.ur.x, database.core_region.ll.x, bin_width, options.columns);
        const int first_row = startBin(clipped.ll.y, database.core_region.ll.y, bin_height, options.rows);
        const int last_row = endBin(clipped.ur.y, database.core_region.ll.y, bin_height, options.rows);
        for (int row = first_row; row <= last_row; ++row) {
            const double bin_bottom = database.core_region.ll.y + static_cast<double>(row) * bin_height;
            const double bin_top = bin_bottom + bin_height;
            const double overlap_y = overlapLength(box.bottom, box.top, bin_bottom, bin_top);
            if (overlap_y <= kEpsilon) continue;
            for (int column = first_column; column <= last_column; ++column) {
                const double bin_left = database.core_region.ll.x + static_cast<double>(column) * bin_width;
                const double bin_right = bin_left + bin_width;
                const double overlap_x = overlapLength(box.left, box.right, bin_left, bin_right);
                const double overlap = overlap_x * overlap_y;
                if (overlap <= kEpsilon) continue;
                const std::size_t index = flatIndex(column, row, options.columns);
                horizontal[index] += net.weight * overlap / box.y_span;
                vertical[index] += net.weight * overlap / box.x_span;
            }
        }
        counted[net_id] = true;
    }

    RudyEvaluation result;
    result.metrics.horizontal_capacity = capacity.horizontal;
    result.metrics.vertical_capacity = capacity.vertical;
    for (const double value : horizontal) result.metrics.horizontal_total_demand += value;
    for (const double value : vertical) result.metrics.vertical_total_demand += value;
    result.metrics.horizontal_mean_demand = result.metrics.horizontal_total_demand / static_cast<double>(bin_count);
    result.metrics.vertical_mean_demand = result.metrics.vertical_total_demand / static_cast<double>(bin_count);
    result.metrics.counted_nets = static_cast<std::size_t>(std::count(counted.begin(), counted.end(), true));
    if (!capacity.valid()) return result;

    std::vector<double> horizontal_adjoint(bin_count, 0.0);
    std::vector<double> vertical_adjoint(bin_count, 0.0);
    std::vector<double> utilization_samples;
    utilization_samples.reserve(bin_count);
    for (std::size_t index = 0; index < bin_count; ++index) {
        const double horizontal_utilization = horizontal[index] / capacity.horizontal;
        const double vertical_utilization = vertical[index] / capacity.vertical;
        const PenaltyValue horizontal_penalty = evaluatePenalty(horizontal_utilization, options);
        const PenaltyValue vertical_penalty = evaluatePenalty(vertical_utilization, options);
        result.energy += (horizontal_penalty.value + vertical_penalty.value) / static_cast<double>(bin_count);
        horizontal_adjoint[index] = horizontal_penalty.derivative /
                                    (static_cast<double>(bin_count) * capacity.horizontal);
        vertical_adjoint[index] = vertical_penalty.derivative /
                                  (static_cast<double>(bin_count) * capacity.vertical);
        result.metrics.proxy_overflow +=
            std::max(0.0, horizontal_utilization - 1.0) + std::max(0.0, vertical_utilization - 1.0);
        const double utilization = std::max(horizontal_utilization, vertical_utilization);
        result.metrics.maximum_utilization = std::max(result.metrics.maximum_utilization, utilization);
        utilization_samples.push_back(utilization);
    }
    result.metrics.proxy_overflow /= static_cast<double>(bin_count);
    std::sort(utilization_samples.begin(), utilization_samples.end());
    const std::size_t p95_index = static_cast<std::size_t>(std::floor(
        0.95 * static_cast<double>(std::max<std::size_t>(1U, utilization_samples.size() - 1U))));
    result.metrics.p95_utilization = utilization_samples[p95_index];
    if (!calculate_gradient || options.penalty_model == RudyPenaltyModel::Disabled) return result;

    result.gradient.resize(database.modules.size());
    for (NetId net_id = 0; net_id < database.nets.size(); ++net_id) {
        if (!counted[net_id]) continue;
        const Net& net = database.nets[net_id];
        const NetBox& box = boxes[net_id];
        const Rect clipped{{std::max(box.left, database.core_region.ll.x),
                            std::max(box.bottom, database.core_region.ll.y)},
                           {std::min(box.right, database.core_region.ur.x),
                            std::min(box.top, database.core_region.ur.y)}};
        if (clipped.area() <= kEpsilon) continue;
        const int first_column = startBin(clipped.ll.x, database.core_region.ll.x, bin_width, options.columns);
        const int last_column = endBin(clipped.ur.x, database.core_region.ll.x, bin_width, options.columns);
        const int first_row = startBin(clipped.ll.y, database.core_region.ll.y, bin_height, options.rows);
        const int last_row = endBin(clipped.ur.y, database.core_region.ll.y, bin_height, options.rows);
        const int gradient_first_column =
            std::max(0, first_column - (liesOnInteriorGridBoundary(box.left, database.core_region.ll.x, bin_width,
                                                                   options.columns)
                                            ? 1
                                            : 0));
        const int gradient_last_column =
            std::min(options.columns - 1, last_column + (liesOnInteriorGridBoundary(box.right,
                                                                                      database.core_region.ll.x,
                                                                                      bin_width, options.columns)
                                                             ? 1
                                                             : 0));
        const int gradient_first_row =
            std::max(0, first_row - (liesOnInteriorGridBoundary(box.bottom, database.core_region.ll.y, bin_height,
                                                                options.rows)
                                         ? 1
                                         : 0));
        const int gradient_last_row =
            std::min(options.rows - 1, last_row + (liesOnInteriorGridBoundary(box.top, database.core_region.ll.y,
                                                                               bin_height, options.rows)
                                                       ? 1
                                                       : 0));
        double gradient_left = 0.0;
        double gradient_right = 0.0;
        double gradient_bottom = 0.0;
        double gradient_top = 0.0;
        for (int row = gradient_first_row; row <= gradient_last_row; ++row) {
            const double bin_bottom = database.core_region.ll.y + static_cast<double>(row) * bin_height;
            const double bin_top = bin_bottom + bin_height;
            const double overlap_y = overlapLength(box.bottom, box.top, bin_bottom, bin_top);
            const double bottom_derivative = lowerOverlapDerivative(box.bottom, bin_bottom, bin_top, overlap_y);
            const double top_derivative = upperOverlapDerivative(box.top, bin_bottom, bin_top, overlap_y);
            if (overlap_y <= kEpsilon && bottom_derivative == 0.0 && top_derivative == 0.0) continue;
            for (int column = gradient_first_column; column <= gradient_last_column; ++column) {
                const double bin_left = database.core_region.ll.x + static_cast<double>(column) * bin_width;
                const double bin_right = bin_left + bin_width;
                const double overlap_x = overlapLength(box.left, box.right, bin_left, bin_right);
                const double overlap = overlap_x * overlap_y;
                const double left_derivative = lowerOverlapDerivative(box.left, bin_left, bin_right, overlap_x);
                const double right_derivative = upperOverlapDerivative(box.right, bin_left, bin_right, overlap_x);
                if (overlap_x <= kEpsilon && left_derivative == 0.0 && right_derivative == 0.0) continue;
                const std::size_t index = flatIndex(column, row, options.columns);
                const double horizontal_adjoint_value = horizontal_adjoint[index];
                const double vertical_adjoint_value = vertical_adjoint[index];
                const double overlap_left = left_derivative * overlap_y;
                const double overlap_right = right_derivative * overlap_y;
                const double overlap_bottom = overlap_x * bottom_derivative;
                const double overlap_top = overlap_x * top_derivative;
                const double weight = net.weight;
                const double x_span_left_term =
                    box.x_is_physical ? overlap / (box.x_span * box.x_span) : 0.0;
                const double x_span_right_term =
                    box.x_is_physical ? -overlap / (box.x_span * box.x_span) : 0.0;
                const double y_span_bottom_term =
                    box.y_is_physical ? overlap / (box.y_span * box.y_span) : 0.0;
                const double y_span_top_term =
                    box.y_is_physical ? -overlap / (box.y_span * box.y_span) : 0.0;
                gradient_left += horizontal_adjoint_value * weight * overlap_left / box.y_span +
                                 vertical_adjoint_value * weight *
                                     (overlap_left / box.x_span + x_span_left_term);
                gradient_right += horizontal_adjoint_value * weight * overlap_right / box.y_span +
                                  vertical_adjoint_value * weight *
                                      (overlap_right / box.x_span + x_span_right_term);
                gradient_bottom += horizontal_adjoint_value * weight *
                                       (overlap_bottom / box.y_span + y_span_bottom_term) +
                                   vertical_adjoint_value * weight * overlap_bottom / box.x_span;
                gradient_top += horizontal_adjoint_value * weight *
                                    (overlap_top / box.y_span + y_span_top_term) +
                                vertical_adjoint_value * weight * overlap_top / box.x_span;
            }
        }

        std::size_t left_count = 0;
        std::size_t right_count = 0;
        std::size_t bottom_count = 0;
        std::size_t top_count = 0;
        for (const PinId pin_id : net.pins) {
            const Vec2 position = database.pinPosition(pin_id);
            if (approximatelyEqual(position.x, box.x_is_physical ? box.left : box.raw_left)) ++left_count;
            if (approximatelyEqual(position.x, box.x_is_physical ? box.right : box.raw_right)) ++right_count;
            if (approximatelyEqual(position.y, box.y_is_physical ? box.bottom : box.raw_bottom)) ++bottom_count;
            if (approximatelyEqual(position.y, box.y_is_physical ? box.top : box.raw_top)) ++top_count;
        }
        for (const PinId pin_id : net.pins) {
            const Pin& pin = database.pins[pin_id];
            if (database.modules[pin.module].is_fixed) continue;
            const Vec2 position = database.pinPosition(pin_id);
            Vec2& gradient = result.gradient[pin.module];
            if (box.x_is_physical) {
                if (left_count > 0U && approximatelyEqual(position.x, box.left)) {
                    gradient.x += gradient_left / static_cast<double>(left_count);
                }
                if (right_count > 0U && approximatelyEqual(position.x, box.right)) {
                    gradient.x += gradient_right / static_cast<double>(right_count);
                }
            } else {
                const double center_gradient = 0.5 * (gradient_left + gradient_right);
                if (left_count > 0U && approximatelyEqual(position.x, box.raw_left)) {
                    gradient.x += center_gradient / static_cast<double>(left_count);
                }
                if (right_count > 0U && approximatelyEqual(position.x, box.raw_right)) {
                    gradient.x += center_gradient / static_cast<double>(right_count);
                }
            }
            if (box.y_is_physical) {
                if (bottom_count > 0U && approximatelyEqual(position.y, box.bottom)) {
                    gradient.y += gradient_bottom / static_cast<double>(bottom_count);
                }
                if (top_count > 0U && approximatelyEqual(position.y, box.top)) {
                    gradient.y += gradient_top / static_cast<double>(top_count);
                }
            } else {
                const double center_gradient = 0.5 * (gradient_bottom + gradient_top);
                if (bottom_count > 0U && approximatelyEqual(position.y, box.raw_bottom)) {
                    gradient.y += center_gradient / static_cast<double>(bottom_count);
                }
                if (top_count > 0U && approximatelyEqual(position.y, box.raw_top)) {
                    gradient.y += center_gradient / static_cast<double>(top_count);
                }
            }
        }
    }
    return result;
}

}  // namespace myplacement
