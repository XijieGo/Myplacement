#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "myplacement/core/ResourceLimits.hpp"
#include "myplacement/model/PlacementDatabase.hpp"

namespace myplacement {

// RUDY is a routing-demand proxy, not a replacement for a technology-aware
// global router.  The names below deliberately use "proxy" instead of
// "route overflow" so reports cannot accidentally overclaim physical routing
// quality when the input only provides BookShelf placement data.
enum class RudyPenaltyModel { Disabled, HingeL2, SoftplusL2, HingeL4 };

std::string toString(RudyPenaltyModel model);
RudyPenaltyModel parseRudyPenaltyModel(const std::string& value);

struct RudyOptions {
    int columns = 64;
    int rows = 64;
    // A net whose span is smaller than this fraction of a bin is expanded only
    // for RUDY rasterization.  This avoids singular 1/span demand for vertical
    // or horizontal two-pin nets; only physical translation, not artificial
    // expansion or contraction, is differentiated on the expanded axis.
    double minimum_span_in_bins = 0.25;
    // The reference capacities are derived once from the first active
    // placement state: capacity = capacity_factor * mean directional demand.
    // They are then fixed for the rest of a run, so the optimizer cannot make
    // the target easier merely by changing the current placement.
    // A capacity equal to the activation-state directional mean is the
    // conservative, technology-agnostic default.  Larger or smaller values
    // remain explicit experiment controls, not inferred physical tracks.
    double capacity_factor = 1.00;
    // Temperature in normalized-utilization units for softplus_l2.
    double softplus_temperature = 0.10;
    RudyPenaltyModel penalty_model = RudyPenaltyModel::Disabled;
};

struct RudyCapacity {
    double horizontal = 0.0;
    double vertical = 0.0;

    [[nodiscard]] bool valid() const {
        return std::isfinite(horizontal) && std::isfinite(vertical) && horizontal > kEpsilon && vertical > kEpsilon;
    }
};

struct RudyMetrics {
    double horizontal_total_demand = 0.0;
    double vertical_total_demand = 0.0;
    double horizontal_mean_demand = 0.0;
    double vertical_mean_demand = 0.0;
    double horizontal_capacity = 0.0;
    double vertical_capacity = 0.0;
    // These are normalized RUDY-demand statistics.  They are intentionally
    // not named routing overflow because BookShelf has no track capacities.
    double proxy_overflow = 0.0;
    double maximum_utilization = 0.0;
    double p95_utilization = 0.0;
    std::size_t counted_nets = 0;
};

struct RudyEvaluation {
    RudyMetrics metrics;
    double energy = 0.0;
    // Indexed by PlacementDatabase::modules.  Fixed modules have zero
    // gradient, and callers can select movable modules as appropriate.
    std::vector<Vec2> gradient;
};

// Produces stable directional reference capacities from an unpenalized RUDY
// evaluation.  A caller normally invokes this once at routability activation.
RudyCapacity calibrateRudyCapacity(const RudyMetrics& metrics, double capacity_factor);

// Computes the exact rectangular-overlap RUDY map described by Spindler and
// Johannes.  With a valid fixed capacity, it also evaluates a hotspot penalty
// and its analytical subgradient with respect to movable module centers.
RudyEvaluation evaluateRudy(const PlacementDatabase& database, const RudyOptions& options,
                            const RudyCapacity& capacity = {}, bool calculate_gradient = false);

}  // namespace myplacement
