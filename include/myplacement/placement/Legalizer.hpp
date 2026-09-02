#pragma once

#include <cstddef>
#include <string>

#include "myplacement/model/PlacementDatabase.hpp"

namespace myplacement {

// Abacus is the production default.  GreedyIsotonic is retained as a
// reproducible baseline for QoR experiments.
enum class LegalizationStrategy { Abacus, GreedyIsotonic };

std::string toString(LegalizationStrategy strategy);
LegalizationStrategy parseLegalizationStrategy(const std::string& text);

struct LegalizationOptions {
    LegalizationStrategy strategy = LegalizationStrategy::Abacus;
    int candidate_rows = 8;
    int macro_search_radius = 80;
    // The Abacus paper recommends evaluating both x-sort directions because
    // their legal placements can differ slightly.  This trades roughly 2x
    // legalization work for a deterministic, movement-based QoR choice.
    bool abacus_bidirectional = true;
    double epsilon = 1e-6;
};

struct LegalityReport {
    bool legal = false;
    std::size_t placed_standard_cells = 0;
    std::size_t unplaced_standard_cells = 0;
    std::size_t overlap_pairs = 0;
    std::size_t out_of_core_modules = 0;
    std::size_t off_row_modules = 0;
    LegalizationStrategy strategy = LegalizationStrategy::Abacus;
    bool abacus_reverse_pass_selected = false;
    // All displacement metrics refer to standard cells and compare the
    // legalized lower-left positions with the incoming global placement.
    double standard_cell_total_displacement = 0.0;
    double standard_cell_total_squared_displacement = 0.0;
    double standard_cell_weighted_squared_displacement = 0.0;
    double standard_cell_maximum_displacement = 0.0;
    double elapsed_seconds = 0.0;
};

class Legalizer {
public:
    LegalityReport legalize(PlacementDatabase& database, const LegalizationOptions& options) const;
    LegalityReport check(const PlacementDatabase& database, double epsilon = 1e-6) const;
};

}  // namespace myplacement
