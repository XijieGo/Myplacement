#pragma once

#include <cstddef>

#include "myplacement/model/PlacementDatabase.hpp"

namespace myplacement {

struct LegalizationOptions {
    int candidate_rows = 8;
    int macro_search_radius = 80;
    double epsilon = 1e-6;
};

struct LegalityReport {
    bool legal = false;
    std::size_t placed_standard_cells = 0;
    std::size_t unplaced_standard_cells = 0;
    std::size_t overlap_pairs = 0;
    std::size_t out_of_core_modules = 0;
    std::size_t off_row_modules = 0;
};

class Legalizer {
public:
    LegalityReport legalize(PlacementDatabase& database, const LegalizationOptions& options) const;
    LegalityReport check(const PlacementDatabase& database, double epsilon = 1e-6) const;
};

}  // namespace myplacement
