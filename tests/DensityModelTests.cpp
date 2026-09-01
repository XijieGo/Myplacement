#include "TestAssertions.hpp"

#include "myplacement/metrics/Metrics.hpp"

#include <stdexcept>
#include <vector>

namespace myplacement::test {

void runDensityModelTests() {
    const Rect core{{0.0, 0.0}, {10.0, 10.0}};
    DensityMap map(core, 2, 2, 0.5);
    // Only the left half is legal placement area. The right half is a dark
    // obstacle, equivalent to an unavailable row interval in BookShelf SCL.
    map.setPlaceableRegions(std::vector<Rect>{{{0.0, 0.0}, {5.0, 10.0}}});
    map.addRectangle({{0.0, 0.0}, {2.0, 5.0}}, DensityLayer::Movable);  // area 10
    map.addRectangle({{0.0, 5.0}, {2.0, 10.0}}, DensityLayer::Macro);   // area 10
    map.addRectangle({{2.0, 0.0}, {5.0, 5.0}}, DensityLayer::Fixed);    // area 15
    map.addRectangle({{2.0, 5.0}, {5.0, 10.0}}, DensityLayer::Filler);  // area 15

    const DensityMetrics without_fillers = map.metrics(false);
    const DensityMetrics with_fillers = map.metrics(true);
    expectNear(without_fillers.dark_area, 50.0, 1e-12,
               "Site-row masking did not convert the non-row half into dark area.");
    expectNear(without_fillers.placeable_area, 50.0, 1e-12,
               "Site-row masking produced an incorrect legal placement area.");
    // Only standard cells and target-scaled macros define the course metric
    // denominator. Fixed terminals, filler, and dark area must not dilute it.
    expectNear(without_fillers.normalization_area, 15.0, 1e-12,
               "Course density denominator incorrectly includes fixed or dark area.");
    expectNear(with_fillers.normalization_area, 15.0, 1e-12,
               "Fillers incorrectly changed the course density denominator.");
    expectNear(without_fillers.total_charge_area, 47.5, 1e-12,
               "Scaled fixed, macro, and dark density charge is incorrect.");
    expectNear(with_fillers.total_charge_area, 55.0, 1e-12,
               "Target-scaled filler density charge is incorrect.");
    expectNear(without_fillers.total_overflow_area, 5.0, 1e-12,
               "Course density overflow area is incorrect.");
    expectNear(without_fillers.normalized_overflow, 1.0 / 3.0, 1e-12,
               "Course density overflow was normalized by the wrong area.");
    expectNear(map.at(1, 0).dark_area, 25.0, 1e-12,
               "The dark-area mask was not distributed to individual bins.");

    bool rejected_overlapping_rows = false;
    try {
        DensityMap invalid_map(core, 2, 2, 0.5);
        invalid_map.setPlaceableRegions(std::vector<Rect>{core, core});
    } catch (const std::invalid_argument&) {
        rejected_overlapping_rows = true;
    }
    expect(rejected_overlapping_rows, "Overlapping SiteRows were accepted as a density mask.");
}

}  // namespace myplacement::test
