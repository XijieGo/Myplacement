#include "TestAssertions.hpp"

#include "myplacement/placement/Legalizer.hpp"

#include <limits>

namespace myplacement::test {
namespace {

PlacementDatabase makeOverlappingSingleRowDatabase() {
    PlacementDatabase database;
    SiteRow row;
    row.bottom = 0.0;
    row.height = 1.0;
    row.site_width = 1.0;
    row.site_spacing = 1.0;
    row.x_start = 0.0;
    row.site_count = 30U;
    row.orientation = Orientation::North;
    database.rows.push_back(row);

    Module first;
    first.name = "first";
    first.width = 10.0;
    first.height = 1.0;
    first.setLowerLeft({6.0, 0.0});
    database.addModule(first);

    Module second;
    second.name = "second";
    second.width = 10.0;
    second.height = 1.0;
    second.setLowerLeft({8.0, 0.0});
    database.addModule(second);

    database.refreshDerivedData();
    return database;
}

}  // namespace

void runLegalizerTests() {
    PlacementDatabase database = makeOverlappingSingleRowDatabase();
    LegalizationOptions options;
    options.strategy = LegalizationStrategy::Abacus;
    options.abacus_bidirectional = false;
    const LegalityReport report = Legalizer().legalize(database, options);

    expect(report.legal, "Abacus did not legalize an overlapping single-row placement.");
    expectNear(database.modules[0].lowerLeft().x, 2.0, 1e-9,
               "Abacus did not re-center the first cell when its cluster collapsed.");
    expectNear(database.modules[1].lowerLeft().x, 12.0, 1e-9,
               "Abacus did not materialize the collapsed cluster in global-x order.");
    expectNear(report.standard_cell_total_displacement, 8.0, 1e-9,
               "Abacus movement report does not match the known two-cell solution.");
    expectNear(report.standard_cell_weighted_squared_displacement, 320.0, 1e-9,
               "Abacus weighted quadratic movement does not match its cluster objective.");
    expect(parseLegalizationStrategy("legacy") == LegalizationStrategy::GreedyIsotonic,
           "Legacy legalizer alias did not select the greedy baseline.");
    expect(parseLegalizationStrategy("abacus") == LegalizationStrategy::Abacus,
           "Abacus legalizer mode was not parsed.");

    const auto expect_invalid_options = [](const LegalizationOptions& invalid_options,
                                           const char* failure_message) {
        bool rejected = false;
        try {
            PlacementDatabase invalid_database = makeOverlappingSingleRowDatabase();
            Legalizer().legalize(invalid_database, invalid_options);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected, failure_message);
    };

    LegalizationOptions invalid_options;
    invalid_options.candidate_rows = 0;
    expect_invalid_options(invalid_options, "Legalizer accepted zero candidate rows.");
    invalid_options = {};
    invalid_options.macro_search_radius = -1;
    expect_invalid_options(invalid_options, "Legalizer accepted a negative macro search radius.");
    invalid_options = {};
    invalid_options.epsilon = std::numeric_limits<double>::quiet_NaN();
    expect_invalid_options(invalid_options, "Legalizer accepted a non-finite legality tolerance.");
    invalid_options = {};
    invalid_options.strategy = static_cast<LegalizationStrategy>(99);
    expect_invalid_options(invalid_options, "Legalizer accepted an unknown strategy value.");

    bool rejected_invalid_tolerance = false;
    try {
        Legalizer().check(database, -1.0);
    } catch (const std::invalid_argument&) {
        rejected_invalid_tolerance = true;
    }
    expect(rejected_invalid_tolerance, "Legality checker accepted a negative tolerance.");
}

}  // namespace myplacement::test
