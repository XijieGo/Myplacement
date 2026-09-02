#include "TestAssertions.hpp"

#include "myplacement/metrics/Metrics.hpp"
#include "myplacement/placement/DetailedPlacer.hpp"
#include "myplacement/placement/Legalizer.hpp"

#include <cmath>
#include <limits>
#include <utility>

namespace myplacement::test {
namespace {

PlacementDatabase makeReorderingFixture() {
    PlacementDatabase database;
    database.rows.push_back({0.0, 10.0, 1.0, 1.0, 0.0, 40U, Orientation::North});
    const auto add_cell = [&](const char* name, double lower_left_x) {
        Module module;
        module.name = name;
        module.width = 5.0;
        module.height = 10.0;
        module.setLowerLeft({lower_left_x, 0.0});
        database.addModule(std::move(module));
    };

    // The legal order A,C,D,B requires a coordinated four-cell reorder.  No
    // single adjacent swap improves the two-net HPWL, whereas A,B,C,D does.
    add_cell("A", 0.0);
    add_cell("C", 5.0);
    add_cell("D", 10.0);
    add_cell("B", 15.0);
    database.refreshDerivedData();

    const auto add_two_pin_net = [&](const char* name, ModuleId first, ModuleId second) {
        const NetId net = database.addNet({name, 1.0, {}});
        database.addPin({first, net, {}, PinDirection::Bidirectional});
        database.addPin({second, net, {}, PinDirection::Bidirectional});
    };
    add_two_pin_net("AB", 0U, 3U);
    add_two_pin_net("CD", 1U, 2U);
    return database;
}

}  // namespace

void runDetailedPlacementTests() {
    PlacementDatabase swap_database = makeReorderingFixture();
    const double initial_hpwl = calculateHpwl(swap_database).hpwl;
    DetailedPlacementOptions swap_options;
    swap_options.method = DetailedPlacementMethod::AdjacentSwap;
    swap_options.passes = 2;
    const DetailedPlacementResult swap = DetailedPlacer().run(swap_database, swap_options);
    expectNear(swap.hpwl_before, initial_hpwl, 1e-12, "Detailed placement measured the wrong initial HPWL.");
    expectNear(swap.hpwl_after, initial_hpwl, 1e-12,
               "An adjacent swap escaped a strictly local optimum in the fixture.");
    expect(Legalizer().check(swap_database).legal, "Adjacent-swap detailed placement broke legality.");

    PlacementDatabase window_database = makeReorderingFixture();
    DetailedPlacementOptions window_options;
    window_options.method = DetailedPlacementMethod::WindowReorder;
    window_options.passes = 1;
    window_options.window_size = 4;
    const DetailedPlacementResult window = DetailedPlacer().run(window_database, window_options);
    expect(window.accepted_operations == 1U,
           "Window reordering did not accept the coordinated improvement in the fixture.");
    expect(window.evaluated_permutations == 24U,
           "Four-cell window reordering did not enumerate every candidate ordering.");
    expect(window.hpwl_after < window.hpwl_before - 1e-9,
           "Window reordering did not improve final legal HPWL.");
    expectNear(window.hpwl_after, 10.0, 1e-12,
               "Window reordering produced the wrong best ordering for the fixture.");
    expect(Legalizer().check(window_database).legal, "Window detailed placement broke legality.");

    expect(parseDetailedPlacementMethod("adjacent-swap") == DetailedPlacementMethod::AdjacentSwap,
           "Detailed-placement adjacent-swap alias was not parsed.");
    expect(parseDetailedPlacementMethod("window-reorder") == DetailedPlacementMethod::WindowReorder,
           "Detailed-placement window-reorder alias was not parsed.");

    const auto expect_invalid_options = [](const DetailedPlacementOptions& invalid_options,
                                           const char* failure_message) {
        bool rejected = false;
        try {
            PlacementDatabase invalid_database = makeReorderingFixture();
            DetailedPlacer().run(invalid_database, invalid_options);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected, failure_message);
    };
    DetailedPlacementOptions invalid_options;
    invalid_options.method = DetailedPlacementMethod::AdjacentSwap;
    invalid_options.improvement_epsilon = std::numeric_limits<double>::quiet_NaN();
    expect_invalid_options(invalid_options, "Detailed placement accepted a non-finite improvement tolerance.");
    invalid_options = {};
    invalid_options.method = static_cast<DetailedPlacementMethod>(99);
    expect_invalid_options(invalid_options, "Detailed placement accepted an unknown method.");
    invalid_options = {};
    invalid_options.method = DetailedPlacementMethod::AdjacentSwap;
    invalid_options.compute_backend = DetailedPlacementBackend::Auto;
    invalid_options.cuda_device = 0;
    expect_invalid_options(invalid_options, "Detailed placement accepted an invalid CUDA device.");
}

}  // namespace myplacement::test
