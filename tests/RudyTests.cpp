#include "TestAssertions.hpp"

#include "myplacement/metrics/Rudy.hpp"

#include <cmath>
#include <limits>
#include <string>

namespace myplacement::test {
namespace {

PlacementDatabase makeTwoPinDatabase() {
    PlacementDatabase database;
    database.core_region = {{0.0, 0.0}, {10.0, 10.0}};
    Module first;
    first.name = "first";
    first.width = 1.0;
    first.height = 1.0;
    first.center = {1.3, 1.7};
    Module second;
    second.name = "second";
    second.width = 1.0;
    second.height = 1.0;
    second.center = {6.4, 4.6};
    const ModuleId first_id = database.addModule(first);
    const ModuleId second_id = database.addModule(second);
    Net net;
    net.name = "n";
    net.weight = 1.0;
    const NetId net_id = database.addNet(net);
    database.addPin({first_id, net_id, {}, PinDirection::Bidirectional});
    database.addPin({second_id, net_id, {}, PinDirection::Bidirectional});
    database.refreshDerivedData();
    // This fixture has no site rows, so restore its deliberately specified
    // analytical region after refreshDerivedData().
    database.core_region = {{0.0, 0.0}, {10.0, 10.0}};
    return database;
}

}  // namespace

void runRudyTests() {
    PlacementDatabase database = makeTwoPinDatabase();
    RudyOptions options;
    options.columns = 5;
    options.rows = 5;
    options.minimum_span_in_bins = 0.01;
    options.penalty_model = RudyPenaltyModel::HingeL2;

    const RudyEvaluation raw = evaluateRudy(database, options);
    expectNear(raw.metrics.horizontal_total_demand, 5.1, 1e-12,
               "RUDY horizontal demand no longer integrates to the net x-span.");
    expectNear(raw.metrics.vertical_total_demand, 2.9, 1e-12,
               "RUDY vertical demand no longer integrates to the net y-span.");
    expect(raw.metrics.counted_nets == 1U, "RUDY did not count the two-pin net.");

    const RudyCapacity capacity = calibrateRudyCapacity(raw.metrics, 0.45);
    const RudyEvaluation evaluation = evaluateRudy(database, options, capacity, true);
    expect(std::isfinite(evaluation.energy) && evaluation.energy > 0.0,
           "RUDY hotspot energy is not finite and positive.");
    expect(evaluation.metrics.maximum_utilization > 1.0,
           "RUDY capacity calibration did not expose a hotspot.");

    constexpr double delta = 1e-5;
    database.modules[0].center.x += delta;
    const double plus = evaluateRudy(database, options, capacity).energy;
    database.modules[0].center.x -= 2.0 * delta;
    const double minus = evaluateRudy(database, options, capacity).energy;
    database.modules[0].center.x += delta;
    const double finite_difference_x = (plus - minus) / (2.0 * delta);
    expectNear(evaluation.gradient[0].x, finite_difference_x, 3e-6,
               "RUDY analytical x-gradient disagrees with a finite difference.");

    database.modules[0].center.y += delta;
    const double plus_y = evaluateRudy(database, options, capacity).energy;
    database.modules[0].center.y -= 2.0 * delta;
    const double minus_y = evaluateRudy(database, options, capacity).energy;
    database.modules[0].center.y += delta;
    const double finite_difference_y = (plus_y - minus_y) / (2.0 * delta);
    expectNear(evaluation.gradient[0].y, finite_difference_y, 3e-6,
               "RUDY analytical y-gradient disagrees with a finite difference.");

    options.penalty_model = RudyPenaltyModel::SoftplusL2;
    const RudyEvaluation smooth = evaluateRudy(database, options, capacity, true);
    expect(std::isfinite(smooth.energy) && std::isfinite(smooth.gradient[0].x) &&
               std::isfinite(smooth.gradient[0].y),
           "Softplus RUDY penalty produced a non-finite value.");
    database.modules[0].center.x += delta;
    const double smooth_plus = evaluateRudy(database, options, capacity).energy;
    database.modules[0].center.x -= 2.0 * delta;
    const double smooth_minus = evaluateRudy(database, options, capacity).energy;
    database.modules[0].center.x += delta;
    expectNear(smooth.gradient[0].x, (smooth_plus - smooth_minus) / (2.0 * delta), 3e-6,
               "Softplus RUDY analytical x-gradient disagrees with a finite difference.");

    options.penalty_model = RudyPenaltyModel::HingeL4;
    const RudyEvaluation hinge_l4 = evaluateRudy(database, options, capacity, true);
    expect(std::abs(hinge_l4.gradient[0].x) > 1e-10,
           "Hinge-L4 gradient fixture no longer exercises an over-capacity RUDY bin.");
    database.modules[0].center.x += delta;
    const double hinge_l4_plus = evaluateRudy(database, options, capacity).energy;
    database.modules[0].center.x -= 2.0 * delta;
    const double hinge_l4_minus = evaluateRudy(database, options, capacity).energy;
    database.modules[0].center.x += delta;
    expectNear(hinge_l4.gradient[0].x, (hinge_l4_plus - hinge_l4_minus) / (2.0 * delta), 3e-6,
               "Hinge-L4 RUDY analytical x-gradient disagrees with a finite difference.");
    expect(parseRudyPenaltyModel("hinge_l4") == RudyPenaltyModel::HingeL4,
           "RUDY penalty parser did not select hinge_l4.");
    expect(toString(RudyPenaltyModel::SoftplusL2) == "rudy_softplus_l2",
           "RUDY penalty name changed unexpectedly.");

    // Exact RUDY is piecewise differentiable at bin boundaries. The selected
    // Clarke subgradient must remain usable there rather than becoming a
    // zero-force fixed point when a quadratic initializer lands on a grid.
    // Softplus is intentional here: below the hard threshold it still has a
    // smooth nonzero derivative, whereas Hinge-L4 correctly has no force.
    options.penalty_model = RudyPenaltyModel::SoftplusL2;
    PlacementDatabase boundary = makeTwoPinDatabase();
    boundary.modules[0].center.x = 2.0;
    boundary.modules[1].center.x = 6.0;
    const RudyEvaluation boundary_raw = evaluateRudy(boundary, options);
    const RudyCapacity boundary_capacity = calibrateRudyCapacity(boundary_raw.metrics, 0.45);
    const RudyEvaluation boundary_evaluation = evaluateRudy(boundary, options, boundary_capacity, true);
    const double boundary_gradient_l1 = std::abs(boundary_evaluation.gradient[0].x) +
                                        std::abs(boundary_evaluation.gradient[1].x);
    expect(boundary_gradient_l1 > 1e-10,
           "RUDY grid-boundary subgradient unexpectedly vanished.");

    PlacementDatabase degenerate = makeTwoPinDatabase();
    degenerate.modules[1].center = degenerate.modules[0].center;
    const RudyEvaluation degenerate_result = evaluateRudy(degenerate, options, capacity, true);
    expect(std::isfinite(degenerate_result.energy), "Degenerate RUDY net produced a non-finite energy.");
    // Minimum-span regularization may not invent an expansion force, but a
    // narrow net still has to feel a real translation force. Translate both
    // endpoints together to avoid the non-smooth raw-extrema tie.
    degenerate.modules[0].center.x += delta;
    degenerate.modules[1].center.x += delta;
    const double degenerate_plus = evaluateRudy(degenerate, options, capacity).energy;
    degenerate.modules[0].center.x -= 2.0 * delta;
    degenerate.modules[1].center.x -= 2.0 * delta;
    const double degenerate_minus = evaluateRudy(degenerate, options, capacity).energy;
    degenerate.modules[0].center.x += delta;
    degenerate.modules[1].center.x += delta;
    const double degenerate_finite_difference = (degenerate_plus - degenerate_minus) / (2.0 * delta);
    expectNear(degenerate_result.gradient[0].x + degenerate_result.gradient[1].x,
               degenerate_finite_difference, 3e-6,
               "Minimum-span RUDY regularization lost the net translation gradient.");

    const auto expect_invalid_rudy = [&](const RudyOptions& invalid_options, const RudyCapacity& invalid_capacity,
                                         const char* failure_message) {
        bool rejected = false;
        try {
            evaluateRudy(database, invalid_options, invalid_capacity);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected, failure_message);
    };
    RudyOptions invalid_options = options;
    invalid_options.minimum_span_in_bins = std::numeric_limits<double>::quiet_NaN();
    expect_invalid_rudy(invalid_options, {}, "RUDY accepted a non-finite minimum span.");
    invalid_options = options;
    invalid_options.penalty_model = static_cast<RudyPenaltyModel>(99);
    expect_invalid_rudy(invalid_options, {}, "RUDY accepted an unknown penalty model.");
    expect_invalid_rudy(options, {1.0, 0.0}, "RUDY accepted a partially specified capacity.");
}

}  // namespace myplacement::test
