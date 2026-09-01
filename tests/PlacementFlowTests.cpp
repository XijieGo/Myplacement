#include "TestAssertions.hpp"

#include "myplacement/export/GdsWriter.hpp"
#include "myplacement/export/Renderer.hpp"
#include "myplacement/io/BookshelfParser.hpp"
#include "myplacement/metrics/Metrics.hpp"
#include "myplacement/placement/GlobalPlacer.hpp"
#include "myplacement/placement/InitialPlacer.hpp"
#include "myplacement/placement/Legalizer.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace myplacement::test {
namespace {

void expectMovablesInCore(const PlacementDatabase& database) {
    for (const ModuleId id : database.movableModules()) {
        const Rect rect = database.modules[id].rect();
        expect(rect.ll.x >= database.core_region.ll.x - kEpsilon, "A movable module escaped the core on x.");
        expect(rect.ur.x <= database.core_region.ur.x + kEpsilon, "A movable module escaped the core on x.");
        expect(rect.ll.y >= database.core_region.ll.y - kEpsilon, "A movable module escaped the core on y.");
        expect(rect.ur.y <= database.core_region.ur.y + kEpsilon, "A movable module escaped the core on y.");
    }
}

void expectFiniteGlobalHistory(const GlobalPlacementResult& result, bool expect_every_iteration) {
    expect(result.completed_iterations > 0, "Global placement did not run.");
    expect(!result.history.empty(), "Global placement did not emit diagnostics.");
    if (expect_every_iteration) {
        expect(result.history.size() == static_cast<std::size_t>(result.completed_iterations),
               "Adaptive placement did not record one diagnostic row per iteration.");
    }
    for (const GlobalPlacementIteration& row : result.history) {
        for (const double value : {row.hpwl, row.overflow, row.design_overflow, row.smooth_wirelength,
                                   row.density_energy, row.objective, row.penalty, row.smoothing, row.step_size,
                                   row.maximum_displacement, row.gradient_norm, row.curvature}) {
            expect(std::isfinite(value), "Global-placement diagnostic contains a non-finite value.");
        }
        expect(row.step_size >= 0.0 && row.maximum_displacement >= 0.0 && row.backtracks >= 0,
               "Global-placement diagnostic has an invalid controller value.");
    }
}

void expectFile(const std::filesystem::path& path, const std::string& expected_prefix) {
    expect(std::filesystem::exists(path), "Expected output file was not written: " + path.string());
    expect(std::filesystem::file_size(path) > expected_prefix.size(), "Output file is unexpectedly small.");
    std::ifstream input(path, std::ios::binary);
    std::string prefix(expected_prefix.size(), '\0');
    input.read(prefix.data(), static_cast<std::streamsize>(prefix.size()));
    expect(prefix == expected_prefix, "Output file has an unexpected signature: " + path.string());
}

std::uint16_t readBigEndianU16(const std::vector<unsigned char>& bytes, std::size_t offset) {
    return static_cast<std::uint16_t>((static_cast<std::uint16_t>(bytes.at(offset)) << 8U) |
                                      static_cast<std::uint16_t>(bytes.at(offset + 1U)));
}

void verifyGdsStructure(const std::filesystem::path& path, std::size_t expected_boundaries) {
    std::ifstream input(path, std::ios::binary);
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    expect(bytes.size() >= 6U, "GDSII file is shorter than its header.");
    std::size_t offset = 0;
    std::size_t boundary_count = 0;
    std::size_t end_element_count = 0;
    bool saw_header = false;
    bool saw_end_library = false;
    while (offset < bytes.size()) {
        expect(offset + 4U <= bytes.size(), "GDSII record header is truncated.");
        const std::uint16_t length = readBigEndianU16(bytes, offset);
        expect(length >= 4U && (length % 2U) == 0U, "GDSII record length is invalid.");
        expect(offset + length <= bytes.size(), "GDSII record extends beyond file end.");
        const unsigned char record_type = bytes[offset + 2U];
        const unsigned char data_type = bytes[offset + 3U];
        if (offset == 0U) {
            expect(length == 6U && record_type == 0x00U && data_type == 0x02U,
                   "GDSII header record is invalid.");
            saw_header = true;
        }
        if (record_type == 0x08U) ++boundary_count;
        if (record_type == 0x11U) ++end_element_count;
        if (record_type == 0x10U) {
            expect(data_type == 0x03U && length == 44U, "GDSII rectangle XY record is invalid.");
        }
        if (record_type == 0x04U) saw_end_library = true;
        offset += length;
    }
    expect(saw_header && saw_end_library, "GDSII library boundary records are missing.");
    expect(boundary_count == expected_boundaries, "GDSII did not write every module as a boundary.");
    expect(end_element_count == expected_boundaries, "GDSII boundary termination count is wrong.");
}

}  // namespace

void runPlacementFlowTests() {
    const std::filesystem::path fixture = std::filesystem::path(MYPLACEMENT_TEST_FIXTURE_DIR) / "tiny.aux";
    PlacementDatabase database = BookshelfParser().parseAux(fixture);
    const DatabaseSummary summary = database.summary();
    expect(summary.module_count == 4U, "Parser did not read all modules.");
    expect(summary.movable_count == 3U, "Parser did not classify movable modules.");
    expect(summary.fixed_count == 1U, "Parser did not classify the fixed terminal.");
    expect(summary.net_count == 2U && summary.pin_count == 5U, "Parser did not reconstruct the netlist.");
    expect(std::abs(database.core_region.width() - 100.0) < 1e-9, "Core width is wrong.");
    expect(std::abs(database.core_region.height() - 40.0) < 1e-9, "Core height is wrong.");
    expect(calculateHpwl(database).hpwl > 0.0, "HPWL was not calculated.");

    const std::vector<Vec2> original_centers = [&]() {
        std::vector<Vec2> centers;
        for (const Module& module : database.modules) centers.push_back(module.center);
        return centers;
    }();
    InitialPlacer initial_placer;
    for (const InitialMethod method : {InitialMethod::Random, InitialMethod::Clustering, InitialMethod::Quadratic}) {
        for (std::size_t index = 0; index < database.modules.size(); ++index) {
            database.modules[index].center = original_centers[index];
        }
        InitialPlacementOptions options;
        options.quadratic_outer_iterations = 3;
        initial_placer.run(database, method, options);
        expectMovablesInCore(database);
        expect(std::isfinite(calculateHpwl(database).hpwl), "Initial placement returned non-finite HPWL.");
    }

    GlobalPlacementOptions global_options;
    global_options.iterations = 24;
    global_options.bins_x = 8;
    global_options.bins_y = 8;
    global_options.maximum_fillers = 100;
    const GlobalPlacementResult global = GlobalPlacer().run(database, global_options);
    expectFiniteGlobalHistory(global, true);
    expect(global.accepted_iterations > 0, "Adaptive global placement accepted no update.");
    expect(global.restored_best_checkpoint, "Adaptive global placement did not restore a checkpoint.");
    expect(global.best_checkpoint_iteration <= global.completed_iterations,
           "Adaptive checkpoint iteration is outside the run.");
    expectMovablesInCore(database);

    PlacementDatabase controller_database = BookshelfParser().parseAux(fixture);
    InitialPlacementOptions controller_initial_options;
    controller_initial_options.quadratic_outer_iterations = 3;
    initial_placer.run(controller_database, InitialMethod::Quadratic, controller_initial_options);
    GlobalPlacementOptions controller_options;
    controller_options.iterations = 24;
    controller_options.bins_x = 8;
    controller_options.bins_y = 8;
    controller_options.maximum_fillers = 100;
    controller_options.initial_movement_in_bins = 1.0;
    controller_options.maximum_movement_in_bins = 1.0;
    controller_options.armijo_coefficient = 0.5;
    controller_options.objective_increase_for_density = 0.0;
    const GlobalPlacementResult controller_global = GlobalPlacer().run(controller_database, controller_options);
    expectFiniteGlobalHistory(controller_global, true);
    expect(controller_global.rejected_candidates > 0, "An aggressive controller test did not exercise backtracking.");
    expect(controller_global.momentum_restarts > 0,
           "An aggressive controller test did not exercise momentum restart.");
    expectMovablesInCore(controller_database);

    PlacementDatabase periodic_database = BookshelfParser().parseAux(fixture);
    InitialPlacementOptions periodic_initial_options;
    periodic_initial_options.quadratic_outer_iterations = 3;
    initial_placer.run(periodic_database, InitialMethod::Quadratic, periodic_initial_options);
    global_options.density_field_boundary = DensityFieldBoundary::Periodic;
    const GlobalPlacementResult periodic_global = GlobalPlacer().run(periodic_database, global_options);
    expectFiniteGlobalHistory(periodic_global, true);
    expectMovablesInCore(periodic_database);

    PlacementDatabase legacy_database = BookshelfParser().parseAux(fixture);
    initial_placer.run(legacy_database, InitialMethod::Quadratic, periodic_initial_options);
    global_options.optimizer = GlobalOptimizer::Legacy;
    const GlobalPlacementResult legacy_global = GlobalPlacer().run(legacy_database, global_options);
    expectFiniteGlobalHistory(legacy_global, false);
    expect(!legacy_global.restored_best_checkpoint,
           "The legacy baseline unexpectedly reported an adaptive checkpoint.");
    expectMovablesInCore(legacy_database);

    expect(parseGlobalOptimizer("closed-loop") == GlobalOptimizer::Adaptive,
           "Closed-loop optimizer alias did not select the adaptive optimizer.");
    expect(parseGlobalOptimizer("open_loop") == GlobalOptimizer::Legacy,
           "Open-loop optimizer alias did not select the legacy optimizer.");
    expect(parseComputeBackend("gpu") == ComputeBackend::Cuda,
           "GPU backend alias did not select CUDA.");
    expect(parseComputeBackend("auto") == ComputeBackend::Auto,
           "Automatic backend selection was not parsed.");
    bool rejected_invalid_controller_option = false;
    try {
        PlacementDatabase invalid_options_database = BookshelfParser().parseAux(fixture);
        GlobalPlacementOptions invalid_options;
        invalid_options.iterations = 1;
        invalid_options.bins_x = 8;
        invalid_options.bins_y = 8;
        invalid_options.feasible_refinement_iterations = 0;
        GlobalPlacer().run(invalid_options_database, invalid_options);
    } catch (const std::invalid_argument&) {
        rejected_invalid_controller_option = true;
    }
    expect(rejected_invalid_controller_option,
           "Adaptive global placement accepted an invalid feasible-refinement limit.");

    const LegalityReport legality = Legalizer().legalize(database, {});
    expect(legality.legal, "Legalizer did not produce a legal tiny placement.");

    const std::filesystem::path output = std::filesystem::temp_directory_path() / "myplacement-test-output";
    Renderer().writeBitmap(database, output / "placement.bmp");
    GdsWriter().write(database, output / "placement.gds");
    expectFile(output / "placement.bmp", std::string("BM", 2));
    expectFile(output / "placement.gds", std::string("\x00\x06\x00\x02", 4));
    verifyGdsStructure(output / "placement.gds", database.modules.size());

    PlacementDatabase mixed = BookshelfParser().parseAux(
        std::filesystem::path(MYPLACEMENT_TEST_FIXTURE_DIR) / "mixed.aux");
    expect(mixed.summary().macro_count == 1U, "Mixed-size fixture did not classify its macro.");
    const DensityMetrics mixed_density = calculateDensity(mixed, 8, 8, 0.85);
    expectNear(mixed_density.normalization_area, 540.0, 1e-9,
               "Macro density was not target-scaled in the course denominator.");
    GlobalPlacementOptions mixed_global_options;
    mixed_global_options.iterations = 16;
    mixed_global_options.bins_x = 8;
    mixed_global_options.bins_y = 8;
    mixed_global_options.maximum_fillers = 100;
    const GlobalPlacementResult mixed_global = GlobalPlacer().run(mixed, mixed_global_options);
    expectFiniteGlobalHistory(mixed_global, true);
    expectMovablesInCore(mixed);
    const LegalityReport mixed_legality = Legalizer().legalize(mixed, {});
    expect(mixed_legality.legal, "Legalizer did not resolve macro/fixed-terminal overlap.");
}

}  // namespace myplacement::test
