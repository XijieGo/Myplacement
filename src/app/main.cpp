#include "myplacement/export/GdsWriter.hpp"
#include "myplacement/export/Renderer.hpp"
#include "myplacement/io/BookshelfParser.hpp"
#include "myplacement/metrics/Metrics.hpp"
#include "myplacement/placement/GlobalPlacer.hpp"
#include "myplacement/placement/InitialPlacer.hpp"
#include "myplacement/placement/Legalizer.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace myplacement {
namespace {

struct CommandLine {
    std::filesystem::path aux_path;
    // Ad-hoc runs must not mix with the curated diagnostics or verified results.
    std::filesystem::path output_directory = "out/scratch";
    InitialMethod initial_method = InitialMethod::Quadratic;
    bool compare_initial_methods = false;
    bool run_global = true;
    bool run_legalization = true;
    bool write_bitmap = true;
    bool write_gds = true;
    bool parse_only = false;
    InitialPlacementOptions initial_options;
    GlobalPlacementOptions global_options;
    LegalizationOptions legalization_options;
};

void printUsage(std::ostream& output) {
    output << "Usage: myplace <design.aux> [options]\n\n"
           << "Core options:\n"
           << "  --output <directory>        Output directory (default: out/scratch)\n"
           << "  --initial <method|all>      random, cluster, quadratic, or all\n"
           << "  --iterations <count>        Global-placement iteration limit\n"
           << "  --quadratic-iters <count>  Reweighted quadratic outer iterations\n"
           << "  --quadratic-solver-iters <count>  Linear iterations per quadratic solve\n"
           << "  --bins <count>              Square density grid size\n"
           << "  --density-field <mode>      neumann (default) or periodic A/B baseline\n"
           << "  --global-optimizer <mode>   adaptive (default) or legacy A/B baseline\n"
           << "  --compute-backend <mode>    cpu (default), cuda, or auto\n"
           << "  --gpu-device <1..4>         Shared-server CUDA device (default: 1)\n"
           << "  --gpu-memory-limit-gib <n>  Explicit CUDA allocation cap, 1..40 (default: 40)\n"
           << "  --target-density <value>    Target density in (0, 1]\n"
           << "  --seed <integer>            Reproducible random seed\n"
           << "  --no-global                 Stop after initial placement\n"
           << "  --no-legalize               Do not run legalization\n"
           << "  --no-bmp                    Do not render BMP images\n"
           << "  --no-gds                    Do not export GDSII\n"
           << "  --parse-only                Parse and validate input only\n"
           << "  --help                      Show this message\n";
}

std::string takeValue(int& index, int argc, char* argv[], const std::string& option) {
    if (++index >= argc) throw std::invalid_argument(option + " requires a value.");
    return argv[index];
}

CommandLine parseCommandLine(int argc, char* argv[]) {
    if (argc <= 1) {
        printUsage(std::cerr);
        throw std::invalid_argument("A BookShelf AUX file is required.");
    }
    CommandLine command;
    bool saw_input = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            printUsage(std::cout);
            std::exit(0);
        }
        if (argument == "--output") {
            command.output_directory = takeValue(index, argc, argv, argument);
        } else if (argument == "--initial") {
            const std::string value = takeValue(index, argc, argv, argument);
            if (value == "all") {
                command.compare_initial_methods = true;
                command.initial_method = InitialMethod::Quadratic;
            } else {
                command.initial_method = parseInitialMethod(value);
            }
        } else if (argument == "--iterations") {
            command.global_options.iterations = std::stoi(takeValue(index, argc, argv, argument));
        } else if (argument == "--bins") {
            const int bins = std::stoi(takeValue(index, argc, argv, argument));
            command.global_options.bins_x = bins;
            command.global_options.bins_y = bins;
        } else if (argument == "--density-field") {
            command.global_options.density_field_boundary =
                parseDensityFieldBoundary(takeValue(index, argc, argv, argument));
        } else if (argument == "--global-optimizer") {
            command.global_options.optimizer = parseGlobalOptimizer(takeValue(index, argc, argv, argument));
        } else if (argument == "--compute-backend") {
            command.global_options.compute_backend = parseComputeBackend(takeValue(index, argc, argv, argument));
        } else if (argument == "--gpu-device") {
            command.global_options.cuda_device = std::stoi(takeValue(index, argc, argv, argument));
            if (command.global_options.cuda_device < 1 || command.global_options.cuda_device > 4) {
                throw std::invalid_argument("--gpu-device must be between 1 and 4 on this shared server.");
            }
        } else if (argument == "--gpu-memory-limit-gib") {
            const unsigned long long gib = std::stoull(takeValue(index, argc, argv, argument));
            if (gib == 0ULL || gib > 40ULL) {
                throw std::invalid_argument("--gpu-memory-limit-gib must be between 1 and 40.");
            }
            command.global_options.maximum_cuda_memory_bytes =
                static_cast<std::size_t>(gib * 1024ULL * 1024ULL * 1024ULL);
        } else if (argument == "--target-density") {
            command.global_options.target_density = std::stod(takeValue(index, argc, argv, argument));
        } else if (argument == "--seed") {
            const auto seed = static_cast<std::uint64_t>(std::stoull(takeValue(index, argc, argv, argument)));
            command.initial_options.seed = seed;
            command.global_options.seed = seed;
        } else if (argument == "--quadratic-iters") {
            command.initial_options.quadratic_outer_iterations = std::stoi(takeValue(index, argc, argv, argument));
        } else if (argument == "--quadratic-solver-iters") {
            command.initial_options.quadratic_solver_iterations = std::stoi(takeValue(index, argc, argv, argument));
        } else if (argument == "--no-global") {
            command.run_global = false;
        } else if (argument == "--no-legalize") {
            command.run_legalization = false;
        } else if (argument == "--no-bmp") {
            command.write_bitmap = false;
        } else if (argument == "--no-gds") {
            command.write_gds = false;
        } else if (argument == "--parse-only") {
            command.parse_only = true;
        } else if (!argument.empty() && argument.front() == '-') {
            throw std::invalid_argument("Unknown option: " + argument);
        } else if (!saw_input) {
            command.aux_path = argument;
            saw_input = true;
        } else {
            throw std::invalid_argument("Only one BookShelf AUX input is supported.");
        }
    }
    if (!saw_input) throw std::invalid_argument("A BookShelf AUX file is required.");
    return command;
}

void printSummary(const PlacementDatabase& database) {
    const DatabaseSummary summary = database.summary();
    std::cout << "Parsed design\n"
              << "  core: (" << database.core_region.ll.x << ", " << database.core_region.ll.y << ") -> ("
              << database.core_region.ur.x << ", " << database.core_region.ur.y << ")\n"
              << "  modules: " << summary.module_count << " (movable " << summary.movable_count << ", fixed "
              << summary.fixed_count << ", macros " << summary.macro_count << ")\n"
              << "  nets: " << summary.net_count << ", pins: " << summary.pin_count << "\n";
}

void restoreCenters(PlacementDatabase& database, const std::vector<Vec2>& centers) {
    for (std::size_t index = 0; index < database.modules.size(); ++index) {
        database.modules[index].center = centers[index];
    }
}

void writeInitialComparison(const std::filesystem::path& path,
                            const std::vector<InitialPlacementResult>& reports) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Unable to write initial placement report: " + path.string());
    output << "method,hpwl_before,hpwl_after,elapsed_seconds,iterations\n";
    output << std::setprecision(12);
    for (const InitialPlacementResult& report : reports) {
        output << toString(report.method) << ',' << report.hpwl_before << ',' << report.hpwl_after << ','
               << report.elapsed_seconds << ',' << report.iterations << '\n';
    }
}

void writeGlobalHistory(const std::filesystem::path& path, const GlobalPlacementResult& result) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Unable to write global placement history: " + path.string());
    output << "iteration,hpwl,optimizer_overflow,design_overflow,smooth_wirelength,density_energy,objective,"
              "penalty,smoothing,step_size,maximum_displacement,gradient_norm,curvature,backtracks,"
              "momentum_restarted,accepted,best_checkpoint\n";
    output << std::setprecision(12);
    for (const GlobalPlacementIteration& row : result.history) {
        output << row.iteration << ',' << row.hpwl << ',' << row.overflow << ',' << row.design_overflow << ','
               << row.smooth_wirelength << ',' << row.density_energy << ',' << row.objective << ',' << row.penalty
               << ',' << row.smoothing << ',' << row.step_size << ',' << row.maximum_displacement << ','
               << row.gradient_norm << ',' << row.curvature << ',' << row.backtracks << ','
               << (row.momentum_restarted ? 1 : 0) << ',' << (row.accepted ? 1 : 0) << ','
               << (row.best_checkpoint ? 1 : 0) << '\n';
    }
}

void writeOverview(const std::filesystem::path& path, const PlacementDatabase& database,
                   const WirelengthMetrics& wirelength, const DensityMetrics& density,
                   DensityFieldBoundary density_field_boundary, GlobalOptimizer optimizer,
                   ComputeBackend requested_backend,
                   const GlobalPlacementResult* global_result,
                   const LegalityReport* legality) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Unable to write overview: " + path.string());
    const DatabaseSummary summary = database.summary();
    output << std::setprecision(12)
           << "modules=" << summary.module_count << '\n'
           << "movable_modules=" << summary.movable_count << '\n'
           << "fixed_modules=" << summary.fixed_count << '\n'
           << "macros=" << summary.macro_count << '\n'
           << "nets=" << summary.net_count << '\n'
           << "pins=" << summary.pin_count << '\n'
           << "density_field=" << toString(density_field_boundary) << '\n'
           << "global_optimizer=" << toString(optimizer) << '\n'
           << "compute_backend_requested=" << toString(requested_backend) << '\n'
           << "hpwl=" << wirelength.hpwl << '\n'
           << "density_metric=course_eplace_v1" << '\n'
           << "normalized_overflow=" << density.normalized_overflow << '\n'
           << "total_overflow_area=" << density.total_overflow_area << '\n'
           << "density_normalization_area=" << density.normalization_area << '\n'
           << "density_charge_area=" << density.total_charge_area << '\n'
           << "placeable_area=" << density.placeable_area << '\n'
           << "dark_area=" << density.dark_area << '\n'
           << "maximum_density=" << density.maximum_density << '\n';
    if (global_result != nullptr) {
        output << "global_completed_iterations=" << global_result->completed_iterations << '\n'
               << "global_accepted_iterations=" << global_result->accepted_iterations << '\n'
               << "global_rejected_candidates=" << global_result->rejected_candidates << '\n'
               << "global_momentum_restarts=" << global_result->momentum_restarts << '\n'
               << "global_best_checkpoint_iteration=" << global_result->best_checkpoint_iteration << '\n'
               << "global_best_checkpoint_hpwl=" << global_result->best_checkpoint_hpwl << '\n'
               << "global_best_checkpoint_overflow=" << global_result->best_checkpoint_overflow << '\n'
               << "global_restored_best_checkpoint="
               << (global_result->restored_best_checkpoint ? "true" : "false") << '\n'
               << "compute_backend_used=" << toString(global_result->compute_backend_used) << '\n'
               << "cuda_device_used=" << global_result->cuda_device_used << '\n'
               << "cuda_reserved_memory_bytes=" << global_result->cuda_reserved_memory_bytes << '\n';
    }
    if (legality != nullptr) {
        output << "legal=" << (legality->legal ? "true" : "false") << '\n'
               << "overlap_pairs=" << legality->overlap_pairs << '\n'
               << "out_of_core_modules=" << legality->out_of_core_modules << '\n'
               << "off_row_modules=" << legality->off_row_modules << '\n'
               << "unplaced_standard_cells=" << legality->unplaced_standard_cells << '\n';
    }
}

}  // namespace
}  // namespace myplacement

int main(int argc, char* argv[]) {
    using namespace myplacement;
    try {
        const CommandLine command = parseCommandLine(argc, argv);
        std::filesystem::create_directories(command.output_directory);
        PlacementDatabase database = BookshelfParser().parseAux(command.aux_path);
        printSummary(database);
        if (command.parse_only) return 0;

        Renderer renderer;
        const std::vector<Vec2> original_centers = [&]() {
            std::vector<Vec2> centers;
            centers.reserve(database.modules.size());
            for (const Module& module : database.modules) centers.push_back(module.center);
            return centers;
        }();

        InitialPlacer initial_placer;
        if (command.compare_initial_methods) {
            std::vector<InitialPlacementResult> reports;
            for (const InitialMethod method : {InitialMethod::Random, InitialMethod::Clustering, InitialMethod::Quadratic}) {
                restoreCenters(database, original_centers);
                InitialPlacementResult report = initial_placer.run(database, method, command.initial_options);
                reports.push_back(report);
                std::cout << "Initial " << toString(method) << ": HPWL " << report.hpwl_before << " -> "
                          << report.hpwl_after << " in " << report.elapsed_seconds << " s\n";
                if (command.write_bitmap) {
                    renderer.writeBitmap(database, command.output_directory /
                        ("01_initial_" + toString(method) + ".bmp"));
                }
            }
            writeInitialComparison(command.output_directory / "initial_comparison.csv", reports);
        }

        restoreCenters(database, original_centers);
        const InitialPlacementResult initial_result =
            initial_placer.run(database, command.initial_method, command.initial_options);
        std::cout << "Selected initial method " << toString(command.initial_method) << ": HPWL "
                  << initial_result.hpwl_before << " -> " << initial_result.hpwl_after << " in "
                  << initial_result.elapsed_seconds << " s\n";
        if (command.write_bitmap) renderer.writeBitmap(database, command.output_directory / "02_initial_selected.bmp");

        GlobalPlacementResult global_result;
        const GlobalPlacementResult* global_result_pointer = nullptr;
        if (command.run_global) {
            global_result = GlobalPlacer().run(database, command.global_options);
            global_result_pointer = &global_result;
            std::cout << "Global placement [" << toString(command.global_options.density_field_boundary)
                      << ", " << toString(command.global_options.optimizer)
                      << ", " << toString(global_result.compute_backend_used) << "]: HPWL "
                      << global_result.hpwl_before << " -> "
                      << global_result.hpwl_after << ", overflow " << global_result.overflow_before << " -> "
                      << global_result.overflow_after << ", accepted=" << global_result.accepted_iterations
                      << ", rejected=" << global_result.rejected_candidates
                      << ", restarts=" << global_result.momentum_restarts
                      << ", checkpoint=" << global_result.best_checkpoint_iteration << " in "
                      << global_result.elapsed_seconds << " s\n";
            writeGlobalHistory(command.output_directory / "global_history.csv", global_result);
            if (command.write_bitmap) renderer.writeBitmap(database, command.output_directory / "03_global.bmp");
        }

        LegalityReport legality;
        const LegalityReport* legality_pointer = nullptr;
        if (command.run_legalization) {
            legality = Legalizer().legalize(database, command.legalization_options);
            legality_pointer = &legality;
            std::cout << "Legalization: " << (legality.legal ? "PASS" : "CHECK REQUIRED")
                      << ", overlaps=" << legality.overlap_pairs << ", off-row=" << legality.off_row_modules
                      << ", unplaced=" << legality.unplaced_standard_cells << '\n';
            if (command.write_bitmap) renderer.writeBitmap(database, command.output_directory / "04_legalized.bmp");
        }

        if (command.write_gds) {
            GdsWriter().write(database, command.output_directory / "placement.gds");
        }
        const WirelengthMetrics wirelength = calculateHpwl(database);
        const DensityMetrics density = calculateDensity(database, command.global_options.bins_x,
                                                        command.global_options.bins_y,
                                                        command.global_options.target_density);
        writeOverview(command.output_directory / "overview.txt", database, wirelength, density,
                      command.global_options.density_field_boundary, command.global_options.optimizer,
                      command.global_options.compute_backend,
                      global_result_pointer, legality_pointer);
        std::cout << "Outputs written to " << command.output_directory << '\n';
        return legality_pointer != nullptr && !legality_pointer->legal ? 2 : 0;
    } catch (const std::exception& exception) {
        std::cerr << "myplace: " << exception.what() << '\n';
        return 1;
    }
}
