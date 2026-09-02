#include "myplacement/export/GdsWriter.hpp"
#include "myplacement/export/Renderer.hpp"
#include "myplacement/io/BookshelfParser.hpp"
#include "myplacement/metrics/Metrics.hpp"
#include "myplacement/placement/CudaDevicePolicy.hpp"
#include "myplacement/placement/DetailedPlacer.hpp"
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
    DetailedPlacementOptions detailed_options;
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
           << "  --routability-model <mode>  disabled (default), rudy_hinge_l2, rudy_softplus_l2, or rudy_hinge_l4\n"
           << "  --rudy-bins <count>         Square RUDY proxy grid size (default: 64)\n"
           << "  --rudy-validation-bins <n>  Held-out RUDY diagnostic grid; 0 disables it (default: 0)\n"
           << "  --rudy-validation-capacity-factor <v>  Fixed capacity multiple for held-out scoring\n"
           << "  --rudy-capacity-factor <v>  Activation-demand multiple used as fixed RUDY capacity (default: 1.0)\n"
           << "  --rudy-weight <v>           Normalized RUDY-gradient emphasis (default: 0.20)\n"
           << "  --rudy-start-overflow <v>   Enable RUDY after design density reaches this overflow\n"
           << "  --rudy-ramp-iters <count>   Accepted steps used to ramp RUDY weight (default: 24)\n"
           << "  --rudy-min-span-bins <v>    Minimum RUDY net span as a bin fraction (default: 0.25)\n"
           << "  --rudy-softplus-temp <v>    Softplus utilization temperature (default: 0.10)\n"
           << "  --compute-backend <mode>    cpu (default), cuda, or auto\n"
           << "  --gpu-device <1..4,7>       Permitted shared-server CUDA device (default: 1)\n"
           << "  --gpu-memory-limit-gib <n>  Explicit CUDA allocation cap, 1..40 (default: 40)\n"
           << "  --target-density <value>    Target density in (0, 1]\n"
           << "  --seed <integer>            Reproducible random seed\n"
           << "  --no-global                 Stop after initial placement\n"
           << "  --no-legalize               Do not run legalization\n"
           << "  --legalizer <mode>          abacus (default) or greedy A/B baseline\n"
           << "  --detailed-placement <mode> none (default), swap, or window\n"
           << "  --detailed-backend <mode>   cpu (default), cuda, or auto\n"
           << "  --detailed-passes <count>  Detailed-placement optimization passes\n"
           << "  --detailed-window <count>  Detailed-placement window size, 2 through 6\n"
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
        } else if (argument == "--routability-model") {
            command.global_options.rudy_options.penalty_model =
                parseRudyPenaltyModel(takeValue(index, argc, argv, argument));
        } else if (argument == "--rudy-bins") {
            const int bins = std::stoi(takeValue(index, argc, argv, argument));
            command.global_options.rudy_options.columns = bins;
            command.global_options.rudy_options.rows = bins;
        } else if (argument == "--rudy-validation-bins") {
            command.global_options.rudy_validation_bins = std::stoi(takeValue(index, argc, argv, argument));
        } else if (argument == "--rudy-validation-capacity-factor") {
            command.global_options.rudy_validation_capacity_factor =
                std::stod(takeValue(index, argc, argv, argument));
        } else if (argument == "--rudy-capacity-factor") {
            command.global_options.rudy_options.capacity_factor =
                std::stod(takeValue(index, argc, argv, argument));
        } else if (argument == "--rudy-weight") {
            command.global_options.routability_weight_scale =
                std::stod(takeValue(index, argc, argv, argument));
        } else if (argument == "--rudy-start-overflow") {
            command.global_options.routability_start_overflow =
                std::stod(takeValue(index, argc, argv, argument));
        } else if (argument == "--rudy-ramp-iters") {
            command.global_options.routability_ramp_iterations =
                std::stoi(takeValue(index, argc, argv, argument));
        } else if (argument == "--rudy-min-span-bins") {
            command.global_options.rudy_options.minimum_span_in_bins =
                std::stod(takeValue(index, argc, argv, argument));
        } else if (argument == "--rudy-softplus-temp") {
            command.global_options.rudy_options.softplus_temperature =
                std::stod(takeValue(index, argc, argv, argument));
        } else if (argument == "--compute-backend") {
            command.global_options.compute_backend = parseComputeBackend(takeValue(index, argc, argv, argument));
        } else if (argument == "--gpu-device") {
            command.global_options.cuda_device = std::stoi(takeValue(index, argc, argv, argument));
            if (!isPermittedCudaDevice(command.global_options.cuda_device)) {
                throw std::invalid_argument("--gpu-device must be 1 through 4 or 7 on this shared server.");
            }
            command.detailed_options.cuda_device = command.global_options.cuda_device;
        } else if (argument == "--gpu-memory-limit-gib") {
            const unsigned long long gib = std::stoull(takeValue(index, argc, argv, argument));
            if (gib == 0ULL || gib > 40ULL) {
                throw std::invalid_argument("--gpu-memory-limit-gib must be between 1 and 40.");
            }
            command.global_options.maximum_cuda_memory_bytes =
                static_cast<std::size_t>(gib * 1024ULL * 1024ULL * 1024ULL);
            command.detailed_options.maximum_cuda_memory_bytes = command.global_options.maximum_cuda_memory_bytes;
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
        } else if (argument == "--legalizer") {
            command.legalization_options.strategy =
                parseLegalizationStrategy(takeValue(index, argc, argv, argument));
        } else if (argument == "--detailed-placement") {
            command.detailed_options.method = parseDetailedPlacementMethod(takeValue(index, argc, argv, argument));
        } else if (argument == "--detailed-backend") {
            command.detailed_options.compute_backend =
                parseDetailedPlacementBackend(takeValue(index, argc, argv, argument));
        } else if (argument == "--detailed-passes") {
            command.detailed_options.passes = std::stoi(takeValue(index, argc, argv, argument));
        } else if (argument == "--detailed-window") {
            command.detailed_options.window_size = std::stoi(takeValue(index, argc, argv, argument));
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
              "penalty,smoothing,step_size,maximum_displacement,gradient_norm,curvature,rudy_energy,"
              "rudy_proxy_overflow,rudy_maximum_utilization,rudy_weight,backtracks,momentum_restarted,"
              "accepted,best_checkpoint,rudy_active\n";
    output << std::setprecision(12);
    for (const GlobalPlacementIteration& row : result.history) {
        output << row.iteration << ',' << row.hpwl << ',' << row.overflow << ',' << row.design_overflow << ','
               << row.smooth_wirelength << ',' << row.density_energy << ',' << row.objective << ',' << row.penalty
               << ',' << row.smoothing << ',' << row.step_size << ',' << row.maximum_displacement << ','
               << row.gradient_norm << ',' << row.curvature << ',' << row.rudy_energy << ','
               << row.rudy_proxy_overflow << ',' << row.rudy_maximum_utilization << ',' << row.rudy_weight << ','
               << row.backtracks << ','
               << (row.momentum_restarted ? 1 : 0) << ',' << (row.accepted ? 1 : 0) << ','
               << (row.best_checkpoint ? 1 : 0) << ',' << (row.rudy_active ? 1 : 0) << '\n';
    }
}

void writeOverview(const std::filesystem::path& path, const PlacementDatabase& database,
                   const WirelengthMetrics& wirelength, const DensityMetrics& density,
                   DensityFieldBoundary density_field_boundary, GlobalOptimizer optimizer,
                   ComputeBackend requested_backend, LegalizationStrategy legalization_strategy,
                   RudyPenaltyModel rudy_model,
                   const RudyOptions& rudy_options, double routability_start_overflow,
                   double routability_weight_scale, int routability_ramp_iterations, int rudy_validation_bins,
                   double rudy_validation_capacity_factor,
                   const GlobalPlacementResult* global_result,
                   const LegalityReport* legality, const LegalityReport* legalization_report,
                   DetailedPlacementMethod detailed_method,
                   DetailedPlacementBackend detailed_backend,
                   const DetailedPlacementResult* detailed_result) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Unable to write overview: " + path.string());
    const DatabaseSummary summary = database.summary();
    RudyEvaluation final_rudy;
    RudyEvaluation final_rudy_validation;
    bool has_final_rudy = false;
    bool has_final_rudy_validation = false;
    if (global_result != nullptr) {
        const RudyCapacity rudy_capacity{global_result->rudy_metrics.horizontal_capacity,
                                         global_result->rudy_metrics.vertical_capacity};
        if (rudy_capacity.valid()) {
            final_rudy = evaluateRudy(database, rudy_options, rudy_capacity);
            has_final_rudy = true;
        }
        const RudyCapacity validation_capacity{global_result->rudy_validation_metrics.horizontal_capacity,
                                               global_result->rudy_validation_metrics.vertical_capacity};
        if (rudy_validation_bins > 0 && validation_capacity.valid()) {
            RudyOptions validation_options = rudy_options;
            validation_options.columns = rudy_validation_bins;
            validation_options.rows = rudy_validation_bins;
            validation_options.capacity_factor = rudy_validation_capacity_factor;
            final_rudy_validation = evaluateRudy(database, validation_options, validation_capacity);
            has_final_rudy_validation = true;
        }
    }
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
           << "legalization_strategy=" << toString(legalization_strategy) << '\n'
           << "detailed_placement=" << toString(detailed_method) << '\n'
           << "detailed_backend_requested=" << toString(detailed_backend) << '\n'
           << "routability_metric=" << toString(rudy_model) << '\n'
           << "rudy_grid_columns=" << rudy_options.columns << '\n'
           << "rudy_grid_rows=" << rudy_options.rows << '\n'
           << "rudy_capacity_factor=" << rudy_options.capacity_factor << '\n'
           << "rudy_minimum_span_in_bins=" << rudy_options.minimum_span_in_bins << '\n'
           << "rudy_softplus_temperature=" << rudy_options.softplus_temperature << '\n'
           << "rudy_validation_bins=" << rudy_validation_bins << '\n'
           << "rudy_validation_capacity_factor=" << rudy_validation_capacity_factor << '\n'
           << "routability_start_overflow=" << routability_start_overflow << '\n'
           << "routability_weight_scale=" << routability_weight_scale << '\n'
           << "routability_ramp_iterations=" << routability_ramp_iterations << '\n'
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
               << "global_elapsed_seconds=" << global_result->elapsed_seconds << '\n'
               << "compute_backend_used=" << toString(global_result->compute_backend_used) << '\n'
               << "cuda_device_used=" << global_result->cuda_device_used << '\n'
               << "cuda_reserved_memory_bytes=" << global_result->cuda_reserved_memory_bytes << '\n';
        if (has_final_rudy) {
            output << "rudy_evaluation_stage=final_database" << '\n'
                   << "rudy_global_checkpoint_energy=" << global_result->rudy_energy_after << '\n'
                   << "rudy_global_checkpoint_proxy_overflow="
                   << global_result->rudy_metrics.proxy_overflow << '\n'
                   << "rudy_energy=" << final_rudy.energy << '\n'
                   << "rudy_proxy_overflow=" << final_rudy.metrics.proxy_overflow << '\n'
                   << "rudy_maximum_utilization=" << final_rudy.metrics.maximum_utilization << '\n'
                   << "rudy_p95_utilization=" << final_rudy.metrics.p95_utilization << '\n'
                   << "rudy_horizontal_capacity=" << final_rudy.metrics.horizontal_capacity << '\n'
                   << "rudy_vertical_capacity=" << final_rudy.metrics.vertical_capacity << '\n';
        }
        if (has_final_rudy_validation) {
            output << "rudy_validation_global_checkpoint_energy="
                   << global_result->rudy_validation_energy_after << '\n'
                   << "rudy_validation_global_checkpoint_proxy_overflow="
                   << global_result->rudy_validation_metrics.proxy_overflow << '\n'
                   << "rudy_validation_energy=" << final_rudy_validation.energy << '\n'
                   << "rudy_validation_proxy_overflow="
                   << final_rudy_validation.metrics.proxy_overflow << '\n'
                   << "rudy_validation_maximum_utilization="
                   << final_rudy_validation.metrics.maximum_utilization << '\n'
                   << "rudy_validation_p95_utilization="
                   << final_rudy_validation.metrics.p95_utilization << '\n'
                   << "rudy_validation_horizontal_capacity="
                   << final_rudy_validation.metrics.horizontal_capacity << '\n'
                   << "rudy_validation_vertical_capacity="
                   << final_rudy_validation.metrics.vertical_capacity << '\n';
        }
    }
    if (legality != nullptr) {
        output << "legal=" << (legality->legal ? "true" : "false") << '\n'
               << "overlap_pairs=" << legality->overlap_pairs << '\n'
               << "out_of_core_modules=" << legality->out_of_core_modules << '\n'
               << "off_row_modules=" << legality->off_row_modules << '\n'
               << "unplaced_standard_cells=" << legality->unplaced_standard_cells << '\n';
    }
    if (legalization_report != nullptr) {
        output << "legalization_strategy_used=" << toString(legalization_report->strategy) << '\n'
               << "legalization_reverse_pass_selected="
               << (legalization_report->abacus_reverse_pass_selected ? "true" : "false") << '\n'
               << "legalization_elapsed_seconds=" << legalization_report->elapsed_seconds << '\n'
               << "standard_cell_total_displacement="
               << legalization_report->standard_cell_total_displacement << '\n'
               << "standard_cell_total_squared_displacement="
               << legalization_report->standard_cell_total_squared_displacement << '\n'
               << "standard_cell_weighted_squared_displacement="
               << legalization_report->standard_cell_weighted_squared_displacement << '\n'
               << "standard_cell_maximum_displacement="
               << legalization_report->standard_cell_maximum_displacement << '\n';
    }
    if (detailed_result != nullptr) {
        output << "detailed_hpwl_before=" << detailed_result->hpwl_before << '\n'
               << "detailed_hpwl_after=" << detailed_result->hpwl_after << '\n'
               << "detailed_elapsed_seconds=" << detailed_result->elapsed_seconds << '\n'
               << "detailed_completed_passes=" << detailed_result->completed_passes << '\n'
               << "detailed_evaluated_windows=" << detailed_result->evaluated_windows << '\n'
               << "detailed_evaluated_permutations=" << detailed_result->evaluated_permutations << '\n'
               << "detailed_accepted_operations=" << detailed_result->accepted_operations << '\n'
               << "detailed_compute_backend_used="
               << toString(detailed_result->compute_backend_used) << '\n'
               << "detailed_cuda_device_used=" << detailed_result->cuda_device_used << '\n'
               << "detailed_cuda_reserved_memory_bytes="
               << detailed_result->cuda_reserved_memory_bytes << '\n';
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
        LegalityReport legalization_report;
        const LegalityReport* legality_pointer = nullptr;
        const LegalityReport* legalization_report_pointer = nullptr;
        if (command.run_legalization) {
            legality = Legalizer().legalize(database, command.legalization_options);
            legalization_report = legality;
            legality_pointer = &legality;
            legalization_report_pointer = &legalization_report;
            std::cout << "Legalization [" << toString(legality.strategy)
                      << (legality.abacus_reverse_pass_selected ? ", reverse" : "") << "]: "
                      << (legality.legal ? "PASS" : "CHECK REQUIRED")
                      << ", overlaps=" << legality.overlap_pairs << ", off-row=" << legality.off_row_modules
                      << ", unplaced=" << legality.unplaced_standard_cells
                      << ", std-cell movement=" << legality.standard_cell_total_displacement
                      << ", time=" << legality.elapsed_seconds << " s\n";
            if (command.write_bitmap) renderer.writeBitmap(database, command.output_directory / "04_legalized.bmp");
        }

        DetailedPlacementResult detailed_result;
        const DetailedPlacementResult* detailed_result_pointer = nullptr;
        if (command.detailed_options.method != DetailedPlacementMethod::None) {
            if (!command.run_legalization) {
                throw std::invalid_argument("Detailed placement requires legalization; remove --no-legalize.");
            }
            detailed_result = DetailedPlacer().run(database, command.detailed_options);
            detailed_result_pointer = &detailed_result;
            legality = Legalizer().check(database, command.legalization_options.epsilon);
            legality_pointer = &legality;
            std::cout << "Detailed placement [" << toString(command.detailed_options.method) << ", "
                      << toString(detailed_result.compute_backend_used) << "]: HPWL "
                      << detailed_result.hpwl_before << " -> " << detailed_result.hpwl_after
                      << ", accepted=" << detailed_result.accepted_operations
                      << ", windows=" << detailed_result.evaluated_windows << " in "
                      << detailed_result.elapsed_seconds << " s\n";
            if (!legality.legal) {
                throw std::runtime_error("Detailed placement violated legalization constraints.");
            }
            if (command.write_bitmap) renderer.writeBitmap(database, command.output_directory / "05_detailed.bmp");
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
                      command.global_options.compute_backend, command.legalization_options.strategy,
                      command.global_options.rudy_options.penalty_model,
                      command.global_options.rudy_options, command.global_options.routability_start_overflow,
                      command.global_options.routability_weight_scale,
                      command.global_options.routability_ramp_iterations,
                      command.global_options.rudy_validation_bins,
                      command.global_options.rudy_validation_capacity_factor,
                      global_result_pointer, legality_pointer, legalization_report_pointer,
                      command.detailed_options.method, command.detailed_options.compute_backend,
                      detailed_result_pointer);
        std::cout << "Outputs written to " << command.output_directory << '\n';
        return legality_pointer != nullptr && !legality_pointer->legal ? 2 : 0;
    } catch (const std::exception& exception) {
        std::cerr << "myplace: " << exception.what() << '\n';
        return 1;
    }
}
