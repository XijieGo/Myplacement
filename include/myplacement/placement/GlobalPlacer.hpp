#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "myplacement/metrics/Metrics.hpp"
#include "myplacement/metrics/Rudy.hpp"
#include "myplacement/placement/ElectrostaticField.hpp"

namespace myplacement {

enum class GlobalOptimizer { Legacy, Adaptive };

// CPU is the deterministic reference implementation. CUDA offloads the
// adaptive optimizer's numerical evaluation; Auto uses CUDA only when a safe
// compatible device is available.
enum class ComputeBackend { Cpu, Cuda, Auto };

std::string toString(GlobalOptimizer optimizer);
GlobalOptimizer parseGlobalOptimizer(const std::string& value);
std::string toString(ComputeBackend backend);
ComputeBackend parseComputeBackend(const std::string& value);

struct GlobalPlacementOptions {
    // Large course benchmarks can require about 255 adaptive iterations before
    // meeting the course density constraint. Early stopping keeps easy cases
    // short, while this budget prevents a4-scale designs from stopping early.
    int iterations = 280;
    int bins_x = 64;
    int bins_y = 64;
    int report_interval = 10;
    double target_density = 0.85;
    double maximum_bin_overflow = 0.10;
    double initial_smoothing = 0.75;
    double final_smoothing = 0.12;
    double maximum_movement_in_bins = 0.22;
    double penalty_growth = 1.035;
    GlobalOptimizer optimizer = GlobalOptimizer::Adaptive;
    int maximum_backtracks = 6;
    double backtracking_ratio = 0.5;
    double armijo_coefficient = 1e-4;
    double maximum_momentum = 0.92;
    double initial_movement_in_bins = 0.12;
    double objective_increase_for_density = 0.002;
    double hpwl_growth_restart_threshold = 0.02;
    double density_stall_threshold = 0.002;
    double density_penalty_increase = 1.05;
    double density_penalty_decrease = 0.88;
    double minimum_density_penalty = 1e-4;
    double maximum_density_penalty = 1e6;
    double smoothing_overflow_exponent = 1.0;
    int feasible_refinement_iterations = 12;
    std::size_t maximum_fillers = 50000;
    std::uint64_t seed = 2026;
    // BookShelf has no routing tracks or layer capacities.  This optional
    // term optimizes a clearly labelled RUDY demand-hotspot proxy, rather than
    // claiming signoff-router overflow.
    RudyOptions rudy_options;
    // Optional held-out RUDY resolution used only for post-run diagnostics.
    // It is calibrated at the same reference state as the objective grid but
    // never contributes a force, so it detects grid-specific proxy gaming.
    int rudy_validation_bins = 0;
    // Kept separate from the objective capacity factor so parameter sweeps
    // can be assessed under one stable held-out threshold.
    double rudy_validation_capacity_factor = 1.50;
    double routability_start_overflow = 0.20;
    double routability_weight_scale = 0.20;
    int routability_ramp_iterations = 24;
    DensityFieldBoundary density_field_boundary = DensityFieldBoundary::Neumann;
    ComputeBackend compute_backend = ComputeBackend::Cpu;
    // This project is hosted on a shared server where placement jobs may use
    // only physical GPUs 1 through 4 or GPU 7. The CUDA backend enforces that policy.
    int cuda_device = 1;
    // Explicit device allocations are budgeted before any CUDA allocation.
    // Keep the default below the 80 GiB device capacity and leave room for
    // other users of a shared GPU.
    std::size_t maximum_cuda_memory_bytes = 40ULL * 1024ULL * 1024ULL * 1024ULL;
};

struct GlobalPlacementIteration {
    int iteration = 0;
    double hpwl = 0.0;
    double overflow = 0.0;
    double design_overflow = 0.0;
    double smooth_wirelength = 0.0;
    double density_energy = 0.0;
    double objective = 0.0;
    double penalty = 0.0;
    double smoothing = 0.0;
    double step_size = 0.0;
    double maximum_displacement = 0.0;
    double gradient_norm = 0.0;
    double curvature = 0.0;
    double rudy_energy = 0.0;
    double rudy_proxy_overflow = 0.0;
    double rudy_maximum_utilization = 0.0;
    double rudy_weight = 0.0;
    int backtracks = 0;
    bool momentum_restarted = false;
    bool accepted = false;
    bool best_checkpoint = false;
    bool rudy_active = false;
};

struct GlobalPlacementResult {
    double hpwl_before = 0.0;
    double hpwl_after = 0.0;
    double overflow_before = 0.0;
    double overflow_after = 0.0;
    int completed_iterations = 0;
    int accepted_iterations = 0;
    int momentum_restarts = 0;
    int rejected_candidates = 0;
    int best_checkpoint_iteration = 0;
    bool restored_best_checkpoint = false;
    double best_checkpoint_hpwl = 0.0;
    double best_checkpoint_overflow = 0.0;
    double elapsed_seconds = 0.0;
    ComputeBackend compute_backend_used = ComputeBackend::Cpu;
    int cuda_device_used = -1;
    std::size_t cuda_reserved_memory_bytes = 0;
    RudyMetrics rudy_metrics;
    double rudy_energy_after = 0.0;
    RudyMetrics rudy_validation_metrics;
    double rudy_validation_energy_after = 0.0;
    std::vector<GlobalPlacementIteration> history;
};

class GlobalPlacer {
public:
    GlobalPlacementResult run(PlacementDatabase& database, const GlobalPlacementOptions& options) const;
};

}  // namespace myplacement
