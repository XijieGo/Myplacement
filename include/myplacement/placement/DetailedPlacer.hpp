#pragma once

#include <cstddef>
#include <string>

#include "myplacement/model/PlacementDatabase.hpp"

namespace myplacement {

// Detailed placement is intentionally a post-legalization stage.  Every
// accepted operation preserves the row, site grid, and occupied row interval,
// so it can improve final HPWL without invalidating the legal solution.
enum class DetailedPlacementMethod { None, AdjacentSwap, WindowReorder };
enum class DetailedPlacementBackend { Cpu, Cuda, Auto };

std::string toString(DetailedPlacementMethod method);
DetailedPlacementMethod parseDetailedPlacementMethod(const std::string& value);
std::string toString(DetailedPlacementBackend backend);
DetailedPlacementBackend parseDetailedPlacementBackend(const std::string& value);

struct DetailedPlacementOptions {
    DetailedPlacementMethod method = DetailedPlacementMethod::None;
    // Two staggered passes cover both window alignments while retaining the
    // quality/latency sweet spot established on the course-scale A1 study.
    int passes = 2;
    // WindowReorder enumerates all orderings of this many consecutive cells.
    // The implementation caps it at six, keeping the exhaustive search bounded
    // while leaving a large, GPU-friendly candidate set on real benchmarks.
    int window_size = 4;
    // High-fanout nets make exhaustive local evaluation disproportionately
    // expensive.  Windows touching one are skipped rather than approximated,
    // which keeps every accepted move an exact HPWL improvement.
    std::size_t maximum_net_degree = 64;
    double improvement_epsilon = 1e-9;
    // CUDA evaluates every permutation of an independent four-cell window in
    // parallel.  CPU remains the exact reference; Auto uses CUDA only when a
    // compatible, safely provisioned device is available.
    DetailedPlacementBackend compute_backend = DetailedPlacementBackend::Cpu;
    int cuda_device = 1;
    std::size_t maximum_cuda_memory_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
};

struct DetailedPlacementResult {
    double hpwl_before = 0.0;
    double hpwl_after = 0.0;
    double elapsed_seconds = 0.0;
    std::size_t evaluated_windows = 0;
    std::size_t evaluated_permutations = 0;
    std::size_t accepted_operations = 0;
    int completed_passes = 0;
    DetailedPlacementBackend compute_backend_used = DetailedPlacementBackend::Cpu;
    int cuda_device_used = -1;
    std::size_t cuda_reserved_memory_bytes = 0;
};

class DetailedPlacer {
public:
    DetailedPlacementResult run(PlacementDatabase& database,
                                const DetailedPlacementOptions& options) const;
};

}  // namespace myplacement
