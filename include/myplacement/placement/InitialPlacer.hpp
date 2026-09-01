#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "myplacement/metrics/Metrics.hpp"

namespace myplacement {

enum class InitialMethod { Random, Clustering, Quadratic };

std::string toString(InitialMethod method);
InitialMethod parseInitialMethod(const std::string& value);

struct InitialPlacementOptions {
    std::uint64_t seed = 2026;
    std::size_t target_cluster_size = 128;
    std::size_t maximum_cluster_size = 256;
    int cluster_relaxation_iterations = 24;
    int quadratic_outer_iterations = 5;
    int quadratic_solver_iterations = 100;
    double quadratic_tolerance = 1e-5;
    double minimum_distance = 1.0;
    std::size_t clique_degree_limit = 96;
};

struct InitialPlacementResult {
    InitialMethod method = InitialMethod::Random;
    double hpwl_before = 0.0;
    double hpwl_after = 0.0;
    double elapsed_seconds = 0.0;
    int iterations = 0;
};

class InitialPlacer {
public:
    InitialPlacementResult run(PlacementDatabase& database, InitialMethod method,
                               const InitialPlacementOptions& options) const;
};

}  // namespace myplacement
