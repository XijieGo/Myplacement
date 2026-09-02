#include "myplacement/placement/InitialPlacer.hpp"

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace myplacement {
namespace {

using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using Triplet = Eigen::Triplet<double>;

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

bool isKnownInitialMethod(InitialMethod method) {
    switch (method) {
        case InitialMethod::Random:
        case InitialMethod::Clustering:
        case InitialMethod::Quadratic: return true;
    }
    return false;
}

void validateInitialPlacementOptions(InitialMethod method, const InitialPlacementOptions& options) {
    if (!isKnownInitialMethod(method)) {
        throw std::invalid_argument("Unknown initial placement method value.");
    }
    if (method == InitialMethod::Clustering &&
        (options.target_cluster_size == 0U || options.maximum_cluster_size == 0U ||
         options.cluster_relaxation_iterations < 0)) {
        throw std::invalid_argument("Clustering initial-placement options are outside their valid range.");
    }
    if (method == InitialMethod::Quadratic &&
        (options.quadratic_outer_iterations <= 0 || options.quadratic_solver_iterations <= 0 ||
         !std::isfinite(options.quadratic_tolerance) || options.quadratic_tolerance <= 0.0 ||
         !std::isfinite(options.minimum_distance) || options.minimum_distance <= 0.0)) {
        throw std::invalid_argument("Quadratic initial-placement options are outside their valid range.");
    }
}

class DisjointSet {
public:
    explicit DisjointSet(std::size_t count) : parent_(count), size_(count, 1U) {
        std::iota(parent_.begin(), parent_.end(), 0U);
    }

    std::size_t find(std::size_t item) {
        if (parent_[item] != item) parent_[item] = find(parent_[item]);
        return parent_[item];
    }

    [[nodiscard]] std::size_t size(std::size_t item) { return size_[find(item)]; }

    bool unite(std::size_t left, std::size_t right, std::size_t maximum_size) {
        left = find(left);
        right = find(right);
        if (left == right || size_[left] + size_[right] > maximum_size) return false;
        if (size_[left] < size_[right]) std::swap(left, right);
        parent_[right] = left;
        size_[left] += size_[right];
        return true;
    }

private:
    std::vector<std::size_t> parent_;
    std::vector<std::size_t> size_;
};

struct ClusterEdge {
    std::size_t left = 0;
    std::size_t right = 0;
    double weight = 0.0;
};

struct Cluster {
    std::vector<ModuleId> members;
    Vec2 center;
    double area = 0.0;
    double width = 0.0;
    double height = 0.0;
};

void randomPlacement(PlacementDatabase& database, std::uint64_t seed) {
    std::mt19937_64 generator(seed);
    for (const ModuleId id : database.movableModules()) {
        Module& module = database.modules[id];
        const double min_x = database.core_region.ll.x + module.width * 0.5;
        const double max_x = database.core_region.ur.x - module.width * 0.5;
        const double min_y = database.core_region.ll.y + module.height * 0.5;
        const double max_y = database.core_region.ur.y - module.height * 0.5;
        if (min_x <= max_x) {
            std::uniform_real_distribution<double> distribution(min_x, max_x);
            module.center.x = distribution(generator);
        }
        if (min_y <= max_y) {
            std::uniform_real_distribution<double> distribution(min_y, max_y);
            module.center.y = distribution(generator);
        }
        database.clampModuleToCore(id);
    }
}

void clusteringPlacement(PlacementDatabase& database, const InitialPlacementOptions& options) {
    const std::vector<ModuleId>& movable = database.movableModules();
    if (movable.empty()) return;

    std::vector<int> movable_index(database.modules.size(), -1);
    for (std::size_t index = 0; index < movable.size(); ++index) {
        movable_index[movable[index]] = static_cast<int>(index);
    }

    std::vector<ClusterEdge> edges;
    edges.reserve(database.pins.size());
    for (const Net& net : database.nets) {
        std::vector<std::size_t> members;
        members.reserve(net.pins.size());
        for (const PinId pin_id : net.pins) {
            const int index = movable_index[database.pins[pin_id].module];
            if (index >= 0) members.push_back(static_cast<std::size_t>(index));
        }
        std::sort(members.begin(), members.end());
        members.erase(std::unique(members.begin(), members.end()), members.end());
        if (members.size() < 2U) continue;
        const double weight = net.weight / static_cast<double>(members.size() - 1U);
        const std::size_t anchor = members.front();
        for (std::size_t index = 1; index < members.size(); ++index) {
            edges.push_back({anchor, members[index], weight});
        }
    }
    std::sort(edges.begin(), edges.end(), [](const ClusterEdge& left, const ClusterEdge& right) {
        if (left.weight != right.weight) return left.weight > right.weight;
        if (left.left != right.left) return left.left < right.left;
        return left.right < right.right;
    });

    DisjointSet sets(movable.size());
    const std::size_t target_cluster_count =
        std::max<std::size_t>(1U, (movable.size() + options.target_cluster_size - 1U) /
                                      options.target_cluster_size);
    std::size_t cluster_count = movable.size();
    for (const ClusterEdge& edge : edges) {
        if (cluster_count <= target_cluster_count) break;
        if (sets.unite(edge.left, edge.right, options.maximum_cluster_size)) --cluster_count;
    }

    std::unordered_map<std::size_t, std::size_t> root_to_cluster;
    std::vector<Cluster> clusters;
    std::vector<int> cluster_of_module(database.modules.size(), -1);
    for (std::size_t index = 0; index < movable.size(); ++index) {
        const std::size_t root = sets.find(index);
        auto [iterator, inserted] = root_to_cluster.emplace(root, clusters.size());
        if (inserted) clusters.emplace_back();
        const std::size_t cluster_index = iterator->second;
        clusters[cluster_index].members.push_back(movable[index]);
        cluster_of_module[movable[index]] = static_cast<int>(cluster_index);
    }

    const double core_aspect = std::max(database.core_region.width() / database.core_region.height(), 0.1);
    const int cluster_columns = std::max(1, static_cast<int>(std::ceil(std::sqrt(
        static_cast<double>(clusters.size()) * core_aspect))));
    const int cluster_rows = std::max(1, static_cast<int>((clusters.size() +
        static_cast<std::size_t>(cluster_columns) - 1U) / static_cast<std::size_t>(cluster_columns)));
    const double grid_width = database.core_region.width() / static_cast<double>(cluster_columns);
    const double grid_height = database.core_region.height() / static_cast<double>(cluster_rows);

    for (std::size_t index = 0; index < clusters.size(); ++index) {
        Cluster& cluster = clusters[index];
        for (const ModuleId id : cluster.members) cluster.area += database.modules[id].area();
        cluster.width = std::sqrt(std::max(cluster.area * core_aspect, 1.0)) * 1.15;
        cluster.height = std::max(cluster.area / cluster.width, 1.0) * 1.15;
        const int column = static_cast<int>(index % static_cast<std::size_t>(cluster_columns));
        const int row = static_cast<int>(index / static_cast<std::size_t>(cluster_columns));
        cluster.center = {database.core_region.ll.x + (static_cast<double>(column) + 0.5) * grid_width,
                          database.core_region.ll.y + (static_cast<double>(row) + 0.5) * grid_height};
    }

    for (int iteration = 0; iteration < options.cluster_relaxation_iterations; ++iteration) {
        std::vector<Vec2> weighted_targets(clusters.size());
        std::vector<double> weights(clusters.size(), 0.0);
        for (const Net& net : database.nets) {
            std::vector<int> involved_clusters;
            std::vector<Vec2> fixed_positions;
            involved_clusters.reserve(net.pins.size());
            for (const PinId pin_id : net.pins) {
                const Pin& pin = database.pins[pin_id];
                const int cluster = cluster_of_module[pin.module];
                if (cluster >= 0) {
                    involved_clusters.push_back(cluster);
                } else {
                    fixed_positions.push_back(database.pinPosition(pin_id));
                }
            }
            std::sort(involved_clusters.begin(), involved_clusters.end());
            involved_clusters.erase(std::unique(involved_clusters.begin(), involved_clusters.end()),
                                    involved_clusters.end());
            if (involved_clusters.empty()) continue;

            Vec2 total;
            for (const int cluster : involved_clusters) total += clusters[static_cast<std::size_t>(cluster)].center;
            for (const Vec2 fixed : fixed_positions) total += fixed;
            const double count = static_cast<double>(involved_clusters.size() + fixed_positions.size());
            if (count <= 1.0) continue;
            for (const int cluster : involved_clusters) {
                const Vec2 other_total = total - clusters[static_cast<std::size_t>(cluster)].center;
                const Vec2 target = other_total * (1.0 / (count - 1.0));
                const std::size_t cluster_index = static_cast<std::size_t>(cluster);
                weighted_targets[cluster_index] += target * net.weight;
                weights[cluster_index] += net.weight;
            }
        }
        for (std::size_t index = 0; index < clusters.size(); ++index) {
            Cluster& cluster = clusters[index];
            if (weights[index] > kEpsilon) {
                const Vec2 target = weighted_targets[index] * (1.0 / weights[index]);
                cluster.center = cluster.center * 0.68 + target * 0.32;
            }
            cluster.center.x = clamp(cluster.center.x,
                                     database.core_region.ll.x + cluster.width * 0.5,
                                     database.core_region.ur.x - cluster.width * 0.5);
            cluster.center.y = clamp(cluster.center.y,
                                     database.core_region.ll.y + cluster.height * 0.5,
                                     database.core_region.ur.y - cluster.height * 0.5);
        }
    }

    for (Cluster& cluster : clusters) {
        std::sort(cluster.members.begin(), cluster.members.end(), [&](ModuleId left, ModuleId right) {
            return database.modules[left].area() > database.modules[right].area();
        });
        double cursor_x = cluster.center.x - cluster.width * 0.5;
        double cursor_y = cluster.center.y - cluster.height * 0.5;
        double row_height = 0.0;
        for (const ModuleId id : cluster.members) {
            Module& module = database.modules[id];
            if (cursor_x + module.width > cluster.center.x + cluster.width * 0.5 && row_height > 0.0) {
                cursor_x = cluster.center.x - cluster.width * 0.5;
                cursor_y += row_height;
                row_height = 0.0;
            }
            module.center = {cursor_x + module.width * 0.5, cursor_y + module.height * 0.5};
            cursor_x += module.width;
            row_height = std::max(row_height, module.height);
            database.clampModuleToCore(id);
        }
    }
}

void addPairToQuadraticSystem(const PlacementDatabase& database, PinId left_pin_id, PinId right_pin_id,
                              const std::vector<int>& variable_index, double base_weight,
                              double minimum_distance, std::vector<Triplet>& x_triplets,
                              std::vector<Triplet>& y_triplets, Eigen::VectorXd& x_rhs,
                              Eigen::VectorXd& y_rhs) {
    const Pin& left_pin = database.pins[left_pin_id];
    const Pin& right_pin = database.pins[right_pin_id];
    if (left_pin.module == right_pin.module) return;
    const Module& left_module = database.modules[left_pin.module];
    const Module& right_module = database.modules[right_pin.module];
    const int left_index = variable_index[left_pin.module];
    const int right_index = variable_index[right_pin.module];
    if (left_index < 0 && right_index < 0) return;

    const Vec2 left_offset = transformOffset(left_pin.offset, left_module.orientation);
    const Vec2 right_offset = transformOffset(right_pin.offset, right_module.orientation);
    const Vec2 left_position = left_module.center + left_offset;
    const Vec2 right_position = right_module.center + right_offset;
    const double weight_x = base_weight / std::max(std::abs(left_position.x - right_position.x), minimum_distance);
    const double weight_y = base_weight / std::max(std::abs(left_position.y - right_position.y), minimum_distance);

    const auto addDiagonal = [&](int index, double x_weight, double y_weight) {
        x_triplets.emplace_back(index, index, x_weight);
        y_triplets.emplace_back(index, index, y_weight);
    };
    if (left_index >= 0 && right_index >= 0) {
        addDiagonal(left_index, weight_x, weight_y);
        addDiagonal(right_index, weight_x, weight_y);
        x_triplets.emplace_back(left_index, right_index, -weight_x);
        x_triplets.emplace_back(right_index, left_index, -weight_x);
        y_triplets.emplace_back(left_index, right_index, -weight_y);
        y_triplets.emplace_back(right_index, left_index, -weight_y);
        const Vec2 offset_delta = left_offset - right_offset;
        x_rhs[left_index] -= weight_x * offset_delta.x;
        x_rhs[right_index] += weight_x * offset_delta.x;
        y_rhs[left_index] -= weight_y * offset_delta.y;
        y_rhs[right_index] += weight_y * offset_delta.y;
    } else if (left_index >= 0) {
        addDiagonal(left_index, weight_x, weight_y);
        x_rhs[left_index] += weight_x * (right_module.center.x + right_offset.x - left_offset.x);
        y_rhs[left_index] += weight_y * (right_module.center.y + right_offset.y - left_offset.y);
    } else {
        addDiagonal(right_index, weight_x, weight_y);
        x_rhs[right_index] += weight_x * (left_module.center.x + left_offset.x - right_offset.x);
        y_rhs[right_index] += weight_y * (left_module.center.y + left_offset.y - right_offset.y);
    }
}

void quadraticPlacement(PlacementDatabase& database, const InitialPlacementOptions& options, int& completed_iterations) {
    const std::vector<ModuleId>& movable = database.movableModules();
    if (movable.empty()) return;
    const Vec2 core_center = database.core_region.center();
    for (const ModuleId id : movable) {
        database.modules[id].center = core_center;
        database.clampModuleToCore(id);
    }

    std::vector<int> variable_index(database.modules.size(), -1);
    for (std::size_t index = 0; index < movable.size(); ++index) {
        variable_index[movable[index]] = static_cast<int>(index);
    }

    Eigen::VectorXd x_solution(static_cast<Eigen::Index>(movable.size()));
    Eigen::VectorXd y_solution(static_cast<Eigen::Index>(movable.size()));
    for (std::size_t index = 0; index < movable.size(); ++index) {
        x_solution[static_cast<Eigen::Index>(index)] = database.modules[movable[index]].center.x;
        y_solution[static_cast<Eigen::Index>(index)] = database.modules[movable[index]].center.y;
    }

    for (int outer = 0; outer < options.quadratic_outer_iterations; ++outer) {
        std::vector<Triplet> x_triplets;
        std::vector<Triplet> y_triplets;
        x_triplets.reserve(database.pins.size() * 3U);
        y_triplets.reserve(database.pins.size() * 3U);
        Eigen::VectorXd x_rhs = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(movable.size()));
        Eigen::VectorXd y_rhs = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(movable.size()));

        for (const Net& net : database.nets) {
            const std::size_t degree = net.pins.size();
            if (degree < 2U) continue;
            const double clique_weight = 2.0 * net.weight / static_cast<double>(degree - 1U);
            if (degree <= options.clique_degree_limit) {
                for (std::size_t left = 0; left < degree; ++left) {
                    for (std::size_t right = left + 1U; right < degree; ++right) {
                        addPairToQuadraticSystem(database, net.pins[left], net.pins[right], variable_index,
                                                 clique_weight, options.minimum_distance, x_triplets, y_triplets,
                                                 x_rhs, y_rhs);
                    }
                }
            } else {
                const PinId anchor = net.pins.front();
                const double star_weight = net.weight / static_cast<double>(degree - 1U);
                for (std::size_t index = 1; index < degree; ++index) {
                    addPairToQuadraticSystem(database, anchor, net.pins[index], variable_index, star_weight,
                                             options.minimum_distance, x_triplets, y_triplets, x_rhs, y_rhs);
                }
            }
        }

        constexpr double anchor_weight = 1e-4;
        for (std::size_t index = 0; index < movable.size(); ++index) {
            const Eigen::Index eigen_index = static_cast<Eigen::Index>(index);
            const Module& module = database.modules[movable[index]];
            x_triplets.emplace_back(eigen_index, eigen_index, anchor_weight);
            y_triplets.emplace_back(eigen_index, eigen_index, anchor_weight);
            x_rhs[eigen_index] += anchor_weight * module.center.x;
            y_rhs[eigen_index] += anchor_weight * module.center.y;
        }

        SparseMatrix x_matrix(static_cast<Eigen::Index>(movable.size()), static_cast<Eigen::Index>(movable.size()));
        SparseMatrix y_matrix(static_cast<Eigen::Index>(movable.size()), static_cast<Eigen::Index>(movable.size()));
        x_matrix.setFromTriplets(x_triplets.begin(), x_triplets.end());
        y_matrix.setFromTriplets(y_triplets.begin(), y_triplets.end());

        Eigen::BiCGSTAB<SparseMatrix, Eigen::IdentityPreconditioner> solver;
        solver.setMaxIterations(options.quadratic_solver_iterations);
        solver.setTolerance(options.quadratic_tolerance);
        solver.compute(x_matrix);
        x_solution = solver.solveWithGuess(x_rhs, x_solution);
        solver.compute(y_matrix);
        y_solution = solver.solveWithGuess(y_rhs, y_solution);
        if (!x_solution.allFinite() || !y_solution.allFinite()) {
            throw std::runtime_error("Quadratic placement failed to produce finite coordinates.");
        }

        for (std::size_t index = 0; index < movable.size(); ++index) {
            Module& module = database.modules[movable[index]];
            module.center = {x_solution[static_cast<Eigen::Index>(index)],
                             y_solution[static_cast<Eigen::Index>(index)]};
            database.clampModuleToCore(movable[index]);
            x_solution[static_cast<Eigen::Index>(index)] = module.center.x;
            y_solution[static_cast<Eigen::Index>(index)] = module.center.y;
        }
        completed_iterations = outer + 1;
    }
}

}  // namespace

std::string toString(InitialMethod method) {
    switch (method) {
        case InitialMethod::Random: return "random";
        case InitialMethod::Clustering: return "cluster";
        case InitialMethod::Quadratic: return "quadratic";
    }
    return "random";
}

InitialMethod parseInitialMethod(const std::string& value) {
    const std::string method = lower(value);
    if (method == "random") return InitialMethod::Random;
    if (method == "cluster" || method == "clustering") return InitialMethod::Clustering;
    if (method == "quadratic" || method == "quad") return InitialMethod::Quadratic;
    throw std::invalid_argument("Unknown initial placement method: " + value);
}

InitialPlacementResult InitialPlacer::run(PlacementDatabase& database, InitialMethod method,
                                          const InitialPlacementOptions& options) const {
    validateInitialPlacementOptions(method, options);
    const auto started = std::chrono::steady_clock::now();
    InitialPlacementResult result;
    result.method = method;
    result.hpwl_before = calculateHpwl(database).hpwl;
    if (method == InitialMethod::Random) randomPlacement(database, options.seed);
    if (method == InitialMethod::Clustering) clusteringPlacement(database, options);
    if (method == InitialMethod::Quadratic) quadraticPlacement(database, options, result.iterations);
    result.hpwl_after = calculateHpwl(database).hpwl;
    result.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return result;
}

}  // namespace myplacement
