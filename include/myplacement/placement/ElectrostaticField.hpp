#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "myplacement/core/Geometry.hpp"
#include "myplacement/core/ResourceLimits.hpp"

namespace myplacement {

// Periodic is retained only as an A/B baseline. Neumann is the physical default:
// the normal electric field vanishes at the placement-region boundary.
enum class DensityFieldBoundary { Periodic, Neumann };

std::string toString(DensityFieldBoundary boundary);
DensityFieldBoundary parseDensityFieldBoundary(const std::string& value);

// Solves -Laplace(potential) = density on a cell-centered density grid. The input
// may contain a non-zero mean; solve() removes that DC component as required by a
// Neumann Poisson problem. The class owns FFTW plans and is intentionally
// single-threaded, so its workspace stays bounded and reproducible.
class ElectrostaticField {
public:
    // This hard limit keeps the complete density-map + solver workspace well below
    // a gigabyte even for the DCT implementation; it prevents accidental memory
    // exhaustion from an oversized --bins request.
    static constexpr std::size_t kMaximumBinCount = kMaximumDensityBinCount;

    ElectrostaticField(int columns, int rows, Rect region,
                       DensityFieldBoundary boundary = DensityFieldBoundary::Neumann);
    ~ElectrostaticField();

    ElectrostaticField(const ElectrostaticField&) = delete;
    ElectrostaticField& operator=(const ElectrostaticField&) = delete;
    ElectrostaticField(ElectrostaticField&&) noexcept;
    ElectrostaticField& operator=(ElectrostaticField&&) noexcept;

    void solve(const std::vector<double>& density_deviation);

    [[nodiscard]] Vec2 sampleField(Vec2 position) const;
    [[nodiscard]] double samplePotential(Vec2 position) const;
    [[nodiscard]] Vec2 fieldAt(int column, int row) const;
    [[nodiscard]] double potentialAt(int column, int row) const;
    [[nodiscard]] int columns() const;
    [[nodiscard]] int rows() const;
    [[nodiscard]] const Rect& region() const;
    [[nodiscard]] DensityFieldBoundary boundary() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace myplacement
