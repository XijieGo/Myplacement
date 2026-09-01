#include "TestAssertions.hpp"

#include "myplacement/metrics/Metrics.hpp"
#include "myplacement/placement/ElectrostaticField.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace myplacement::test {

void runDensityFieldTests() {
    constexpr int columns = 16;
    constexpr int rows = 12;
    constexpr int mode_x = 2;
    constexpr int mode_y = 3;
    constexpr double pi = 3.141592653589793238462643383279502884;
    const Rect region{{-3.0, 2.0}, {5.0, 8.0}};
    const double bin_width = region.width() / static_cast<double>(columns);
    const double bin_height = region.height() / static_cast<double>(rows);
    const double wave_x = pi * static_cast<double>(mode_x) / region.width();
    const double wave_y = pi * static_cast<double>(mode_y) / region.height();
    const double denominator = wave_x * wave_x + wave_y * wave_y;

    ElectrostaticField field(columns, rows, region, DensityFieldBoundary::Neumann);
    std::vector<double> source(static_cast<std::size_t>(columns * rows));
    for (int row = 0; row < rows; ++row) {
        const double y = region.ll.y + (static_cast<double>(row) + 0.5) * bin_height;
        for (int column = 0; column < columns; ++column) {
            const double x = region.ll.x + (static_cast<double>(column) + 0.5) * bin_width;
            source[static_cast<std::size_t>(row * columns + column)] =
                std::cos(wave_x * (x - region.ll.x)) * std::cos(wave_y * (y - region.ll.y));
        }
    }
    field.solve(source);

    for (int row = 0; row < rows; ++row) {
        const double y = region.ll.y + (static_cast<double>(row) + 0.5) * bin_height;
        for (int column = 0; column < columns; ++column) {
            const double x = region.ll.x + (static_cast<double>(column) + 0.5) * bin_width;
            const double cosine_x = std::cos(wave_x * (x - region.ll.x));
            const double cosine_y = std::cos(wave_y * (y - region.ll.y));
            const Vec2 electric = field.fieldAt(column, row);
            expectNear(field.potentialAt(column, row), cosine_x * cosine_y / denominator, 2e-12,
                       "DCT potential does not match its analytical Neumann mode.");
            expectNear(electric.x, wave_x * std::sin(wave_x * (x - region.ll.x)) * cosine_y / denominator, 2e-12,
                       "DCT horizontal field has an incorrect coefficient or sign.");
            expectNear(electric.y, wave_y * cosine_x * std::sin(wave_y * (y - region.ll.y)) / denominator, 2e-12,
                       "DCT vertical field has an incorrect coefficient or sign.");
        }
    }

    for (int row = 0; row < rows; ++row) {
        const double y = region.ll.y + (static_cast<double>(row) + 0.5) * bin_height;
        expectNear(field.sampleField({region.ll.x, y}).x, 0.0, 2e-12,
                   "Neumann interpolation leaked normal field through the left boundary.");
        expectNear(field.sampleField({region.ur.x, y}).x, 0.0, 2e-12,
                   "Neumann interpolation leaked normal field through the right boundary.");
    }
    for (int column = 0; column < columns; ++column) {
        const double x = region.ll.x + (static_cast<double>(column) + 0.5) * bin_width;
        expectNear(field.sampleField({x, region.ll.y}).y, 0.0, 2e-12,
                   "Neumann interpolation leaked normal field through the bottom boundary.");
        expectNear(field.sampleField({x, region.ur.y}).y, 0.0, 2e-12,
                   "Neumann interpolation leaked normal field through the top boundary.");
    }

    const auto verify_axis_mode = [&](int axis_mode_x, int axis_mode_y) {
        const double axis_wave_x = pi * static_cast<double>(axis_mode_x) / region.width();
        const double axis_wave_y = pi * static_cast<double>(axis_mode_y) / region.height();
        const double axis_denominator = axis_wave_x * axis_wave_x + axis_wave_y * axis_wave_y;
        for (int row = 0; row < rows; ++row) {
            const double y = region.ll.y + (static_cast<double>(row) + 0.5) * bin_height;
            for (int column = 0; column < columns; ++column) {
                const double x = region.ll.x + (static_cast<double>(column) + 0.5) * bin_width;
                source[static_cast<std::size_t>(row * columns + column)] =
                    std::cos(axis_wave_x * (x - region.ll.x)) * std::cos(axis_wave_y * (y - region.ll.y));
            }
        }
        field.solve(source);
        for (int row = 0; row < rows; ++row) {
            const double y = region.ll.y + (static_cast<double>(row) + 0.5) * bin_height;
            for (int column = 0; column < columns; ++column) {
                const double x = region.ll.x + (static_cast<double>(column) + 0.5) * bin_width;
                const double cosine_x = std::cos(axis_wave_x * (x - region.ll.x));
                const double cosine_y = std::cos(axis_wave_y * (y - region.ll.y));
                const Vec2 electric = field.fieldAt(column, row);
                expectNear(field.potentialAt(column, row), cosine_x * cosine_y / axis_denominator, 2e-12,
                           "DCT scaling is wrong when one Neumann mode is constant.");
                expectNear(electric.x, axis_wave_x * std::sin(axis_wave_x * (x - region.ll.x)) * cosine_y /
                                               axis_denominator,
                           2e-12, "DCT horizontal field is wrong for an axis mode.");
                expectNear(electric.y, axis_wave_y * cosine_x * std::sin(axis_wave_y * (y - region.ll.y)) /
                                               axis_denominator,
                           2e-12, "DCT vertical field is wrong for an axis mode.");
            }
        }
    };
    verify_axis_mode(0, mode_y);
    verify_axis_mode(mode_x, 0);

    field.solve(std::vector<double>(source.size(), 7.0));
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            const Vec2 electric = field.fieldAt(column, row);
            expectNear(field.potentialAt(column, row), 0.0, 2e-12,
                       "The required DC removal did not zero a uniform density source.");
            expectNear(electric.x, 0.0, 2e-12, "A uniform density source produced horizontal field.");
            expectNear(electric.y, 0.0, 2e-12, "A uniform density source produced vertical field.");
        }
    }

    expect(parseDensityFieldBoundary("closed") == DensityFieldBoundary::Neumann,
           "Closed boundary alias did not select Neumann mode.");
    expect(parseDensityFieldBoundary("fft") == DensityFieldBoundary::Periodic,
           "FFT boundary alias did not select periodic mode.");

    bool rejected_oversized_grid = false;
    try {
        ElectrostaticField oversized(1025, 1024, region);
    } catch (const std::invalid_argument&) {
        rejected_oversized_grid = true;
    }
    expect(rejected_oversized_grid, "The density-field memory guard accepted an oversized grid.");

    bool rejected_oversized_density_map = false;
    try {
        DensityMap oversized_map(region, 1025, 1024, 0.85);
    } catch (const std::invalid_argument&) {
        rejected_oversized_density_map = true;
    }
    expect(rejected_oversized_density_map, "The density-map memory guard accepted an oversized grid.");
}

}  // namespace myplacement::test
