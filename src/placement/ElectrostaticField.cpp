#include "myplacement/placement/ElectrostaticField.hpp"

#include <fftw3.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <stdexcept>

namespace myplacement {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

struct FftwDeleter {
    void operator()(void* pointer) const noexcept {
        if (pointer != nullptr) fftw_free(pointer);
    }
};

class FftwRealArray {
public:
    FftwRealArray() = default;

    explicit FftwRealArray(std::size_t count) : storage_(fftw_alloc_real(count)) {
        if (storage_ == nullptr) throw std::bad_alloc();
        std::fill(data(), data() + count, 0.0);
    }

    [[nodiscard]] double* data() { return static_cast<double*>(storage_.get()); }
    [[nodiscard]] const double* data() const { return static_cast<const double*>(storage_.get()); }

private:
    std::unique_ptr<void, FftwDeleter> storage_;
};

class FftwComplexArray {
public:
    FftwComplexArray() = default;

    explicit FftwComplexArray(std::size_t count) : storage_(fftw_alloc_complex(count)) {
        if (storage_ == nullptr) throw std::bad_alloc();
        std::memset(storage_.get(), 0, count * sizeof(fftw_complex));
    }

    [[nodiscard]] fftw_complex* data() {
        return static_cast<fftw_complex*>(storage_.get());
    }

private:
    std::unique_ptr<void, FftwDeleter> storage_;
};

class FftwPlan {
public:
    FftwPlan() = default;
    ~FftwPlan() {
        if (plan_ != nullptr) fftw_destroy_plan(plan_);
    }

    FftwPlan(const FftwPlan&) = delete;
    FftwPlan& operator=(const FftwPlan&) = delete;

    void reset(fftw_plan plan) {
        if (plan == nullptr) throw std::runtime_error("Unable to create an FFTW density-field plan.");
        if (plan_ != nullptr) fftw_destroy_plan(plan_);
        plan_ = plan;
    }

    void execute() const {
        if (plan_ == nullptr) throw std::logic_error("An uninitialized FFTW plan was executed.");
        fftw_execute(plan_);
    }

private:
    fftw_plan plan_ = nullptr;
};

}  // namespace

class ElectrostaticField::Impl {
public:
    Impl(int columns, int rows, Rect region, DensityFieldBoundary boundary)
        : columns_(columns), rows_(rows), region_(region), boundary_(boundary) {
        if (columns_ < 2 || rows_ < 2 || !region_.valid() || region_.area() <= kEpsilon) {
            throw std::invalid_argument("ElectrostaticField requires a valid region and at least a 2 x 2 grid.");
        }
        const std::size_t columns_size = static_cast<std::size_t>(columns_);
        const std::size_t rows_size = static_cast<std::size_t>(rows_);
        if (columns_size > std::numeric_limits<std::size_t>::max() / rows_size) {
            throw std::invalid_argument("Density grid dimensions overflow size_t.");
        }
        element_count_ = columns_size * rows_size;
        if (element_count_ > ElectrostaticField::kMaximumBinCount) {
            throw std::invalid_argument("Density grid exceeds the safe one-million-bin workspace limit.");
        }
        bin_width_ = region_.width() / static_cast<double>(columns_);
        bin_height_ = region_.height() / static_cast<double>(rows_);
        if (boundary_ == DensityFieldBoundary::Neumann) {
            initializeNeumannPlans();
        } else {
            initializePeriodicPlans();
        }
    }

    void solve(const std::vector<double>& density_deviation) {
        if (density_deviation.size() != element_count_) {
            throw std::invalid_argument("Density-field input does not match the configured grid dimensions.");
        }
        const double mean = std::accumulate(density_deviation.begin(), density_deviation.end(), 0.0) /
                            static_cast<double>(element_count_);
        for (std::size_t index = 0; index < element_count_; ++index) {
            source_.data()[index] = density_deviation[index] - mean;
        }
        if (boundary_ == DensityFieldBoundary::Neumann) {
            solveNeumann();
        } else {
            solvePeriodic();
        }
    }

    [[nodiscard]] Vec2 sampleField(Vec2 position) const {
        if (boundary_ == DensityFieldBoundary::Neumann) {
            return {sampleWithParity(field_x_.data(), position, true, false),
                    sampleWithParity(field_y_.data(), position, false, true)};
        }
        return {sampleClamped(field_x_.data(), position), sampleClamped(field_y_.data(), position)};
    }

    [[nodiscard]] double samplePotential(Vec2 position) const {
        if (boundary_ == DensityFieldBoundary::Neumann) {
            return sampleWithParity(potential_.data(), position, false, false);
        }
        return sampleClamped(potential_.data(), position);
    }

    [[nodiscard]] Vec2 fieldAt(int column, int row) const {
        return {field_x_.data()[flatIndex(column, row)], field_y_.data()[flatIndex(column, row)]};
    }

    [[nodiscard]] double potentialAt(int column, int row) const {
        return potential_.data()[flatIndex(column, row)];
    }

    [[nodiscard]] int columns() const { return columns_; }
    [[nodiscard]] int rows() const { return rows_; }
    [[nodiscard]] const Rect& region() const { return region_; }
    [[nodiscard]] DensityFieldBoundary boundary() const { return boundary_; }

private:
    void initializeNeumannPlans() {
        source_ = FftwRealArray(element_count_);
        coefficients_ = FftwRealArray(element_count_);
        potential_coefficients_ = FftwRealArray(element_count_);
        field_x_coefficients_ = FftwRealArray(element_count_);
        field_y_coefficients_ = FftwRealArray(element_count_);
        potential_ = FftwRealArray(element_count_);
        field_x_ = FftwRealArray(element_count_);
        field_y_ = FftwRealArray(element_count_);

        forward_plan_.reset(fftw_plan_r2r_2d(rows_, columns_, source_.data(), coefficients_.data(),
                                             FFTW_REDFT10, FFTW_REDFT10, FFTW_ESTIMATE));
        potential_plan_.reset(fftw_plan_r2r_2d(rows_, columns_, potential_coefficients_.data(), potential_.data(),
                                               FFTW_REDFT01, FFTW_REDFT01, FFTW_ESTIMATE));
        field_x_plan_.reset(fftw_plan_r2r_2d(rows_, columns_, field_x_coefficients_.data(), field_x_.data(),
                                             FFTW_REDFT01, FFTW_RODFT01, FFTW_ESTIMATE));
        field_y_plan_.reset(fftw_plan_r2r_2d(rows_, columns_, field_y_coefficients_.data(), field_y_.data(),
                                             FFTW_RODFT01, FFTW_REDFT01, FFTW_ESTIMATE));
    }

    void initializePeriodicPlans() {
        source_ = FftwRealArray(element_count_);
        potential_ = FftwRealArray(element_count_);
        field_x_ = FftwRealArray(element_count_);
        field_y_ = FftwRealArray(element_count_);
        spectrum_count_ = static_cast<std::size_t>(rows_) * static_cast<std::size_t>(columns_ / 2 + 1);
        source_spectrum_ = FftwComplexArray(spectrum_count_);
        potential_spectrum_ = FftwComplexArray(spectrum_count_);
        field_x_spectrum_ = FftwComplexArray(spectrum_count_);
        field_y_spectrum_ = FftwComplexArray(spectrum_count_);

        forward_plan_.reset(fftw_plan_dft_r2c_2d(rows_, columns_, source_.data(), source_spectrum_.data(),
                                                 FFTW_ESTIMATE));
        potential_plan_.reset(fftw_plan_dft_c2r_2d(rows_, columns_, potential_spectrum_.data(), potential_.data(),
                                                   FFTW_ESTIMATE));
        field_x_plan_.reset(fftw_plan_dft_c2r_2d(rows_, columns_, field_x_spectrum_.data(), field_x_.data(),
                                                 FFTW_ESTIMATE));
        field_y_plan_.reset(fftw_plan_dft_c2r_2d(rows_, columns_, field_y_spectrum_.data(), field_y_.data(),
                                                 FFTW_ESTIMATE));
    }

    void solveNeumann() {
        forward_plan_.execute();
        std::fill(potential_coefficients_.data(), potential_coefficients_.data() + element_count_, 0.0);
        std::fill(field_x_coefficients_.data(), field_x_coefficients_.data() + element_count_, 0.0);
        std::fill(field_y_coefficients_.data(), field_y_coefficients_.data() + element_count_, 0.0);

        // Two unnormalized DCT-II/DCT-III pairs contribute (2*Nx)*(2*Ny).
        const double normalization = 1.0 / (4.0 * static_cast<double>(columns_) * static_cast<double>(rows_));
        for (int row = 0; row < rows_; ++row) {
            const double wave_y = kPi * static_cast<double>(row) / region_.height();
            for (int column = 0; column < columns_; ++column) {
                const double wave_x = kPi * static_cast<double>(column) / region_.width();
                const double denominator = wave_x * wave_x + wave_y * wave_y;
                if (denominator <= kEpsilon) continue;  // Explicitly remove the DC component.

                const std::size_t index = flatIndex(column, row);
                const double potential_coefficient = coefficients_.data()[index] * normalization / denominator;
                potential_coefficients_.data()[index] = potential_coefficient;
                // DST-III synthesizes sin(k*pi*(i+1/2)/N) from input index k-1.
                if (column > 0) {
                    field_x_coefficients_.data()[flatIndex(column - 1, row)] =
                        potential_coefficient * wave_x;
                }
                if (row > 0) {
                    field_y_coefficients_.data()[flatIndex(column, row - 1)] =
                        potential_coefficient * wave_y;
                }
            }
        }
        potential_plan_.execute();
        field_x_plan_.execute();
        field_y_plan_.execute();
    }

    void solvePeriodic() {
        forward_plan_.execute();
        for (int row = 0; row < rows_; ++row) {
            const int signed_row = row <= rows_ / 2 ? row : row - rows_;
            const double wave_y = 2.0 * kPi * static_cast<double>(signed_row) / region_.height();
            for (int column = 0; column <= columns_ / 2; ++column) {
                const double wave_x = 2.0 * kPi * static_cast<double>(column) / region_.width();
                const std::size_t index = spectrumIndex(column, row);
                const double denominator = wave_x * wave_x + wave_y * wave_y;
                if (denominator <= kEpsilon) {
                    potential_spectrum_.data()[index][0] = 0.0;
                    potential_spectrum_.data()[index][1] = 0.0;
                    field_x_spectrum_.data()[index][0] = 0.0;
                    field_x_spectrum_.data()[index][1] = 0.0;
                    field_y_spectrum_.data()[index][0] = 0.0;
                    field_y_spectrum_.data()[index][1] = 0.0;
                    continue;
                }
                const double potential_real = source_spectrum_.data()[index][0] / denominator;
                const double potential_imaginary = source_spectrum_.data()[index][1] / denominator;
                potential_spectrum_.data()[index][0] = potential_real;
                potential_spectrum_.data()[index][1] = potential_imaginary;
                field_x_spectrum_.data()[index][0] = wave_x * potential_imaginary;
                field_x_spectrum_.data()[index][1] = -wave_x * potential_real;
                field_y_spectrum_.data()[index][0] = wave_y * potential_imaginary;
                field_y_spectrum_.data()[index][1] = -wave_y * potential_real;
            }
        }
        potential_plan_.execute();
        field_x_plan_.execute();
        field_y_plan_.execute();
        const double normalization = 1.0 / static_cast<double>(element_count_);
        for (std::size_t index = 0; index < element_count_; ++index) {
            potential_.data()[index] *= normalization;
            field_x_.data()[index] *= normalization;
            field_y_.data()[index] *= normalization;
        }
    }

    [[nodiscard]] std::size_t flatIndex(int column, int row) const {
        if (column < 0 || column >= columns_ || row < 0 || row >= rows_) {
            throw std::out_of_range("Density-field bin index is out of range.");
        }
        return static_cast<std::size_t>(row * columns_ + column);
    }

    [[nodiscard]] std::size_t spectrumIndex(int column, int row) const {
        return static_cast<std::size_t>(row * (columns_ / 2 + 1) + column);
    }

    [[nodiscard]] int reflectIndex(int index, int size, bool odd, bool& negate) const {
        while (index < 0 || index >= size) {
            if (index < 0) {
                index = -index - 1;
            } else {
                index = 2 * size - index - 1;
            }
            if (odd) negate = !negate;
        }
        return index;
    }

    [[nodiscard]] double sampleWithParity(const double* values, Vec2 position, bool odd_x, bool odd_y) const {
        position.x = clamp(position.x, region_.ll.x, region_.ur.x);
        position.y = clamp(position.y, region_.ll.y, region_.ur.y);
        const double grid_x = (position.x - region_.ll.x) / bin_width_ - 0.5;
        const double grid_y = (position.y - region_.ll.y) / bin_height_ - 0.5;
        const int left = static_cast<int>(std::floor(grid_x));
        const int bottom = static_cast<int>(std::floor(grid_y));
        const double tx = grid_x - static_cast<double>(left);
        const double ty = grid_y - static_cast<double>(bottom);
        const auto valueAt = [&](int column, int row) {
            bool negate = false;
            const int reflected_column = reflectIndex(column, columns_, odd_x, negate);
            const int reflected_row = reflectIndex(row, rows_, odd_y, negate);
            const double value = values[flatIndex(reflected_column, reflected_row)];
            return negate ? -value : value;
        };
        const double lower = valueAt(left, bottom) * (1.0 - tx) + valueAt(left + 1, bottom) * tx;
        const double upper = valueAt(left, bottom + 1) * (1.0 - tx) + valueAt(left + 1, bottom + 1) * tx;
        return lower * (1.0 - ty) + upper * ty;
    }

    [[nodiscard]] double sampleClamped(const double* values, Vec2 position) const {
        position.x = clamp(position.x, region_.ll.x, region_.ur.x);
        position.y = clamp(position.y, region_.ll.y, region_.ur.y);
        const double grid_x = clamp((position.x - region_.ll.x) / bin_width_ - 0.5, 0.0,
                                    static_cast<double>(columns_ - 1));
        const double grid_y = clamp((position.y - region_.ll.y) / bin_height_ - 0.5, 0.0,
                                    static_cast<double>(rows_ - 1));
        const int left = static_cast<int>(std::floor(grid_x));
        const int bottom = static_cast<int>(std::floor(grid_y));
        const int right = std::min(columns_ - 1, left + 1);
        const int top = std::min(rows_ - 1, bottom + 1);
        const double tx = grid_x - static_cast<double>(left);
        const double ty = grid_y - static_cast<double>(bottom);
        const double lower = values[flatIndex(left, bottom)] * (1.0 - tx) +
                             values[flatIndex(right, bottom)] * tx;
        const double upper = values[flatIndex(left, top)] * (1.0 - tx) +
                             values[flatIndex(right, top)] * tx;
        return lower * (1.0 - ty) + upper * ty;
    }

    int columns_ = 0;
    int rows_ = 0;
    Rect region_;
    DensityFieldBoundary boundary_ = DensityFieldBoundary::Neumann;
    double bin_width_ = 0.0;
    double bin_height_ = 0.0;
    std::size_t element_count_ = 0;
    std::size_t spectrum_count_ = 0;

    FftwRealArray source_;
    FftwRealArray coefficients_;
    FftwRealArray potential_coefficients_;
    FftwRealArray field_x_coefficients_;
    FftwRealArray field_y_coefficients_;
    FftwRealArray potential_;
    FftwRealArray field_x_;
    FftwRealArray field_y_;
    FftwComplexArray source_spectrum_;
    FftwComplexArray potential_spectrum_;
    FftwComplexArray field_x_spectrum_;
    FftwComplexArray field_y_spectrum_;
    FftwPlan forward_plan_;
    FftwPlan potential_plan_;
    FftwPlan field_x_plan_;
    FftwPlan field_y_plan_;
};

std::string toString(DensityFieldBoundary boundary) {
    return boundary == DensityFieldBoundary::Neumann ? "neumann" : "periodic";
}

DensityFieldBoundary parseDensityFieldBoundary(const std::string& value) {
    const std::string normalized = lower(value);
    if (normalized == "neumann" || normalized == "closed") return DensityFieldBoundary::Neumann;
    if (normalized == "periodic" || normalized == "fft") return DensityFieldBoundary::Periodic;
    throw std::invalid_argument("Unknown density field boundary: " + value + ". Use neumann or periodic.");
}

ElectrostaticField::ElectrostaticField(int columns, int rows, Rect region, DensityFieldBoundary boundary)
    : impl_(std::make_unique<Impl>(columns, rows, region, boundary)) {}

ElectrostaticField::~ElectrostaticField() = default;
ElectrostaticField::ElectrostaticField(ElectrostaticField&&) noexcept = default;
ElectrostaticField& ElectrostaticField::operator=(ElectrostaticField&&) noexcept = default;

void ElectrostaticField::solve(const std::vector<double>& density_deviation) {
    impl_->solve(density_deviation);
}

Vec2 ElectrostaticField::sampleField(Vec2 position) const {
    return impl_->sampleField(position);
}

double ElectrostaticField::samplePotential(Vec2 position) const {
    return impl_->samplePotential(position);
}

Vec2 ElectrostaticField::fieldAt(int column, int row) const {
    return impl_->fieldAt(column, row);
}

double ElectrostaticField::potentialAt(int column, int row) const {
    return impl_->potentialAt(column, row);
}

int ElectrostaticField::columns() const {
    return impl_->columns();
}

int ElectrostaticField::rows() const {
    return impl_->rows();
}

const Rect& ElectrostaticField::region() const {
    return impl_->region();
}

DensityFieldBoundary ElectrostaticField::boundary() const {
    return impl_->boundary();
}

}  // namespace myplacement
