#include "../CudaPlacementBackend.hpp"

#include "myplacement/placement/CudaDevicePolicy.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace myplacement::detail {
namespace {

constexpr int kCudaThreads = 256;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kDeviceInfinity = 1.79769313486231570814527423731704357e308;
constexpr std::size_t kSharedGpuReserveBytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;

// ---- Explicit CUDA resource ownership and allocation accounting ----------------

[[noreturn]] void throwCudaError(const char* operation, cudaError_t status) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throwCudaError(operation, status);
}

void checkCufft(cufftResult status, const char* operation) {
    if (status != CUFFT_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with cuFFT status " +
                                 std::to_string(static_cast<int>(status)) + ".");
    }
}

[[nodiscard]] std::size_t checkedMultiply(std::size_t left, std::size_t right, const char* description) {
    if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::invalid_argument(std::string("CUDA allocation size overflow for ") + description + ".");
    }
    return left * right;
}

[[nodiscard]] std::size_t checkedBytes(std::size_t count, std::size_t element_size, const char* description) {
    return checkedMultiply(count, element_size, description);
}

[[nodiscard]] std::size_t ceilDivide(std::size_t value, std::size_t divisor) {
    return value / divisor + (value % divisor == 0U ? 0U : 1U);
}

class DeviceMemoryBudget {
public:
    explicit DeviceMemoryBudget(std::size_t limit_bytes) : limit_bytes_(limit_bytes) {}

    void reserve(std::size_t bytes, const char* description) {
        if (bytes > limit_bytes_ - reserved_bytes_) {
            std::ostringstream message;
            message << "CUDA allocation budget exceeded while reserving " << description << " (requested " << bytes
                    << " bytes, reserved " << reserved_bytes_ << " bytes, limit " << limit_bytes_ << " bytes).";
            throw std::runtime_error(message.str());
        }
        reserved_bytes_ += bytes;
    }

    [[nodiscard]] std::size_t reservedBytes() const { return reserved_bytes_; }

private:
    std::size_t limit_bytes_ = 0U;
    std::size_t reserved_bytes_ = 0U;
};

template <typename Value>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() {
        if (data_ != nullptr) cudaFree(data_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void allocate(std::size_t count, DeviceMemoryBudget& budget, const char* description) {
        if (count == 0U) return;
        if (data_ != nullptr) throw std::logic_error("A CUDA buffer was allocated twice.");
        const std::size_t bytes = checkedBytes(count, sizeof(Value), description);
        budget.reserve(bytes, description);
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&data_), bytes), description);
        count_ = count;
    }

    void copyFromHost(const std::vector<Value>& values, const char* description) {
        if (values.size() != count_) throw std::invalid_argument("CUDA host/device buffer size mismatch.");
        if (!values.empty()) {
            checkCuda(cudaMemcpy(data_, values.data(), bytes(), cudaMemcpyHostToDevice), description);
        }
    }

    void copyToHost(std::vector<Value>& values, const char* description) const {
        values.resize(count_);
        if (!values.empty()) {
            checkCuda(cudaMemcpy(values.data(), data_, bytes(), cudaMemcpyDeviceToHost), description);
        }
    }

    void zero(const char* description) {
        if (data_ != nullptr) checkCuda(cudaMemset(data_, 0, bytes()), description);
    }

    [[nodiscard]] Value* data() { return data_; }
    [[nodiscard]] const Value* data() const { return data_; }
    [[nodiscard]] std::size_t count() const { return count_; }
    [[nodiscard]] std::size_t bytes() const { return checkedBytes(count_, sizeof(Value), "CUDA buffer"); }

private:
    Value* data_ = nullptr;
    std::size_t count_ = 0U;
};

struct DeviceVec2 {
    double x;
    double y;
};

static_assert(sizeof(DeviceVec2) == sizeof(Vec2), "Host and CUDA position layouts must agree.");

struct NetBounds {
    double maximum_x;
    double minimum_x;
    double maximum_y;
    double minimum_y;
};

struct NetSums {
    double positive_x;
    double negative_x;
    double positive_y;
    double negative_y;
};

enum TransformKind : int { kInverseDct = 0, kInverseDst = 1 };

// ---- Wirelength and density kernels ------------------------------------------------

__device__ inline double deviceClamp(double value, double lower, double upper) {
    return fmax(lower, fmin(value, upper));
}

__device__ inline int reflectIndex(int index, int size, bool odd, bool& negate) {
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

__device__ double sampleWithParity(const double* values, double x, double y, int columns, int rows,
                                   double region_left, double region_bottom, double region_right,
                                   double region_top, double bin_width, double bin_height,
                                   bool odd_x, bool odd_y) {
    x = deviceClamp(x, region_left, region_right);
    y = deviceClamp(y, region_bottom, region_top);
    const double grid_x = (x - region_left) / bin_width - 0.5;
    const double grid_y = (y - region_bottom) / bin_height - 0.5;
    const int left = static_cast<int>(floor(grid_x));
    const int bottom = static_cast<int>(floor(grid_y));
    const double tx = grid_x - static_cast<double>(left);
    const double ty = grid_y - static_cast<double>(bottom);
    const auto value_at = [&](int column, int row) {
        bool negate = false;
        const int reflected_column = reflectIndex(column, columns, odd_x, negate);
        const int reflected_row = reflectIndex(row, rows, odd_y, negate);
        const double value = values[reflected_row * columns + reflected_column];
        return negate ? -value : value;
    };
    const double lower = value_at(left, bottom) * (1.0 - tx) + value_at(left + 1, bottom) * tx;
    const double upper = value_at(left, bottom + 1) * (1.0 - tx) + value_at(left + 1, bottom + 1) * tx;
    return lower * (1.0 - ty) + upper * ty;
}

__global__ void buildPinPositionsKernel(DeviceVec2* pin_positions, const int* pin_particles,
                                        const DeviceVec2* pin_offsets_or_static, const DeviceVec2* particles,
                                        std::size_t pin_count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pin_count) return;
    const int particle = pin_particles[index];
    const DeviceVec2 offset_or_static = pin_offsets_or_static[index];
    if (particle < 0) {
        pin_positions[index] = offset_or_static;
    } else {
        const DeviceVec2 center = particles[particle];
        pin_positions[index] = {center.x + offset_or_static.x, center.y + offset_or_static.y};
    }
}

__global__ void computeNetBoundsKernel(const int* net_offsets, const DeviceVec2* pin_positions,
                                       NetBounds* bounds, int net_count) {
    const int net = blockIdx.x;
    const int thread = threadIdx.x;
    if (net >= net_count) return;

    __shared__ double maximum_x[kCudaThreads];
    __shared__ double minimum_x[kCudaThreads];
    __shared__ double maximum_y[kCudaThreads];
    __shared__ double minimum_y[kCudaThreads];

    const int begin = net_offsets[net];
    const int end = net_offsets[net + 1];
    double local_maximum_x = -kDeviceInfinity;
    double local_minimum_x = kDeviceInfinity;
    double local_maximum_y = -kDeviceInfinity;
    double local_minimum_y = kDeviceInfinity;
    for (int pin = begin + thread; pin < end; pin += blockDim.x) {
        const DeviceVec2 position = pin_positions[pin];
        local_maximum_x = fmax(local_maximum_x, position.x);
        local_minimum_x = fmin(local_minimum_x, position.x);
        local_maximum_y = fmax(local_maximum_y, position.y);
        local_minimum_y = fmin(local_minimum_y, position.y);
    }
    maximum_x[thread] = local_maximum_x;
    minimum_x[thread] = local_minimum_x;
    maximum_y[thread] = local_maximum_y;
    minimum_y[thread] = local_minimum_y;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (thread < offset) {
            maximum_x[thread] = fmax(maximum_x[thread], maximum_x[thread + offset]);
            minimum_x[thread] = fmin(minimum_x[thread], minimum_x[thread + offset]);
            maximum_y[thread] = fmax(maximum_y[thread], maximum_y[thread + offset]);
            minimum_y[thread] = fmin(minimum_y[thread], minimum_y[thread + offset]);
        }
        __syncthreads();
    }
    if (thread == 0) {
        if (end - begin < 2) {
            bounds[net] = {0.0, 0.0, 0.0, 0.0};
        } else {
            bounds[net] = {maximum_x[0], minimum_x[0], maximum_y[0], minimum_y[0]};
        }
    }
}

__global__ void computeNetSumsKernel(const int* net_offsets, const DeviceVec2* pin_positions,
                                     const NetBounds* bounds, const double* net_weights, double smoothing,
                                     NetSums* sums, double* smooth_values, double* hpwl_values, int net_count) {
    const int net = blockIdx.x;
    const int thread = threadIdx.x;
    if (net >= net_count) return;
    const int begin = net_offsets[net];
    const int end = net_offsets[net + 1];
    if (end - begin < 2) {
        if (thread == 0) {
            sums[net] = {0.0, 0.0, 0.0, 0.0};
            smooth_values[net] = 0.0;
            hpwl_values[net] = 0.0;
        }
        return;
    }

    __shared__ double positive_x[kCudaThreads];
    __shared__ double negative_x[kCudaThreads];
    __shared__ double positive_y[kCudaThreads];
    __shared__ double negative_y[kCudaThreads];
    const NetBounds bound = bounds[net];
    double local_positive_x = 0.0;
    double local_negative_x = 0.0;
    double local_positive_y = 0.0;
    double local_negative_y = 0.0;
    for (int pin = begin + thread; pin < end; pin += blockDim.x) {
        const DeviceVec2 position = pin_positions[pin];
        local_positive_x += exp((position.x - bound.maximum_x) / smoothing);
        local_negative_x += exp((bound.minimum_x - position.x) / smoothing);
        local_positive_y += exp((position.y - bound.maximum_y) / smoothing);
        local_negative_y += exp((bound.minimum_y - position.y) / smoothing);
    }
    positive_x[thread] = local_positive_x;
    negative_x[thread] = local_negative_x;
    positive_y[thread] = local_positive_y;
    negative_y[thread] = local_negative_y;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (thread < offset) {
            positive_x[thread] += positive_x[thread + offset];
            negative_x[thread] += negative_x[thread + offset];
            positive_y[thread] += positive_y[thread + offset];
            negative_y[thread] += negative_y[thread + offset];
        }
        __syncthreads();
    }
    if (thread == 0) {
        const NetSums result{positive_x[0], negative_x[0], positive_y[0], negative_y[0]};
        sums[net] = result;
        const double weight = net_weights[net];
        smooth_values[net] = weight *
                             (bound.maximum_x + smoothing * log(result.positive_x) - bound.minimum_x +
                              smoothing * log(result.negative_x) + bound.maximum_y +
                              smoothing * log(result.positive_y) - bound.minimum_y +
                              smoothing * log(result.negative_y));
        hpwl_values[net] = weight *
                           ((bound.maximum_x - bound.minimum_x) + (bound.maximum_y - bound.minimum_y));
    }
}

__global__ void computePinGradientKernel(const DeviceVec2* pin_positions, const int* pin_particles,
                                         const int* pin_nets, const int* net_offsets, const NetBounds* bounds,
                                         const NetSums* sums, const double* net_weights, double smoothing,
                                         DeviceVec2* gradients, std::size_t pin_count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pin_count) return;
    const int particle = pin_particles[index];
    if (particle < 0) return;
    const int net = pin_nets[index];
    if (net_offsets[net + 1] - net_offsets[net] < 2) return;
    const DeviceVec2 position = pin_positions[index];
    const NetBounds bound = bounds[net];
    const NetSums sum = sums[net];
    const double weight = net_weights[net];
    const double gradient_x = exp((position.x - bound.maximum_x) / smoothing) / sum.positive_x -
                              exp((bound.minimum_x - position.x) / smoothing) / sum.negative_x;
    const double gradient_y = exp((position.y - bound.maximum_y) / smoothing) / sum.positive_y -
                              exp((bound.minimum_y - position.y) / smoothing) / sum.negative_y;
    atomicAdd(&gradients[particle].x, weight * gradient_x);
    atomicAdd(&gradients[particle].y, weight * gradient_y);
}

__global__ void depositParticlesKernel(const DeviceVec2* positions, const double* widths, const double* heights,
                                       const int* layers, std::size_t particle_count, double region_left,
                                       double region_bottom, double region_right, double region_top,
                                       double bin_width, double bin_height, int columns, int rows,
                                       double* movable, double* macro, double* filler) {
    const std::size_t particle = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (particle >= particle_count) return;
    const DeviceVec2 center = positions[particle];
    const double left = fmax(region_left, center.x - widths[particle] * 0.5);
    const double right = fmin(region_right, center.x + widths[particle] * 0.5);
    const double bottom = fmax(region_bottom, center.y - heights[particle] * 0.5);
    const double top = fmin(region_top, center.y + heights[particle] * 0.5);
    if (right <= left || top <= bottom) return;

    const int start_column = max(0, min(columns - 1, static_cast<int>(floor((left - region_left) / bin_width))));
    const int end_column = max(0, min(columns - 1,
                                      static_cast<int>(ceil((right - region_left) / bin_width) - 1.0)));
    const int start_row = max(0, min(rows - 1, static_cast<int>(floor((bottom - region_bottom) / bin_height))));
    const int end_row = max(0, min(rows - 1, static_cast<int>(ceil((top - region_bottom) / bin_height) - 1.0)));
    for (int row = start_row; row <= end_row; ++row) {
        const double bin_bottom = region_bottom + static_cast<double>(row) * bin_height;
        const double bin_top = bin_bottom + bin_height;
        const double overlap_y = fmax(0.0, fmin(top, bin_top) - fmax(bottom, bin_bottom));
        if (overlap_y <= 0.0) continue;
        for (int column = start_column; column <= end_column; ++column) {
            const double bin_left = region_left + static_cast<double>(column) * bin_width;
            const double bin_right = bin_left + bin_width;
            const double overlap_x = fmax(0.0, fmin(right, bin_right) - fmax(left, bin_left));
            const double area = overlap_x * overlap_y;
            if (area <= 0.0) continue;
            const int index = row * columns + column;
            if (layers[particle] == 0) atomicAdd(&movable[index], area);
            if (layers[particle] == 1) atomicAdd(&macro[index], area);
            if (layers[particle] == 2) atomicAdd(&filler[index], area);
        }
    }
}

__global__ void buildDensityMetricsKernel(const double* fixed, const double* dark, const double* movable,
                                          const double* macro, const double* filler, double target_density,
                                          double bin_area, std::size_t bin_count, double* deviation,
                                          double* charge_with_fillers, double* charge_without_fillers,
                                          double* normalization, double* overflow_with_fillers,
                                          double* overflow_without_fillers, double* density_with_fillers,
                                          double* density_without_fillers) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= bin_count) return;
    const double base = movable[index] + target_density * (fixed[index] + macro[index] + dark[index]);
    const double without_fillers = base;
    const double with_fillers = base + target_density * filler[index];
    const double capacity = target_density * bin_area;
    charge_with_fillers[index] = with_fillers;
    charge_without_fillers[index] = without_fillers;
    normalization[index] = movable[index] + target_density * macro[index];
    overflow_with_fillers[index] = fmax(0.0, with_fillers - capacity);
    overflow_without_fillers[index] = fmax(0.0, without_fillers - capacity);
    density_with_fillers[index] = with_fillers / bin_area;
    density_without_fillers[index] = without_fillers / bin_area;
    deviation[index] = density_with_fillers[index] - target_density;
}

__global__ void subtractMeanKernel(double* values, std::size_t count, double mean) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) values[index] -= mean;
}

// ---- DCT/DST Neumann field: cuFFT symmetry adapters --------------------------------

__global__ void expandEvenRowsKernel(const double* input, cufftDoubleComplex* spectrum, int rows, int columns) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(2 * columns);
    if (index >= count) return;
    const int row = static_cast<int>(index / static_cast<std::size_t>(2 * columns));
    const int frequency = static_cast<int>(index % static_cast<std::size_t>(2 * columns));
    const int source_column = frequency < columns ? frequency : 2 * columns - frequency - 1;
    spectrum[index] = {input[row * columns + source_column], 0.0};
}

__global__ void extractDctRowsKernel(const cufftDoubleComplex* spectrum, double* output, int rows, int columns) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns);
    if (index >= count) return;
    const int row = static_cast<int>(index / static_cast<std::size_t>(columns));
    const int frequency = static_cast<int>(index % static_cast<std::size_t>(columns));
    const double phase = kPi * static_cast<double>(frequency) / static_cast<double>(2 * columns);
    const cufftDoubleComplex value = spectrum[row * (2 * columns) + frequency];
    output[index] = cos(phase) * value.x + sin(phase) * value.y;
}

__global__ void expandEvenColumnsKernel(const double* input, cufftDoubleComplex* spectrum, int rows, int columns) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(columns) * static_cast<std::size_t>(2 * rows);
    if (index >= count) return;
    const int column = static_cast<int>(index / static_cast<std::size_t>(2 * rows));
    const int frequency = static_cast<int>(index % static_cast<std::size_t>(2 * rows));
    const int source_row = frequency < rows ? frequency : 2 * rows - frequency - 1;
    spectrum[index] = {input[source_row * columns + column], 0.0};
}

__global__ void extractDctColumnsKernel(const cufftDoubleComplex* spectrum, double* output, int rows,
                                        int columns) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns);
    if (index >= count) return;
    const int row = static_cast<int>(index / static_cast<std::size_t>(columns));
    const int column = static_cast<int>(index % static_cast<std::size_t>(columns));
    const double phase = kPi * static_cast<double>(row) / static_cast<double>(2 * rows);
    const cufftDoubleComplex value = spectrum[column * (2 * rows) + row];
    output[index] = cos(phase) * value.x + sin(phase) * value.y;
}

__device__ cufftDoubleComplex inverseSpectrumValue(const double* coefficients, int base, int frequency,
                                                    int length, int kind) {
    const int doubled_length = 2 * length;
    if (kind == kInverseDct) {
        if (frequency == 0) return {coefficients[base], 0.0};
        if (frequency < length) {
            const double value = coefficients[base + frequency];
            const double phase = kPi * static_cast<double>(frequency) / static_cast<double>(doubled_length);
            return {value * cos(phase), value * sin(phase)};
        }
        if (frequency == length) return {0.0, 0.0};
        const int mirror = doubled_length - frequency;
        const double value = coefficients[base + mirror];
        const double phase = kPi * static_cast<double>(mirror) / static_cast<double>(doubled_length);
        return {value * cos(phase), -value * sin(phase)};
    }

    if (frequency == 0) return {0.0, 0.0};
    if (frequency < length) {
        const double value = coefficients[base + frequency - 1];
        const double phase = kPi * static_cast<double>(frequency) / static_cast<double>(doubled_length);
        return {value * sin(phase), -value * cos(phase)};
    }
    if (frequency == length) return {coefficients[base + length - 1], 0.0};
    const int mirror = doubled_length - frequency;
    const double value = coefficients[base + mirror - 1];
    const double phase = kPi * static_cast<double>(mirror) / static_cast<double>(doubled_length);
    return {value * sin(phase), value * cos(phase)};
}

__global__ void prepareInverseRowsKernel(const double* input, cufftDoubleComplex* spectrum, int rows,
                                         int columns, int kind) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(2 * columns);
    if (index >= count) return;
    const int row = static_cast<int>(index / static_cast<std::size_t>(2 * columns));
    const int frequency = static_cast<int>(index % static_cast<std::size_t>(2 * columns));
    spectrum[index] = inverseSpectrumValue(input, row * columns, frequency, columns, kind);
}

__global__ void extractInverseRowsKernel(const cufftDoubleComplex* spectrum, double* output, int rows,
                                         int columns) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns);
    if (index >= count) return;
    const int row = static_cast<int>(index / static_cast<std::size_t>(columns));
    const int column = static_cast<int>(index % static_cast<std::size_t>(columns));
    output[index] = spectrum[row * (2 * columns) + column].x;
}

__global__ void prepareInverseColumnsKernel(const double* input, cufftDoubleComplex* spectrum, int rows,
                                            int columns, int kind) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(columns) * static_cast<std::size_t>(2 * rows);
    if (index >= count) return;
    const int column = static_cast<int>(index / static_cast<std::size_t>(2 * rows));
    const int frequency = static_cast<int>(index % static_cast<std::size_t>(2 * rows));
    const int doubled_rows = 2 * rows;
    cufftDoubleComplex value{0.0, 0.0};
    if (kind == kInverseDct) {
        if (frequency == 0) {
            value = {input[column], 0.0};
        } else if (frequency < rows) {
            const double coefficient = input[frequency * columns + column];
            const double phase = kPi * static_cast<double>(frequency) / static_cast<double>(doubled_rows);
            value = {coefficient * cos(phase), coefficient * sin(phase)};
        } else if (frequency > rows) {
            const int mirror = doubled_rows - frequency;
            const double coefficient = input[mirror * columns + column];
            const double phase = kPi * static_cast<double>(mirror) / static_cast<double>(doubled_rows);
            value = {coefficient * cos(phase), -coefficient * sin(phase)};
        }
    } else if (frequency > 0 && frequency < rows) {
        const double coefficient = input[(frequency - 1) * columns + column];
        const double phase = kPi * static_cast<double>(frequency) / static_cast<double>(doubled_rows);
        value = {coefficient * sin(phase), -coefficient * cos(phase)};
    } else if (frequency == rows) {
        value = {input[(rows - 1) * columns + column], 0.0};
    } else if (frequency > rows) {
        const int mirror = doubled_rows - frequency;
        const double coefficient = input[(mirror - 1) * columns + column];
        const double phase = kPi * static_cast<double>(mirror) / static_cast<double>(doubled_rows);
        value = {coefficient * sin(phase), coefficient * cos(phase)};
    }
    spectrum[index] = value;
}

__global__ void extractInverseColumnsKernel(const cufftDoubleComplex* spectrum, double* output, int rows,
                                            int columns) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns);
    if (index >= count) return;
    const int row = static_cast<int>(index / static_cast<std::size_t>(columns));
    const int column = static_cast<int>(index % static_cast<std::size_t>(columns));
    output[index] = spectrum[column * (2 * rows) + row].x;
}

__global__ void solveSpectralKernel(const double* coefficients, double* potential_coefficients,
                                    double* field_x_coefficients, double* field_y_coefficients, int columns,
                                    int rows, double region_width, double region_height) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(columns) * static_cast<std::size_t>(rows);
    if (index >= count) return;
    const int row = static_cast<int>(index / static_cast<std::size_t>(columns));
    const int column = static_cast<int>(index % static_cast<std::size_t>(columns));
    const double wave_x = kPi * static_cast<double>(column) / region_width;
    const double wave_y = kPi * static_cast<double>(row) / region_height;
    const double denominator = wave_x * wave_x + wave_y * wave_y;
    if (denominator <= 1e-9) {
        potential_coefficients[index] = 0.0;
        return;
    }
    const double normalization = 1.0 / (4.0 * static_cast<double>(columns) * static_cast<double>(rows));
    const double potential = coefficients[index] * normalization / denominator;
    potential_coefficients[index] = potential;
    if (column > 0) field_x_coefficients[row * columns + column - 1] = potential * wave_x;
    if (row > 0) field_y_coefficients[(row - 1) * columns + column] = potential * wave_y;
}

__global__ void computeEnergyKernel(const double* centered_source, const double* potential, double bin_area,
                                    double* energy_terms, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) energy_terms[index] = 0.5 * centered_source[index] * potential[index] * bin_area;
}

__global__ void sampleDensityGradientKernel(const DeviceVec2* positions, const double* charge_areas,
                                            const double* field_x, const double* field_y,
                                            DeviceVec2* gradients, std::size_t particle_count, int columns,
                                            int rows, double region_left, double region_bottom,
                                            double region_right, double region_top, double bin_width,
                                            double bin_height) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= particle_count) return;
    const DeviceVec2 position = positions[index];
    const double electric_x = sampleWithParity(field_x, position.x, position.y, columns, rows, region_left,
                                               region_bottom, region_right, region_top, bin_width, bin_height,
                                               true, false);
    const double electric_y = sampleWithParity(field_y, position.x, position.y, columns, rows, region_left,
                                               region_bottom, region_right, region_top, bin_width, bin_height,
                                               false, true);
    gradients[index] = {-charge_areas[index] * electric_x, -charge_areas[index] * electric_y};
}

__global__ void reduceSumKernel(const double* input, double* output, std::size_t count) {
    extern __shared__ double shared[];
    const unsigned int thread = threadIdx.x;
    const std::size_t first = static_cast<std::size_t>(blockIdx.x) * blockDim.x * 2U + thread;
    double sum = first < count ? input[first] : 0.0;
    const std::size_t second = first + blockDim.x;
    if (second < count) sum += input[second];
    shared[thread] = sum;
    __syncthreads();
    for (unsigned int offset = blockDim.x / 2U; offset > 0U; offset >>= 1U) {
        if (thread < offset) shared[thread] += shared[thread + offset];
        __syncthreads();
    }
    if (thread == 0U) output[blockIdx.x] = shared[0];
}

__global__ void reduceMaxKernel(const double* input, double* output, std::size_t count) {
    extern __shared__ double shared[];
    const unsigned int thread = threadIdx.x;
    const std::size_t first = static_cast<std::size_t>(blockIdx.x) * blockDim.x * 2U + thread;
    double maximum = first < count ? input[first] : -kDeviceInfinity;
    const std::size_t second = first + blockDim.x;
    if (second < count) maximum = fmax(maximum, input[second]);
    shared[thread] = maximum;
    __syncthreads();
    for (unsigned int offset = blockDim.x / 2U; offset > 0U; offset >>= 1U) {
        if (thread < offset) shared[thread] = fmax(shared[thread], shared[thread + offset]);
        __syncthreads();
    }
    if (thread == 0U) output[blockIdx.x] = shared[0];
}

class CudaNeumannField {
public:
    CudaNeumannField(int columns, int rows, Rect region, DeviceMemoryBudget& budget)
        : columns_(columns), rows_(rows), region_(region), count_(static_cast<std::size_t>(columns) * rows) {
        row_spectrum_.allocate(static_cast<std::size_t>(rows_) * static_cast<std::size_t>(2 * columns_), budget,
                               "CUDA DCT row spectrum");
        column_spectrum_.allocate(static_cast<std::size_t>(columns_) * static_cast<std::size_t>(2 * rows_), budget,
                                  "CUDA DCT column spectrum");
        coefficients_.allocate(count_, budget, "CUDA DCT coefficients");
        potential_coefficients_.allocate(count_, budget, "CUDA potential coefficients");
        field_x_coefficients_.allocate(count_, budget, "CUDA horizontal field coefficients");
        field_y_coefficients_.allocate(count_, budget, "CUDA vertical field coefficients");
        inverse_intermediate_.allocate(count_, budget, "CUDA inverse DCT intermediate");
        potential_.allocate(count_, budget, "CUDA potential grid");
        field_x_.allocate(count_, budget, "CUDA horizontal field grid");
        field_y_.allocate(count_, budget, "CUDA vertical field grid");
        try {
            checkCufft(cufftCreate(&row_plan_), "Create CUDA row DCT handle");
            checkCufft(cufftSetAutoAllocation(row_plan_, 0), "Disable CUDA row DCT auto-allocation");
            std::size_t row_workspace_bytes = 0U;
            checkCufft(cufftMakePlan1d(row_plan_, 2 * columns_, CUFFT_Z2Z, rows_, &row_workspace_bytes),
                       "Plan CUDA row DCT transform");
            row_workspace_.allocate(row_workspace_bytes, budget, "CUDA row DCT workspace");
            if (row_workspace_bytes > 0U) {
                checkCufft(cufftSetWorkArea(row_plan_, row_workspace_.data()), "Set CUDA row DCT workspace");
            }

            checkCufft(cufftCreate(&column_plan_), "Create CUDA column DCT handle");
            checkCufft(cufftSetAutoAllocation(column_plan_, 0), "Disable CUDA column DCT auto-allocation");
            std::size_t column_workspace_bytes = 0U;
            checkCufft(cufftMakePlan1d(column_plan_, 2 * rows_, CUFFT_Z2Z, columns_, &column_workspace_bytes),
                       "Plan CUDA column DCT transform");
            column_workspace_.allocate(column_workspace_bytes, budget, "CUDA column DCT workspace");
            if (column_workspace_bytes > 0U) {
                checkCufft(cufftSetWorkArea(column_plan_, column_workspace_.data()),
                           "Set CUDA column DCT workspace");
            }
        } catch (...) {
            if (row_plan_ != 0) cufftDestroy(row_plan_);
            if (column_plan_ != 0) cufftDestroy(column_plan_);
            row_plan_ = 0;
            column_plan_ = 0;
            throw;
        }
    }

    ~CudaNeumannField() {
        if (row_plan_ != 0) cufftDestroy(row_plan_);
        if (column_plan_ != 0) cufftDestroy(column_plan_);
    }

    CudaNeumannField(const CudaNeumannField&) = delete;
    CudaNeumannField& operator=(const CudaNeumannField&) = delete;

    void solve(const double* centered_source) {
        forwardDct(centered_source);
        field_x_coefficients_.zero("Clear CUDA horizontal field coefficients");
        field_y_coefficients_.zero("Clear CUDA vertical field coefficients");
        const dim3 grid(static_cast<unsigned int>(ceilDivide(count_, kCudaThreads)));
        solveSpectralKernel<<<grid, kCudaThreads>>>(coefficients_.data(), potential_coefficients_.data(),
                                                     field_x_coefficients_.data(), field_y_coefficients_.data(),
                                                     columns_, rows_, region_.width(), region_.height());
        checkCuda(cudaGetLastError(), "Solve CUDA Neumann spectrum");
        inverse2d(potential_coefficients_.data(), kInverseDct, kInverseDct, potential_.data());
        inverse2d(field_x_coefficients_.data(), kInverseDct, kInverseDst, field_x_.data());
        inverse2d(field_y_coefficients_.data(), kInverseDst, kInverseDct, field_y_.data());
    }

    [[nodiscard]] const double* potential() const { return potential_.data(); }
    [[nodiscard]] const double* fieldX() const { return field_x_.data(); }
    [[nodiscard]] const double* fieldY() const { return field_y_.data(); }

private:
    void forwardDct(const double* input) {
        const std::size_t row_count = static_cast<std::size_t>(rows_) * static_cast<std::size_t>(2 * columns_);
        const std::size_t grid_count = count_;
        expandEvenRowsKernel<<<static_cast<unsigned int>(ceilDivide(row_count, kCudaThreads)), kCudaThreads>>>(
            input, row_spectrum_.data(), rows_, columns_);
        checkCuda(cudaGetLastError(), "Expand CUDA DCT rows");
        checkCufft(cufftExecZ2Z(row_plan_, row_spectrum_.data(), row_spectrum_.data(), CUFFT_FORWARD),
                   "Execute CUDA row DCT transform");
        extractDctRowsKernel<<<static_cast<unsigned int>(ceilDivide(grid_count, kCudaThreads)), kCudaThreads>>>(
            row_spectrum_.data(), inverse_intermediate_.data(), rows_, columns_);
        checkCuda(cudaGetLastError(), "Extract CUDA DCT rows");

        const std::size_t column_count = static_cast<std::size_t>(columns_) * static_cast<std::size_t>(2 * rows_);
        expandEvenColumnsKernel<<<static_cast<unsigned int>(ceilDivide(column_count, kCudaThreads)), kCudaThreads>>>(
            inverse_intermediate_.data(), column_spectrum_.data(), rows_, columns_);
        checkCuda(cudaGetLastError(), "Expand CUDA DCT columns");
        checkCufft(cufftExecZ2Z(column_plan_, column_spectrum_.data(), column_spectrum_.data(), CUFFT_FORWARD),
                   "Execute CUDA column DCT transform");
        extractDctColumnsKernel<<<static_cast<unsigned int>(ceilDivide(grid_count, kCudaThreads)), kCudaThreads>>>(
            column_spectrum_.data(), coefficients_.data(), rows_, columns_);
        checkCuda(cudaGetLastError(), "Extract CUDA DCT columns");
    }

    void inverse2d(const double* coefficients, int vertical_kind, int horizontal_kind, double* output) {
        const std::size_t column_count = static_cast<std::size_t>(columns_) * static_cast<std::size_t>(2 * rows_);
        prepareInverseColumnsKernel<<<static_cast<unsigned int>(ceilDivide(column_count, kCudaThreads)),
                                       kCudaThreads>>>(coefficients, column_spectrum_.data(), rows_, columns_,
                                                       vertical_kind);
        checkCuda(cudaGetLastError(), "Prepare CUDA inverse column transform");
        checkCufft(cufftExecZ2Z(column_plan_, column_spectrum_.data(), column_spectrum_.data(), CUFFT_INVERSE),
                   "Execute CUDA inverse column transform");
        extractInverseColumnsKernel<<<static_cast<unsigned int>(ceilDivide(count_, kCudaThreads)), kCudaThreads>>>(
            column_spectrum_.data(), inverse_intermediate_.data(), rows_, columns_);
        checkCuda(cudaGetLastError(), "Extract CUDA inverse column transform");

        const std::size_t row_count = static_cast<std::size_t>(rows_) * static_cast<std::size_t>(2 * columns_);
        prepareInverseRowsKernel<<<static_cast<unsigned int>(ceilDivide(row_count, kCudaThreads)), kCudaThreads>>>(
            inverse_intermediate_.data(), row_spectrum_.data(), rows_, columns_, horizontal_kind);
        checkCuda(cudaGetLastError(), "Prepare CUDA inverse row transform");
        checkCufft(cufftExecZ2Z(row_plan_, row_spectrum_.data(), row_spectrum_.data(), CUFFT_INVERSE),
                   "Execute CUDA inverse row transform");
        extractInverseRowsKernel<<<static_cast<unsigned int>(ceilDivide(count_, kCudaThreads)), kCudaThreads>>>(
            row_spectrum_.data(), output, rows_, columns_);
        checkCuda(cudaGetLastError(), "Extract CUDA inverse row transform");
    }

    int columns_ = 0;
    int rows_ = 0;
    Rect region_;
    std::size_t count_ = 0U;
    cufftHandle row_plan_ = 0;
    cufftHandle column_plan_ = 0;
    DeviceBuffer<cufftDoubleComplex> row_spectrum_;
    DeviceBuffer<cufftDoubleComplex> column_spectrum_;
    DeviceBuffer<unsigned char> row_workspace_;
    DeviceBuffer<unsigned char> column_workspace_;
    DeviceBuffer<double> coefficients_;
    DeviceBuffer<double> potential_coefficients_;
    DeviceBuffer<double> field_x_coefficients_;
    DeviceBuffer<double> field_y_coefficients_;
    DeviceBuffer<double> inverse_intermediate_;
    DeviceBuffer<double> potential_;
    DeviceBuffer<double> field_x_;
    DeviceBuffer<double> field_y_;
};

struct HostLayout {
    std::vector<DeviceVec2> pin_offsets_or_static;
    std::vector<int> pin_particles;
    std::vector<int> pin_nets;
    std::vector<int> net_offsets;
    std::vector<double> net_weights;
    std::vector<double> particle_widths;
    std::vector<double> particle_heights;
    std::vector<double> particle_charge_areas;
    std::vector<int> particle_layers;
};

// ---- Host-side backend orchestration ------------------------------------------------

void requireIntCount(std::size_t value, const char* description) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("CUDA backend cannot index ") + description + " larger than INT_MAX.");
    }
}

[[nodiscard]] HostLayout buildHostLayout(const PlacementDatabase& database, const std::vector<ModuleId>& movable,
                                         const std::vector<DensityFiller>& fillers, double target_density) {
    requireIntCount(database.modules.size(), "modules");
    requireIntCount(database.pins.size(), "pins");
    requireIntCount(database.nets.size(), "nets");
    requireIntCount(movable.size() + fillers.size(), "particles");

    HostLayout layout;
    const std::size_t particle_count = movable.size() + fillers.size();
    std::vector<int> module_particle(database.modules.size(), -1);
    for (std::size_t index = 0; index < movable.size(); ++index) {
        module_particle[movable[index]] = static_cast<int>(index);
        const Module& module = database.modules[movable[index]];
        layout.particle_widths.push_back(module.width);
        layout.particle_heights.push_back(module.height);
        layout.particle_charge_areas.push_back(densityChargeArea(module, target_density));
        layout.particle_layers.push_back(module.is_macro ? 1 : 0);
    }
    for (const DensityFiller& filler : fillers) {
        layout.particle_widths.push_back(filler.width);
        layout.particle_heights.push_back(filler.height);
        layout.particle_charge_areas.push_back(densityChargeArea(filler, target_density));
        layout.particle_layers.push_back(2);
    }
    if (layout.particle_widths.size() != particle_count) {
        throw std::logic_error("CUDA particle layout construction failed.");
    }

    layout.net_offsets.reserve(database.nets.size() + 1U);
    layout.net_offsets.push_back(0);
    layout.net_weights.reserve(database.nets.size());
    layout.pin_offsets_or_static.reserve(database.pins.size());
    layout.pin_particles.reserve(database.pins.size());
    layout.pin_nets.reserve(database.pins.size());
    for (std::size_t net_index = 0; net_index < database.nets.size(); ++net_index) {
        const Net& net = database.nets[net_index];
        layout.net_weights.push_back(net.weight);
        for (const PinId pin_id : net.pins) {
            const Pin& pin = database.pins[pin_id];
            const Module& module = database.modules[pin.module];
            const int particle = module_particle[pin.module];
            if (particle >= 0) {
                const Vec2 offset = transformOffset(pin.offset, module.orientation);
                layout.pin_offsets_or_static.push_back({offset.x, offset.y});
            } else {
                const Vec2 position = database.pinPosition(pin_id);
                layout.pin_offsets_or_static.push_back({position.x, position.y});
            }
            layout.pin_particles.push_back(particle);
            layout.pin_nets.push_back(static_cast<int>(net_index));
        }
        requireIntCount(layout.pin_particles.size(), "flattened net pins");
        layout.net_offsets.push_back(static_cast<int>(layout.pin_particles.size()));
    }
    return layout;
}

class CudaPlacementBackendImpl final : public CudaPlacementBackend {
public:
    CudaPlacementBackendImpl(const PlacementDatabase& database, const std::vector<ModuleId>& movable,
                             const std::vector<DensityFiller>& fillers, const DensityMap& static_density,
                             const GlobalPlacementOptions& options, int device, std::size_t allocation_limit)
        : device_(device), columns_(static_density.columns()), rows_(static_density.rows()),
          particle_count_(movable.size() + fillers.size()), bin_count_(static_density.bins().size()),
          region_(static_density.region()), target_density_(options.target_density), budget_(allocation_limit) {
        checkCuda(cudaSetDevice(device_), "Select CUDA device");
        if (static_density.targetDensity() != target_density_) {
            throw std::invalid_argument("CUDA density map target does not match global-placement options.");
        }
        const HostLayout layout = buildHostLayout(database, movable, fillers, target_density_);
        pin_count_ = layout.pin_particles.size();
        net_count_ = layout.net_weights.size();
        requireIntCount(bin_count_, "density bins");

        positions_.allocate(particle_count_, budget_, "CUDA particle positions");
        particle_widths_.allocate(particle_count_, budget_, "CUDA particle widths");
        particle_heights_.allocate(particle_count_, budget_, "CUDA particle heights");
        particle_charge_areas_.allocate(particle_count_, budget_, "CUDA particle charge areas");
        particle_layers_.allocate(particle_count_, budget_, "CUDA particle layers");
        wire_gradient_.allocate(particle_count_, budget_, "CUDA wire gradients");
        density_gradient_.allocate(particle_count_, budget_, "CUDA density gradients");

        pin_offsets_or_static_.allocate(pin_count_, budget_, "CUDA pin offsets");
        pin_particles_.allocate(pin_count_, budget_, "CUDA pin-particle map");
        pin_nets_.allocate(pin_count_, budget_, "CUDA pin-net map");
        pin_positions_.allocate(pin_count_, budget_, "CUDA pin positions");
        net_offsets_.allocate(net_count_ + 1U, budget_, "CUDA net offsets");
        net_weights_.allocate(net_count_, budget_, "CUDA net weights");
        net_bounds_.allocate(net_count_, budget_, "CUDA net bounds");
        net_sums_.allocate(net_count_, budget_, "CUDA net sums");
        net_smooth_.allocate(net_count_, budget_, "CUDA smooth wirelength values");
        net_hpwl_.allocate(net_count_, budget_, "CUDA HPWL values");

        fixed_density_.allocate(bin_count_, budget_, "CUDA fixed density layer");
        dark_density_.allocate(bin_count_, budget_, "CUDA dark density layer");
        movable_density_.allocate(bin_count_, budget_, "CUDA movable density layer");
        macro_density_.allocate(bin_count_, budget_, "CUDA macro density layer");
        filler_density_.allocate(bin_count_, budget_, "CUDA filler density layer");
        deviation_.allocate(bin_count_, budget_, "CUDA density deviation");
        charge_with_fillers_.allocate(bin_count_, budget_, "CUDA charge with fillers");
        charge_without_fillers_.allocate(bin_count_, budget_, "CUDA charge without fillers");
        normalization_.allocate(bin_count_, budget_, "CUDA density normalization");
        overflow_with_fillers_.allocate(bin_count_, budget_, "CUDA overflow with fillers");
        overflow_without_fillers_.allocate(bin_count_, budget_, "CUDA overflow without fillers");
        density_with_fillers_.allocate(bin_count_, budget_, "CUDA density with fillers");
        density_without_fillers_.allocate(bin_count_, budget_, "CUDA density without fillers");
        energy_terms_.allocate(bin_count_, budget_, "CUDA density energy terms");

        const std::size_t reduction_input = std::max(bin_count_, net_count_);
        const std::size_t reduction_count = std::max<std::size_t>(1U, ceilDivide(reduction_input, 2U * kCudaThreads));
        reduction_a_.allocate(reduction_count, budget_, "CUDA reduction workspace A");
        reduction_b_.allocate(reduction_count, budget_, "CUDA reduction workspace B");

        particle_widths_.copyFromHost(layout.particle_widths, "Upload CUDA particle widths");
        particle_heights_.copyFromHost(layout.particle_heights, "Upload CUDA particle heights");
        particle_charge_areas_.copyFromHost(layout.particle_charge_areas, "Upload CUDA particle charge areas");
        particle_layers_.copyFromHost(layout.particle_layers, "Upload CUDA particle layers");
        pin_offsets_or_static_.copyFromHost(layout.pin_offsets_or_static, "Upload CUDA pin offsets");
        pin_particles_.copyFromHost(layout.pin_particles, "Upload CUDA pin-particle map");
        pin_nets_.copyFromHost(layout.pin_nets, "Upload CUDA pin-net map");
        net_offsets_.copyFromHost(layout.net_offsets, "Upload CUDA net offsets");
        net_weights_.copyFromHost(layout.net_weights, "Upload CUDA net weights");

        std::vector<double> fixed(bin_count_);
        std::vector<double> dark(bin_count_);
        for (std::size_t index = 0; index < bin_count_; ++index) {
            const DensityBin& bin = static_density.bins()[index];
            fixed[index] = bin.fixed_area;
            dark[index] = bin.dark_area;
            static_dark_area_ += bin.dark_area;
            static_placeable_area_ += bin.region.area() - bin.dark_area;
        }
        fixed_density_.copyFromHost(fixed, "Upload CUDA fixed density layer");
        dark_density_.copyFromHost(dark, "Upload CUDA dark density layer");
        field_ = std::make_unique<CudaNeumannField>(columns_, rows_, region_, budget_);
    }

    [[nodiscard]] CudaPlacementEvaluation evaluate(const std::vector<Vec2>& particle_positions, double smoothing,
                                                    bool calculate_gradient) override {
        if (particle_positions.size() != particle_count_) {
            throw std::invalid_argument("CUDA placement positions do not match the particle count.");
        }
        if (!std::isfinite(smoothing) || smoothing <= 0.0) {
            throw std::invalid_argument("CUDA global placement requires positive finite smoothing.");
        }
        checkCuda(cudaSetDevice(device_), "Select CUDA device for evaluation");
        if (!particle_positions.empty()) {
            checkCuda(cudaMemcpy(positions_.data(), particle_positions.data(), positions_.bytes(), cudaMemcpyHostToDevice),
                      "Upload CUDA particle positions");
        }

        evaluateWirelength(smoothing, calculate_gradient);
        CudaPlacementEvaluation result;
        result.smooth_wirelength = reduceSum(net_smooth_.data(), net_count_);
        result.hpwl = reduceSum(net_hpwl_.data(), net_count_);
        evaluateDensity(result);
        if (calculate_gradient) {
            evaluateDensityGradient();
            copyGradients(result);
        }
        checkCuda(cudaDeviceSynchronize(), "Synchronize CUDA placement evaluation");
        if (!std::isfinite(result.smooth_wirelength) || !std::isfinite(result.hpwl) ||
            !std::isfinite(result.electrostatic_energy)) {
            throw std::runtime_error("CUDA global-placement evaluation produced a non-finite metric.");
        }
        return result;
    }

    [[nodiscard]] int device() const override { return device_; }
    [[nodiscard]] std::size_t reservedBytes() const override { return budget_.reservedBytes(); }

private:
    void evaluateWirelength(double smoothing, bool calculate_gradient) {
        if (pin_count_ == 0U || net_count_ == 0U) return;
        buildPinPositionsKernel<<<static_cast<unsigned int>(ceilDivide(pin_count_, kCudaThreads)), kCudaThreads>>>(
            pin_positions_.data(), pin_particles_.data(), pin_offsets_or_static_.data(), positions_.data(), pin_count_);
        checkCuda(cudaGetLastError(), "Build CUDA pin positions");
        computeNetBoundsKernel<<<static_cast<unsigned int>(net_count_), kCudaThreads>>>(
            net_offsets_.data(), pin_positions_.data(), net_bounds_.data(), static_cast<int>(net_count_));
        checkCuda(cudaGetLastError(), "Compute CUDA net bounds");
        computeNetSumsKernel<<<static_cast<unsigned int>(net_count_), kCudaThreads>>>(
            net_offsets_.data(), pin_positions_.data(), net_bounds_.data(), net_weights_.data(), smoothing,
            net_sums_.data(), net_smooth_.data(), net_hpwl_.data(), static_cast<int>(net_count_));
        checkCuda(cudaGetLastError(), "Compute CUDA wirelength sums");
        if (!calculate_gradient) return;
        wire_gradient_.zero("Clear CUDA wire gradients");
        computePinGradientKernel<<<static_cast<unsigned int>(ceilDivide(pin_count_, kCudaThreads)), kCudaThreads>>>(
            pin_positions_.data(), pin_particles_.data(), pin_nets_.data(), net_offsets_.data(), net_bounds_.data(),
            net_sums_.data(), net_weights_.data(), smoothing, wire_gradient_.data(), pin_count_);
        checkCuda(cudaGetLastError(), "Compute CUDA wire gradients");
    }

    void evaluateDensity(CudaPlacementEvaluation& result) {
        movable_density_.zero("Clear CUDA movable density layer");
        macro_density_.zero("Clear CUDA macro density layer");
        filler_density_.zero("Clear CUDA filler density layer");
        if (particle_count_ > 0U) {
            depositParticlesKernel<<<static_cast<unsigned int>(ceilDivide(particle_count_, kCudaThreads)),
                                     kCudaThreads>>>(
                positions_.data(), particle_widths_.data(), particle_heights_.data(), particle_layers_.data(),
                particle_count_, region_.ll.x, region_.ll.y, region_.ur.x, region_.ur.y,
                region_.width() / static_cast<double>(columns_), region_.height() / static_cast<double>(rows_),
                columns_, rows_, movable_density_.data(), macro_density_.data(), filler_density_.data());
            checkCuda(cudaGetLastError(), "Deposit CUDA density particles");
        }
        const double bin_area = region_.area() / static_cast<double>(bin_count_);
        buildDensityMetricsKernel<<<static_cast<unsigned int>(ceilDivide(bin_count_, kCudaThreads)), kCudaThreads>>>(
            fixed_density_.data(), dark_density_.data(), movable_density_.data(), macro_density_.data(),
            filler_density_.data(), target_density_, bin_area, bin_count_, deviation_.data(),
            charge_with_fillers_.data(), charge_without_fillers_.data(), normalization_.data(),
            overflow_with_fillers_.data(), overflow_without_fillers_.data(), density_with_fillers_.data(),
            density_without_fillers_.data());
        checkCuda(cudaGetLastError(), "Build CUDA density metrics");

        result.optimizer_density = collectDensityMetrics(true);
        result.design_density = collectDensityMetrics(false);
        const double mean = reduceSum(deviation_.data(), bin_count_) / static_cast<double>(bin_count_);
        subtractMeanKernel<<<static_cast<unsigned int>(ceilDivide(bin_count_, kCudaThreads)), kCudaThreads>>>(
            deviation_.data(), bin_count_, mean);
        checkCuda(cudaGetLastError(), "Remove CUDA density DC component");
        field_->solve(deviation_.data());
        computeEnergyKernel<<<static_cast<unsigned int>(ceilDivide(bin_count_, kCudaThreads)), kCudaThreads>>>(
            deviation_.data(), field_->potential(), bin_area, energy_terms_.data(), bin_count_);
        checkCuda(cudaGetLastError(), "Compute CUDA density energy");
        result.electrostatic_energy = std::max(0.0, reduceSum(energy_terms_.data(), bin_count_));
    }

    void evaluateDensityGradient() {
        if (particle_count_ == 0U) return;
        sampleDensityGradientKernel<<<static_cast<unsigned int>(ceilDivide(particle_count_, kCudaThreads)),
                                      kCudaThreads>>>(
            positions_.data(), particle_charge_areas_.data(), field_->fieldX(), field_->fieldY(),
            density_gradient_.data(), particle_count_, columns_, rows_, region_.ll.x, region_.ll.y, region_.ur.x,
            region_.ur.y, region_.width() / static_cast<double>(columns_),
            region_.height() / static_cast<double>(rows_));
        checkCuda(cudaGetLastError(), "Sample CUDA density gradients");
    }

    [[nodiscard]] DensityMetrics collectDensityMetrics(bool include_fillers) {
        DensityMetrics result;
        result.total_overflow_area = reduceSum(
            include_fillers ? overflow_with_fillers_.data() : overflow_without_fillers_.data(), bin_count_);
        result.total_charge_area = reduceSum(
            include_fillers ? charge_with_fillers_.data() : charge_without_fillers_.data(), bin_count_);
        result.normalization_area = reduceSum(normalization_.data(), bin_count_);
        result.maximum_density = reduceMax(
            include_fillers ? density_with_fillers_.data() : density_without_fillers_.data(), bin_count_);
        result.average_density = reduceSum(
            include_fillers ? density_with_fillers_.data() : density_without_fillers_.data(), bin_count_) /
                                 static_cast<double>(bin_count_);
        result.dark_area = static_dark_area_;
        result.placeable_area = static_placeable_area_;
        result.normalized_overflow = result.total_overflow_area / std::max(result.normalization_area, kEpsilon);
        return result;
    }

    void copyGradients(CudaPlacementEvaluation& result) const {
        std::vector<DeviceVec2> wire(particle_count_);
        std::vector<DeviceVec2> density(particle_count_);
        if (particle_count_ > 0U) {
            checkCuda(cudaMemcpy(wire.data(), wire_gradient_.data(), wire_gradient_.bytes(), cudaMemcpyDeviceToHost),
                      "Download CUDA wire gradients");
            checkCuda(cudaMemcpy(density.data(), density_gradient_.data(), density_gradient_.bytes(),
                                 cudaMemcpyDeviceToHost),
                      "Download CUDA density gradients");
        }
        result.wire_gradient.resize(particle_count_);
        result.density_gradient.resize(particle_count_);
        for (std::size_t index = 0; index < particle_count_; ++index) {
            result.wire_gradient[index] = {wire[index].x, wire[index].y};
            result.density_gradient[index] = {density[index].x, density[index].y};
        }
    }

    [[nodiscard]] double reduceSum(const double* input, std::size_t count) {
        if (count == 0U) return 0.0;
        const double* current = input;
        std::size_t current_count = count;
        bool use_first = true;
        while (current_count > 1U) {
            const std::size_t blocks = ceilDivide(current_count, 2U * kCudaThreads);
            DeviceBuffer<double>& output = use_first ? reduction_a_ : reduction_b_;
            reduceSumKernel<<<static_cast<unsigned int>(blocks), kCudaThreads,
                              static_cast<std::size_t>(kCudaThreads) * sizeof(double)>>>(current, output.data(),
                                                                                          current_count);
            checkCuda(cudaGetLastError(), "Reduce CUDA sum");
            current = output.data();
            current_count = blocks;
            use_first = !use_first;
        }
        double result = 0.0;
        checkCuda(cudaMemcpy(&result, current, sizeof(double), cudaMemcpyDeviceToHost), "Download CUDA sum");
        return result;
    }

    [[nodiscard]] double reduceMax(const double* input, std::size_t count) {
        if (count == 0U) return 0.0;
        const double* current = input;
        std::size_t current_count = count;
        bool use_first = true;
        while (current_count > 1U) {
            const std::size_t blocks = ceilDivide(current_count, 2U * kCudaThreads);
            DeviceBuffer<double>& output = use_first ? reduction_a_ : reduction_b_;
            reduceMaxKernel<<<static_cast<unsigned int>(blocks), kCudaThreads,
                              static_cast<std::size_t>(kCudaThreads) * sizeof(double)>>>(current, output.data(),
                                                                                          current_count);
            checkCuda(cudaGetLastError(), "Reduce CUDA maximum");
            current = output.data();
            current_count = blocks;
            use_first = !use_first;
        }
        double result = 0.0;
        checkCuda(cudaMemcpy(&result, current, sizeof(double), cudaMemcpyDeviceToHost),
                  "Download CUDA maximum");
        return result;
    }

    int device_ = -1;
    int columns_ = 0;
    int rows_ = 0;
    std::size_t particle_count_ = 0U;
    std::size_t pin_count_ = 0U;
    std::size_t net_count_ = 0U;
    std::size_t bin_count_ = 0U;
    Rect region_;
    double target_density_ = 0.0;
    double static_dark_area_ = 0.0;
    double static_placeable_area_ = 0.0;
    DeviceMemoryBudget budget_;

    DeviceBuffer<DeviceVec2> positions_;
    DeviceBuffer<double> particle_widths_;
    DeviceBuffer<double> particle_heights_;
    DeviceBuffer<double> particle_charge_areas_;
    DeviceBuffer<int> particle_layers_;
    DeviceBuffer<DeviceVec2> wire_gradient_;
    DeviceBuffer<DeviceVec2> density_gradient_;
    DeviceBuffer<DeviceVec2> pin_offsets_or_static_;
    DeviceBuffer<int> pin_particles_;
    DeviceBuffer<int> pin_nets_;
    DeviceBuffer<DeviceVec2> pin_positions_;
    DeviceBuffer<int> net_offsets_;
    DeviceBuffer<double> net_weights_;
    DeviceBuffer<NetBounds> net_bounds_;
    DeviceBuffer<NetSums> net_sums_;
    DeviceBuffer<double> net_smooth_;
    DeviceBuffer<double> net_hpwl_;
    DeviceBuffer<double> fixed_density_;
    DeviceBuffer<double> dark_density_;
    DeviceBuffer<double> movable_density_;
    DeviceBuffer<double> macro_density_;
    DeviceBuffer<double> filler_density_;
    DeviceBuffer<double> deviation_;
    DeviceBuffer<double> charge_with_fillers_;
    DeviceBuffer<double> charge_without_fillers_;
    DeviceBuffer<double> normalization_;
    DeviceBuffer<double> overflow_with_fillers_;
    DeviceBuffer<double> overflow_without_fillers_;
    DeviceBuffer<double> density_with_fillers_;
    DeviceBuffer<double> density_without_fillers_;
    DeviceBuffer<double> energy_terms_;
    DeviceBuffer<double> reduction_a_;
    DeviceBuffer<double> reduction_b_;
    std::unique_ptr<CudaNeumannField> field_;
};

}  // namespace

bool cudaPlacementBackendCompiled() {
    return true;
}

std::unique_ptr<CudaPlacementBackend> tryCreateCudaPlacementBackend(
    const PlacementDatabase& database, const std::vector<ModuleId>& movable,
    const std::vector<DensityFiller>& fillers, const DensityMap& static_density,
    const GlobalPlacementOptions& options, std::string& reason) {
    reason.clear();
    if (options.density_field_boundary != DensityFieldBoundary::Neumann) {
        reason = "The CUDA backend currently implements the physical Neumann/DCT field only.";
        return nullptr;
    }
    if (!isPermittedCudaDevice(options.cuda_device)) {
        reason = "CUDA device must be in the shared-server allowance 1 through 4 or 7.";
        return nullptr;
    }

    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess) {
        reason = std::string("CUDA device discovery failed: ") + cudaGetErrorString(count_status);
        return nullptr;
    }
    if (options.cuda_device >= device_count) {
        reason = "Requested CUDA device does not exist on this host.";
        return nullptr;
    }
    if (cudaSetDevice(options.cuda_device) != cudaSuccess) {
        reason = "Unable to select the requested CUDA device.";
        return nullptr;
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, options.cuda_device) != cudaSuccess) {
        reason = "Unable to query the requested CUDA device.";
        return nullptr;
    }
    if (properties.major < 6) {
        reason = "Requested CUDA device lacks required double-precision atomic support.";
        return nullptr;
    }
    std::size_t free_bytes = 0U;
    std::size_t total_bytes = 0U;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        reason = "Unable to query free CUDA memory.";
        return nullptr;
    }
    if (free_bytes <= kSharedGpuReserveBytes) {
        reason = "Requested CUDA device does not have the required 4 GiB shared-memory reserve.";
        return nullptr;
    }
    const std::size_t allocation_limit = std::min(options.maximum_cuda_memory_bytes,
                                                  free_bytes - kSharedGpuReserveBytes);
    if (allocation_limit == 0U) {
        reason = "CUDA memory limit leaves no allocatable device memory.";
        return nullptr;
    }
    try {
        return std::make_unique<CudaPlacementBackendImpl>(database, movable, fillers, static_density, options,
                                                          options.cuda_device, allocation_limit);
    } catch (const std::exception& error) {
        reason = error.what();
        return nullptr;
    }
}

}  // namespace myplacement::detail
