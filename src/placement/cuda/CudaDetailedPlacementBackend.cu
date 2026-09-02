#include "../CudaDetailedPlacementBackend.hpp"

#include "myplacement/metrics/Metrics.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace myplacement::detail {
namespace {

constexpr int kWindowSize = 4;
constexpr int kPermutationCount = 24;
constexpr int kCudaThreads = 32;
constexpr double kDeviceInfinity = 1.79769313486231570814527423731704357e308;
constexpr std::size_t kSharedGpuReserveBytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;

constexpr int kHostPermutations[kPermutationCount][kWindowSize] = {
    {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {0, 3, 2, 1},
    {1, 0, 2, 3}, {1, 0, 3, 2}, {1, 2, 0, 3}, {1, 2, 3, 0}, {1, 3, 0, 2}, {1, 3, 2, 0},
    {2, 0, 1, 3}, {2, 0, 3, 1}, {2, 1, 0, 3}, {2, 1, 3, 0}, {2, 3, 0, 1}, {2, 3, 1, 0},
    {3, 0, 1, 2}, {3, 0, 2, 1}, {3, 1, 0, 2}, {3, 1, 2, 0}, {3, 2, 0, 1}, {3, 2, 1, 0},
};

__device__ __constant__ int kDevicePermutations[kPermutationCount * kWindowSize] = {
    0, 1, 2, 3, 0, 1, 3, 2, 0, 2, 1, 3, 0, 2, 3, 1, 0, 3, 1, 2, 0, 3, 2, 1,
    1, 0, 2, 3, 1, 0, 3, 2, 1, 2, 0, 3, 1, 2, 3, 0, 1, 3, 0, 2, 1, 3, 2, 0,
    2, 0, 1, 3, 2, 0, 3, 1, 2, 1, 0, 3, 2, 1, 3, 0, 2, 3, 0, 1, 2, 3, 1, 0,
    3, 0, 1, 2, 3, 0, 2, 1, 3, 1, 0, 2, 3, 1, 2, 0, 3, 2, 0, 1, 3, 2, 1, 0,
};

struct DeviceVec2 {
    double x;
    double y;
};

static_assert(sizeof(DeviceVec2) == sizeof(Vec2), "Host and CUDA Vec2 layouts must agree.");

[[noreturn]] void throwCudaError(const char* operation, cudaError_t status) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throwCudaError(operation, status);
}

std::size_t checkedMultiply(std::size_t left, std::size_t right, const char* description) {
    if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::invalid_argument(std::string("CUDA allocation size overflow for ") + description + ".");
    }
    return left * right;
}

class DeviceMemoryBudget {
public:
    explicit DeviceMemoryBudget(std::size_t limit_bytes) : limit_bytes_(limit_bytes) {}

    void reserve(std::size_t bytes, const char* description) {
        if (bytes > limit_bytes_ - reserved_bytes_) {
            throw std::runtime_error(std::string("CUDA detailed-placement budget exceeded while reserving ") +
                                     description + ".");
        }
        reserved_bytes_ += bytes;
        peak_bytes_ = std::max(peak_bytes_, reserved_bytes_);
    }

    void release(std::size_t bytes) {
        reserved_bytes_ -= std::min(bytes, reserved_bytes_);
    }

    [[nodiscard]] std::size_t peakBytes() const { return peak_bytes_; }

private:
    std::size_t limit_bytes_ = 0U;
    std::size_t reserved_bytes_ = 0U;
    std::size_t peak_bytes_ = 0U;
};

template <typename Value>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { reset(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void allocate(std::size_t count, DeviceMemoryBudget& budget, const char* description) {
        if (count == 0U) return;
        if (data_ != nullptr) throw std::logic_error("CUDA detailed-placement buffer allocated twice.");
        const std::size_t bytes = checkedMultiply(count, sizeof(Value), description);
        budget.reserve(bytes, description);
        budget_ = &budget;
        bytes_ = bytes;
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&data_), bytes), description);
        count_ = count;
    }

    void copyFromHost(const std::vector<Value>& values, const char* description) {
        if (values.size() != count_) throw std::invalid_argument("CUDA detailed-placement host/device size mismatch.");
        if (!values.empty()) checkCuda(cudaMemcpy(data_, values.data(), bytes_, cudaMemcpyHostToDevice), description);
    }

    void copyToHost(std::vector<Value>& values, const char* description) const {
        values.resize(count_);
        if (count_ > 0U) checkCuda(cudaMemcpy(values.data(), data_, bytes_, cudaMemcpyDeviceToHost), description);
    }

    [[nodiscard]] Value* data() { return data_; }
    [[nodiscard]] const Value* data() const { return data_; }

private:
    void reset() {
        if (data_ != nullptr) {
            cudaFree(data_);
            if (budget_ != nullptr) budget_->release(bytes_);
        }
        data_ = nullptr;
        count_ = 0U;
        bytes_ = 0U;
        budget_ = nullptr;
    }

    Value* data_ = nullptr;
    std::size_t count_ = 0U;
    std::size_t bytes_ = 0U;
    DeviceMemoryBudget* budget_ = nullptr;
};

__device__ double roundedWidth(double width, double spacing) {
    return spacing <= 1e-9 ? width : ceil((width - 1e-9) / spacing) * spacing;
}

__global__ void evaluateWindowPermutationsKernel(
    const DeviceVec2* module_centers, const double* module_widths, const double* module_heights,
    const int* pin_modules, const DeviceVec2* pin_offsets, const int* net_pin_offsets,
    const int* net_pin_ids, const double* net_weights, const int* window_modules,
    const double* window_start_x, const double* window_row_bottom, const double* window_site_spacing,
    const int* window_net_offsets, const int* window_net_ids, int window_count,
    int* best_permutations) {
    const int window = blockIdx.x;
    const int thread = threadIdx.x;
    if (window >= window_count) return;

    __shared__ double values[kCudaThreads];
    __shared__ int indices[kCudaThreads];

    double value = kDeviceInfinity;
    if (thread < kPermutationCount) {
        const int* modules = window_modules + window * kWindowSize;
        double candidate_x[kWindowSize];
        double candidate_y[kWindowSize];
        double cursor = window_start_x[window];
        const double bottom = window_row_bottom[window];
        const double spacing = window_site_spacing[window];
        for (int order_index = 0; order_index < kWindowSize; ++order_index) {
            const int local_module = kDevicePermutations[thread * kWindowSize + order_index];
            const int module = modules[local_module];
            candidate_x[local_module] = cursor + module_widths[module] * 0.5;
            candidate_y[local_module] = bottom + module_heights[module] * 0.5;
            cursor += roundedWidth(module_widths[module], spacing);
        }

        value = 0.0;
        const int net_begin = window_net_offsets[window];
        const int net_end = window_net_offsets[window + 1];
        for (int net_entry = net_begin; net_entry < net_end; ++net_entry) {
            const int net = window_net_ids[net_entry];
            double minimum_x = kDeviceInfinity;
            double maximum_x = -kDeviceInfinity;
            double minimum_y = kDeviceInfinity;
            double maximum_y = -kDeviceInfinity;
            for (int pin_entry = net_pin_offsets[net]; pin_entry < net_pin_offsets[net + 1]; ++pin_entry) {
                const int pin = net_pin_ids[pin_entry];
                const int module = pin_modules[pin];
                double x = module_centers[module].x;
                double y = module_centers[module].y;
                for (int local = 0; local < kWindowSize; ++local) {
                    if (modules[local] == module) {
                        x = candidate_x[local];
                        y = candidate_y[local];
                        break;
                    }
                }
                x += pin_offsets[pin].x;
                y += pin_offsets[pin].y;
                minimum_x = fmin(minimum_x, x);
                maximum_x = fmax(maximum_x, x);
                minimum_y = fmin(minimum_y, y);
                maximum_y = fmax(maximum_y, y);
            }
            value += net_weights[net] * ((maximum_x - minimum_x) + (maximum_y - minimum_y));
        }
    }
    values[thread] = value;
    indices[thread] = thread;
    __syncthreads();
    for (int stride = kCudaThreads / 2; stride > 0; stride /= 2) {
        if (thread < stride && (values[thread + stride] < values[thread] ||
                                 (values[thread + stride] == values[thread] &&
                                  indices[thread + stride] < indices[thread]))) {
            values[thread] = values[thread + stride];
            indices[thread] = indices[thread + stride];
        }
        __syncthreads();
    }
    if (thread == 0) {
        best_permutations[window] = indices[0];
    }
}

bool isStandardCell(const Module& module) {
    return !module.is_fixed && !module.is_macro;
}

double alignUp(double value, double origin, double step) {
    if (step <= kEpsilon) return value;
    return origin + std::ceil((value - origin - kEpsilon) / step) * step;
}

double roundedWidth(const Module& module, const SiteRow& row) {
    return alignUp(module.width, 0.0, row.site_spacing);
}

std::vector<std::size_t> sortedRows(const PlacementDatabase& database) {
    std::vector<std::size_t> result(database.rows.size());
    std::iota(result.begin(), result.end(), 0U);
    std::sort(result.begin(), result.end(), [&](std::size_t left, std::size_t right) {
        return database.rows[left].bottom < database.rows[right].bottom;
    });
    return result;
}

std::size_t nearestRow(const PlacementDatabase& database, const std::vector<std::size_t>& sorted_rows,
                       double coordinate) {
    const auto iterator = std::lower_bound(sorted_rows.begin(), sorted_rows.end(), coordinate,
        [&](std::size_t row_index, double value) {
            const SiteRow& row = database.rows[row_index];
            return row.bottom + row.height * 0.5 < value;
        });
    if (iterator == sorted_rows.begin()) return *iterator;
    if (iterator == sorted_rows.end()) return sorted_rows.back();
    const std::size_t right = *iterator;
    const std::size_t left = *(iterator - 1);
    const double left_distance = std::abs(database.rows[left].bottom + database.rows[left].height * 0.5 - coordinate);
    const double right_distance =
        std::abs(database.rows[right].bottom + database.rows[right].height * 0.5 - coordinate);
    return left_distance <= right_distance ? left : right;
}

struct RowSegment {
    std::size_t row_index = 0;
    std::vector<ModuleId> modules;
};

std::vector<RowSegment> buildRowSegments(const PlacementDatabase& database, double epsilon) {
    if (database.rows.empty()) return {};
    const std::vector<std::size_t> rows = sortedRows(database);
    std::vector<std::vector<ModuleId>> cells_by_row(database.rows.size());
    for (ModuleId id = 0; id < database.modules.size(); ++id) {
        const Module& module = database.modules[id];
        if (!isStandardCell(module)) continue;
        const std::size_t row_index = nearestRow(database, rows, module.center.y);
        if (std::abs(module.lowerLeft().y - database.rows[row_index].bottom) <= epsilon) {
            cells_by_row[row_index].push_back(id);
        }
    }

    std::vector<RowSegment> result;
    for (std::size_t row_index = 0; row_index < cells_by_row.size(); ++row_index) {
        std::vector<ModuleId>& cells = cells_by_row[row_index];
        if (cells.size() < 2U) continue;
        const SiteRow& row = database.rows[row_index];
        std::sort(cells.begin(), cells.end(), [&](ModuleId left, ModuleId right) {
            return database.modules[left].lowerLeft().x < database.modules[right].lowerLeft().x;
        });
        RowSegment segment;
        segment.row_index = row_index;
        for (const ModuleId id : cells) {
            if (!segment.modules.empty()) {
                const Module& previous = database.modules[segment.modules.back()];
                const double expected = previous.lowerLeft().x + roundedWidth(previous, row);
                if (std::abs(database.modules[id].lowerLeft().x - expected) > epsilon) {
                    if (segment.modules.size() >= 2U) result.push_back(std::move(segment));
                    segment = {};
                    segment.row_index = row_index;
                }
            }
            segment.modules.push_back(id);
        }
        if (segment.modules.size() >= 2U) result.push_back(std::move(segment));
    }
    return result;
}

struct HostWindow {
    std::size_t row_index = 0;
    std::vector<ModuleId> modules;
    std::vector<NetId> nets;
    bool gpu_eligible = false;
};

HostWindow makeWindow(const PlacementDatabase& database, std::size_t row_index,
                      std::vector<ModuleId> modules, std::size_t maximum_net_degree) {
    HostWindow result;
    result.row_index = row_index;
    result.modules = std::move(modules);
    for (const ModuleId id : result.modules) {
        for (const PinId pin_id : database.modules[id].pins) result.nets.push_back(database.pins[pin_id].net);
    }
    std::sort(result.nets.begin(), result.nets.end());
    result.nets.erase(std::unique(result.nets.begin(), result.nets.end()), result.nets.end());
    result.gpu_eligible = result.modules.size() == static_cast<std::size_t>(kWindowSize) && !result.nets.empty();
    for (const NetId net_id : result.nets) {
        if (database.nets[net_id].pins.size() > maximum_net_degree) result.gpu_eligible = false;
    }
    return result;
}

Vec2 pinPositionWithOverrides(const PlacementDatabase& database, PinId pin_id,
                              const HostWindow& window, const std::vector<Vec2>& centers) {
    const Pin& pin = database.pins[pin_id];
    Vec2 center = database.modules[pin.module].center;
    for (std::size_t index = 0; index < window.modules.size(); ++index) {
        if (window.modules[index] == pin.module) {
            center = centers[index];
            break;
        }
    }
    return center + transformOffset(pin.offset, database.modules[pin.module].orientation);
}

double localHpwl(const PlacementDatabase& database, const HostWindow& window,
                 const std::vector<Vec2>& centers) {
    double result = 0.0;
    for (const NetId net_id : window.nets) {
        const Net& net = database.nets[net_id];
        double minimum_x = std::numeric_limits<double>::infinity();
        double maximum_x = -std::numeric_limits<double>::infinity();
        double minimum_y = std::numeric_limits<double>::infinity();
        double maximum_y = -std::numeric_limits<double>::infinity();
        for (const PinId pin_id : net.pins) {
            const Vec2 position = pinPositionWithOverrides(database, pin_id, window, centers);
            minimum_x = std::min(minimum_x, position.x);
            maximum_x = std::max(maximum_x, position.x);
            minimum_y = std::min(minimum_y, position.y);
            maximum_y = std::max(maximum_y, position.y);
        }
        result += net.weight * ((maximum_x - minimum_x) + (maximum_y - minimum_y));
    }
    return result;
}

std::vector<Vec2> centersForOrder(const PlacementDatabase& database, const HostWindow& window,
                                  const std::vector<std::size_t>& order) {
    const SiteRow& row = database.rows[window.row_index];
    std::vector<Vec2> centers(window.modules.size());
    double cursor = database.modules[window.modules.front()].lowerLeft().x;
    for (const std::size_t local : order) {
        const Module& module = database.modules[window.modules[local]];
        centers[local] = {cursor + module.width * 0.5, row.bottom + module.height * 0.5};
        cursor += roundedWidth(module, row);
    }
    return centers;
}

void applyOrder(PlacementDatabase& database, const HostWindow& window, const std::vector<std::size_t>& order) {
    const SiteRow& row = database.rows[window.row_index];
    double cursor = database.modules[window.modules.front()].lowerLeft().x;
    for (const std::size_t local : order) {
        Module& module = database.modules[window.modules[local]];
        module.setLowerLeft({cursor, row.bottom});
        if (row.orientation != Orientation::Unknown) module.orientation = row.orientation;
        cursor += roundedWidth(module, row);
    }
}

struct CpuWindowResult {
    bool improved = false;
    std::vector<std::size_t> order;
    std::size_t permutations = 0;
};

CpuWindowResult optimizeCpuWindow(PlacementDatabase& database, const HostWindow& window,
                                  const DetailedPlacementOptions& options) {
    if (window.nets.empty()) return {};
    for (const NetId net_id : window.nets) {
        if (database.nets[net_id].pins.size() > options.maximum_net_degree) return {};
    }
    std::vector<Vec2> original;
    original.reserve(window.modules.size());
    for (const ModuleId id : window.modules) original.push_back(database.modules[id].center);
    double best = localHpwl(database, window, original);
    std::vector<std::size_t> order(window.modules.size());
    std::iota(order.begin(), order.end(), 0U);
    CpuWindowResult result;
    do {
        const std::vector<Vec2> centers = centersForOrder(database, window, order);
        const double value = localHpwl(database, window, centers);
        ++result.permutations;
        if (value < best - options.improvement_epsilon) {
            best = value;
            result.order = order;
            result.improved = true;
        }
    } while (std::next_permutation(order.begin(), order.end()));
    return result;
}

void requireInt(std::size_t value, const char* description) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("CUDA detailed-placement cannot represent ") + description + ".");
    }
}

struct StaticDeviceData {
    std::vector<DeviceVec2> centers;
    std::vector<double> widths;
    std::vector<double> heights;
    std::vector<int> pin_modules;
    std::vector<DeviceVec2> pin_offsets;
    std::vector<int> net_offsets;
    std::vector<int> net_pin_ids;
    std::vector<double> net_weights;
};

StaticDeviceData buildStaticDeviceData(const PlacementDatabase& database) {
    requireInt(database.modules.size(), "module count");
    requireInt(database.pins.size(), "pin count");
    requireInt(database.nets.size(), "net count");
    StaticDeviceData result;
    result.centers.reserve(database.modules.size());
    result.widths.reserve(database.modules.size());
    result.heights.reserve(database.modules.size());
    for (const Module& module : database.modules) {
        result.centers.push_back({module.center.x, module.center.y});
        result.widths.push_back(module.width);
        result.heights.push_back(module.height);
    }
    result.pin_modules.resize(database.pins.size());
    result.pin_offsets.resize(database.pins.size());
    for (PinId pin_id = 0; pin_id < database.pins.size(); ++pin_id) {
        const Pin& pin = database.pins[pin_id];
        result.pin_modules[pin_id] = static_cast<int>(pin.module);
        const Vec2 offset = transformOffset(pin.offset, database.modules[pin.module].orientation);
        result.pin_offsets[pin_id] = {offset.x, offset.y};
    }
    result.net_offsets.reserve(database.nets.size() + 1U);
    result.net_offsets.push_back(0);
    for (const Net& net : database.nets) {
        for (const PinId pin_id : net.pins) result.net_pin_ids.push_back(static_cast<int>(pin_id));
        requireInt(result.net_pin_ids.size(), "flattened net-pin count");
        result.net_offsets.push_back(static_cast<int>(result.net_pin_ids.size()));
        result.net_weights.push_back(net.weight);
    }
    return result;
}

void refreshCenters(const PlacementDatabase& database, std::vector<DeviceVec2>& centers) {
    centers.resize(database.modules.size());
    for (std::size_t index = 0; index < database.modules.size(); ++index) {
        centers[index] = {database.modules[index].center.x, database.modules[index].center.y};
    }
}

void refreshPinOffsets(const PlacementDatabase& database, std::vector<DeviceVec2>& pin_offsets) {
    pin_offsets.resize(database.pins.size());
    for (PinId pin_id = 0; pin_id < database.pins.size(); ++pin_id) {
        const Pin& pin = database.pins[pin_id];
        const Vec2 offset = transformOffset(pin.offset, database.modules[pin.module].orientation);
        pin_offsets[pin_id] = {offset.x, offset.y};
    }
}

struct ModulePlacementState {
    Vec2 center;
    Orientation orientation = Orientation::Unknown;
};

std::vector<ModulePlacementState> snapshotModulePlacement(const PlacementDatabase& database) {
    std::vector<ModulePlacementState> snapshot;
    snapshot.reserve(database.modules.size());
    for (const Module& module : database.modules) snapshot.push_back({module.center, module.orientation});
    return snapshot;
}

void restoreModulePlacement(PlacementDatabase& database, const std::vector<ModulePlacementState>& snapshot) {
    const std::size_t count = std::min(database.modules.size(), snapshot.size());
    for (std::size_t index = 0; index < count; ++index) {
        database.modules[index].center = snapshot[index].center;
        database.modules[index].orientation = snapshot[index].orientation;
    }
}

}  // namespace

bool cudaDetailedPlacementBackendCompiled() {
    return true;
}

bool tryRunCudaDetailedPlacement(PlacementDatabase& database, const DetailedPlacementOptions& options,
                                 DetailedPlacementResult& result, std::string& reason) {
    reason.clear();
    if (options.method != DetailedPlacementMethod::WindowReorder || options.window_size != kWindowSize) {
        reason = "CUDA detailed placement currently supports window reordering with a four-cell window only.";
        return false;
    }
    if (options.passes <= 0 || options.maximum_net_degree < 2U || options.improvement_epsilon < 0.0 ||
        options.cuda_device < 1 || options.cuda_device > 4 || options.maximum_cuda_memory_bytes == 0U) {
        reason = "CUDA detailed-placement options are outside their valid range.";
        return false;
    }

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || options.cuda_device >= device_count) {
        reason = "Requested CUDA device is unavailable on this host.";
        return false;
    }
    if (cudaSetDevice(options.cuda_device) != cudaSuccess) {
        reason = "Unable to select the requested CUDA device.";
        return false;
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, options.cuda_device) != cudaSuccess || properties.major < 6) {
        reason = "Requested CUDA device lacks required double-precision support.";
        return false;
    }
    std::size_t free_bytes = 0U;
    std::size_t total_bytes = 0U;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess || free_bytes <= kSharedGpuReserveBytes) {
        reason = "Requested CUDA device lacks the required 4 GiB shared-memory reserve.";
        return false;
    }

    std::vector<ModulePlacementState> placement_checkpoint;
    try {
        placement_checkpoint = snapshotModulePlacement(database);
        const std::size_t allocation_limit =
            std::min(options.maximum_cuda_memory_bytes, free_bytes - kSharedGpuReserveBytes);
        DeviceMemoryBudget budget(allocation_limit);
        StaticDeviceData host = buildStaticDeviceData(database);

        DeviceBuffer<DeviceVec2> device_centers;
        DeviceBuffer<double> device_widths;
        DeviceBuffer<double> device_heights;
        DeviceBuffer<int> device_pin_modules;
        DeviceBuffer<DeviceVec2> device_pin_offsets;
        DeviceBuffer<int> device_net_offsets;
        DeviceBuffer<int> device_net_pin_ids;
        DeviceBuffer<double> device_net_weights;
        device_centers.allocate(host.centers.size(), budget, "CUDA detailed module centers");
        device_widths.allocate(host.widths.size(), budget, "CUDA detailed module widths");
        device_heights.allocate(host.heights.size(), budget, "CUDA detailed module heights");
        device_pin_modules.allocate(host.pin_modules.size(), budget, "CUDA detailed pin modules");
        device_pin_offsets.allocate(host.pin_offsets.size(), budget, "CUDA detailed pin offsets");
        device_net_offsets.allocate(host.net_offsets.size(), budget, "CUDA detailed net offsets");
        device_net_pin_ids.allocate(host.net_pin_ids.size(), budget, "CUDA detailed net pin ids");
        device_net_weights.allocate(host.net_weights.size(), budget, "CUDA detailed net weights");
        device_widths.copyFromHost(host.widths, "Upload CUDA detailed module widths");
        device_heights.copyFromHost(host.heights, "Upload CUDA detailed module heights");
        device_pin_modules.copyFromHost(host.pin_modules, "Upload CUDA detailed pin modules");
        device_pin_offsets.copyFromHost(host.pin_offsets, "Upload CUDA detailed pin offsets");
        device_net_offsets.copyFromHost(host.net_offsets, "Upload CUDA detailed net offsets");
        device_net_pin_ids.copyFromHost(host.net_pin_ids, "Upload CUDA detailed net pin ids");
        device_net_weights.copyFromHost(host.net_weights, "Upload CUDA detailed net weights");

        result = {};
        result.hpwl_before = calculateHpwl(database).hpwl;
        result.compute_backend_used = DetailedPlacementBackend::Cuda;
        result.cuda_device_used = options.cuda_device;
        const auto started = std::chrono::steady_clock::now();

        for (int pass = 0; pass < options.passes; ++pass) {
            const std::vector<RowSegment> segments = buildRowSegments(database, 1e-6);
            const std::size_t phase = static_cast<std::size_t>((pass % 2) * (kWindowSize / 2));
            std::vector<HostWindow> gpu_windows;
            std::vector<HostWindow> cpu_windows;
            for (const RowSegment& segment : segments) {
                for (std::size_t start = phase; start + 1U < segment.modules.size(); start += kWindowSize) {
                    const std::size_t count = std::min<std::size_t>(kWindowSize, segment.modules.size() - start);
                    if (count < 2U) break;
                    std::vector<ModuleId> modules(segment.modules.begin() + static_cast<std::ptrdiff_t>(start),
                                                  segment.modules.begin() + static_cast<std::ptrdiff_t>(start + count));
                    HostWindow window = makeWindow(database, segment.row_index, std::move(modules),
                                                   options.maximum_net_degree);
                    if (window.gpu_eligible) {
                        gpu_windows.push_back(std::move(window));
                    } else {
                        cpu_windows.push_back(std::move(window));
                    }
                }
            }

            if (!gpu_windows.empty()) {
                requireInt(gpu_windows.size(), "window count");
                std::vector<int> window_modules;
                std::vector<double> window_start_x;
                std::vector<double> window_bottom;
                std::vector<double> window_spacing;
                std::vector<int> window_net_offsets;
                std::vector<int> window_net_ids;
                window_modules.reserve(gpu_windows.size() * kWindowSize);
                window_start_x.reserve(gpu_windows.size());
                window_bottom.reserve(gpu_windows.size());
                window_spacing.reserve(gpu_windows.size());
                window_net_offsets.reserve(gpu_windows.size() + 1U);
                window_net_offsets.push_back(0);
                for (const HostWindow& window : gpu_windows) {
                    const SiteRow& row = database.rows[window.row_index];
                    for (const ModuleId id : window.modules) window_modules.push_back(static_cast<int>(id));
                    window_start_x.push_back(database.modules[window.modules.front()].lowerLeft().x);
                    window_bottom.push_back(row.bottom);
                    window_spacing.push_back(row.site_spacing);
                    for (const NetId id : window.nets) window_net_ids.push_back(static_cast<int>(id));
                    requireInt(window_net_ids.size(), "window-net incidence count");
                    window_net_offsets.push_back(static_cast<int>(window_net_ids.size()));
                }
                refreshCenters(database, host.centers);
                refreshPinOffsets(database, host.pin_offsets);
                device_centers.copyFromHost(host.centers, "Upload CUDA detailed module centers");
                device_pin_offsets.copyFromHost(host.pin_offsets, "Upload CUDA detailed pin offsets");

                DeviceBuffer<int> device_window_modules;
                DeviceBuffer<double> device_window_start_x;
                DeviceBuffer<double> device_window_bottom;
                DeviceBuffer<double> device_window_spacing;
                DeviceBuffer<int> device_window_net_offsets;
                DeviceBuffer<int> device_window_net_ids;
                DeviceBuffer<int> device_best_permutations;
                device_window_modules.allocate(window_modules.size(), budget, "CUDA detailed window modules");
                device_window_start_x.allocate(window_start_x.size(), budget, "CUDA detailed window start x");
                device_window_bottom.allocate(window_bottom.size(), budget, "CUDA detailed window bottoms");
                device_window_spacing.allocate(window_spacing.size(), budget, "CUDA detailed window spacing");
                device_window_net_offsets.allocate(window_net_offsets.size(), budget, "CUDA detailed window net offsets");
                device_window_net_ids.allocate(window_net_ids.size(), budget, "CUDA detailed window net ids");
                device_best_permutations.allocate(gpu_windows.size(), budget, "CUDA detailed best permutations");
                device_window_modules.copyFromHost(window_modules, "Upload CUDA detailed window modules");
                device_window_start_x.copyFromHost(window_start_x, "Upload CUDA detailed window start x");
                device_window_bottom.copyFromHost(window_bottom, "Upload CUDA detailed window bottoms");
                device_window_spacing.copyFromHost(window_spacing, "Upload CUDA detailed window spacing");
                device_window_net_offsets.copyFromHost(window_net_offsets, "Upload CUDA detailed window net offsets");
                device_window_net_ids.copyFromHost(window_net_ids, "Upload CUDA detailed window net ids");
                evaluateWindowPermutationsKernel<<<static_cast<unsigned int>(gpu_windows.size()), kCudaThreads>>>(
                    device_centers.data(), device_widths.data(), device_heights.data(), device_pin_modules.data(),
                    device_pin_offsets.data(), device_net_offsets.data(), device_net_pin_ids.data(),
                    device_net_weights.data(), device_window_modules.data(), device_window_start_x.data(),
                    device_window_bottom.data(), device_window_spacing.data(), device_window_net_offsets.data(),
                    device_window_net_ids.data(), static_cast<int>(gpu_windows.size()), device_best_permutations.data());
                checkCuda(cudaGetLastError(), "Launch CUDA detailed window evaluator");
                checkCuda(cudaDeviceSynchronize(), "Synchronize CUDA detailed window evaluator");

                std::vector<int> best_permutations;
                device_best_permutations.copyToHost(best_permutations, "Download CUDA detailed best permutations");
                for (std::size_t index = 0; index < gpu_windows.size(); ++index) {
                    ++result.evaluated_windows;
                    result.evaluated_permutations += kPermutationCount;
                    const int permutation = best_permutations[index];
                    if (permutation <= 0 || permutation >= kPermutationCount) {
                        continue;
                    }
                    std::vector<std::size_t> order(kWindowSize);
                    for (int local = 0; local < kWindowSize; ++local) {
                        order[static_cast<std::size_t>(local)] =
                            static_cast<std::size_t>(kHostPermutations[permutation][local]);
                    }
                    std::vector<Vec2> original;
                    original.reserve(kWindowSize);
                    for (const ModuleId id : gpu_windows[index].modules) original.push_back(database.modules[id].center);
                    const double before = localHpwl(database, gpu_windows[index], original);
                    const std::vector<Vec2> candidate = centersForOrder(database, gpu_windows[index], order);
                    const double after = localHpwl(database, gpu_windows[index], candidate);
                    if (after < before - options.improvement_epsilon) {
                        applyOrder(database, gpu_windows[index], order);
                        ++result.accepted_operations;
                    }
                }
            }

            for (const HostWindow& window : cpu_windows) {
                CpuWindowResult cpu = optimizeCpuWindow(database, window, options);
                ++result.evaluated_windows;
                result.evaluated_permutations += cpu.permutations;
                if (cpu.improved) {
                    applyOrder(database, window, cpu.order);
                    ++result.accepted_operations;
                }
            }
            result.completed_passes = pass + 1;
        }
        result.hpwl_after = calculateHpwl(database).hpwl;
        result.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        result.cuda_reserved_memory_bytes = budget.peakBytes();
        return true;
    } catch (const std::exception& error) {
        restoreModulePlacement(database, placement_checkpoint);
        reason = error.what();
        return false;
    }
}

}  // namespace myplacement::detail
