#include "myplacement/export/GdsWriter.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace myplacement {
namespace {

enum class GdsRecord : std::uint8_t {
    Header = 0x00,
    BgnLib = 0x01,
    LibName = 0x02,
    Units = 0x03,
    EndLib = 0x04,
    BgnStr = 0x05,
    StrName = 0x06,
    EndStr = 0x07,
    Boundary = 0x08,
    Layer = 0x0D,
    DataType = 0x0E,
    XY = 0x10,
    EndElement = 0x11,
};

enum class GdsDataType : std::uint8_t { None = 0x00, Int16 = 0x02, Int32 = 0x03, Real8 = 0x05, String = 0x06 };

class BinaryWriter {
public:
    explicit BinaryWriter(const std::filesystem::path& path) : output_(path, std::ios::binary) {
        if (!output_) throw std::runtime_error("Unable to create GDSII file: " + path.string());
    }

    void record(GdsRecord type, GdsDataType data_type, const std::vector<std::uint8_t>& payload = {}) {
        if ((payload.size() % 2U) != 0U) throw std::invalid_argument("GDSII record payload must be even-sized.");
        const std::size_t length = payload.size() + 4U;
        if (length > std::numeric_limits<std::uint16_t>::max()) {
            throw std::invalid_argument("GDSII record is too long.");
        }
        writeU16(static_cast<std::uint16_t>(length));
        output_.put(static_cast<char>(type));
        output_.put(static_cast<char>(data_type));
        output_.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
        if (!output_) throw std::runtime_error("Failed while writing GDSII data.");
    }

    static std::vector<std::uint8_t> int16(std::initializer_list<std::int16_t> values) {
        std::vector<std::uint8_t> result;
        result.reserve(values.size() * 2U);
        for (const std::int16_t value : values) appendI16(result, value);
        return result;
    }

    static std::vector<std::uint8_t> int32(const std::vector<std::int32_t>& values) {
        std::vector<std::uint8_t> result;
        result.reserve(values.size() * 4U);
        for (const std::int32_t value : values) appendI32(result, value);
        return result;
    }

    static std::vector<std::uint8_t> string(std::string value) {
        if ((value.size() % 2U) != 0U) value.push_back('\0');
        return {value.begin(), value.end()};
    }

    static std::vector<std::uint8_t> real8(double value) {
        std::vector<std::uint8_t> result(8U, 0U);
        if (value == 0.0) return result;
        const bool negative = value < 0.0;
        value = std::abs(value);
        int exponent = 64;
        while (value >= 1.0) {
            value /= 16.0;
            ++exponent;
        }
        while (value < 1.0 / 16.0) {
            value *= 16.0;
            --exponent;
        }
        if (exponent <= 0 || exponent >= 128) throw std::out_of_range("GDSII real8 exponent out of range.");
        result[0] = static_cast<std::uint8_t>(exponent | (negative ? 0x80 : 0x00));
        const long double mantissa_scale = static_cast<long double>(1ULL << 56U);
        std::uint64_t mantissa = static_cast<std::uint64_t>(std::llround(static_cast<long double>(value) * mantissa_scale));
        for (int index = 7; index >= 1; --index) {
            result[static_cast<std::size_t>(index)] = static_cast<std::uint8_t>(mantissa & 0xFFU);
            mantissa >>= 8U;
        }
        return result;
    }

private:
    void writeU16(std::uint16_t value) {
        output_.put(static_cast<char>((value >> 8U) & 0xFFU));
        output_.put(static_cast<char>(value & 0xFFU));
    }
    static void appendI16(std::vector<std::uint8_t>& output, std::int16_t value) {
        const std::uint16_t bits = static_cast<std::uint16_t>(value);
        output.push_back(static_cast<std::uint8_t>((bits >> 8U) & 0xFFU));
        output.push_back(static_cast<std::uint8_t>(bits & 0xFFU));
    }
    static void appendI32(std::vector<std::uint8_t>& output, std::int32_t value) {
        const std::uint32_t bits = static_cast<std::uint32_t>(value);
        output.push_back(static_cast<std::uint8_t>((bits >> 24U) & 0xFFU));
        output.push_back(static_cast<std::uint8_t>((bits >> 16U) & 0xFFU));
        output.push_back(static_cast<std::uint8_t>((bits >> 8U) & 0xFFU));
        output.push_back(static_cast<std::uint8_t>(bits & 0xFFU));
    }

    std::ofstream output_;
};

std::vector<std::uint8_t> timestampPayload() {
    const std::time_t now = std::time(nullptr);
    std::tm local_time{};
    localtime_r(&now, &local_time);
    const std::initializer_list<std::int16_t> values = {
        static_cast<std::int16_t>(local_time.tm_year + 1900), static_cast<std::int16_t>(local_time.tm_mon + 1),
        static_cast<std::int16_t>(local_time.tm_mday), static_cast<std::int16_t>(local_time.tm_hour),
        static_cast<std::int16_t>(local_time.tm_min), static_cast<std::int16_t>(local_time.tm_sec),
        static_cast<std::int16_t>(local_time.tm_year + 1900), static_cast<std::int16_t>(local_time.tm_mon + 1),
        static_cast<std::int16_t>(local_time.tm_mday), static_cast<std::int16_t>(local_time.tm_hour),
        static_cast<std::int16_t>(local_time.tm_min), static_cast<std::int16_t>(local_time.tm_sec)};
    return BinaryWriter::int16(values);
}

std::int32_t coordinate(double value, double scale) {
    const double scaled = std::round(value * scale);
    if (scaled < static_cast<double>(std::numeric_limits<std::int32_t>::min()) ||
        scaled > static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
        throw std::out_of_range("Layout coordinate cannot be represented by GDSII int32.");
    }
    return static_cast<std::int32_t>(scaled);
}

void writeBoundary(BinaryWriter& writer, const Rect& rectangle, std::int16_t layer, double scale) {
    writer.record(GdsRecord::Boundary, GdsDataType::None);
    writer.record(GdsRecord::Layer, GdsDataType::Int16, BinaryWriter::int16({layer}));
    writer.record(GdsRecord::DataType, GdsDataType::Int16, BinaryWriter::int16({0}));
    const std::int32_t left = coordinate(rectangle.ll.x, scale);
    const std::int32_t right = coordinate(rectangle.ur.x, scale);
    const std::int32_t bottom = coordinate(rectangle.ll.y, scale);
    const std::int32_t top = coordinate(rectangle.ur.y, scale);
    writer.record(GdsRecord::XY, GdsDataType::Int32,
                  BinaryWriter::int32({left, bottom, right, bottom, right, top, left, top, left, bottom}));
    writer.record(GdsRecord::EndElement, GdsDataType::None);
}

}  // namespace

void GdsWriter::write(const PlacementDatabase& database, const std::filesystem::path& path,
                      const GdsWriteOptions& options) const {
    if (options.database_units_per_layout_unit <= 0.0) {
        throw std::invalid_argument("GDSII coordinate scale must be positive.");
    }
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    BinaryWriter writer(path);
    writer.record(GdsRecord::Header, GdsDataType::Int16, BinaryWriter::int16({600}));
    writer.record(GdsRecord::BgnLib, GdsDataType::Int16, timestampPayload());
    writer.record(GdsRecord::LibName, GdsDataType::String, BinaryWriter::string(options.library_name));
    std::vector<std::uint8_t> units = BinaryWriter::real8(1e-6);
    const std::vector<std::uint8_t> database_unit = BinaryWriter::real8(1e-9);
    units.insert(units.end(), database_unit.begin(), database_unit.end());
    writer.record(GdsRecord::Units, GdsDataType::Real8, units);
    writer.record(GdsRecord::BgnStr, GdsDataType::Int16, timestampPayload());
    writer.record(GdsRecord::StrName, GdsDataType::String, BinaryWriter::string(options.structure_name));
    for (const Module& module : database.modules) {
        const std::int16_t layer = module.is_fixed ? 3 : (module.is_macro ? 2 : 1);
        writeBoundary(writer, module.rect(), layer, options.database_units_per_layout_unit);
    }
    writer.record(GdsRecord::EndStr, GdsDataType::None);
    writer.record(GdsRecord::EndLib, GdsDataType::None);
}

}  // namespace myplacement
