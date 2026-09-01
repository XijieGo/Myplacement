#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "myplacement/core/Geometry.hpp"

namespace myplacement {

using ModuleId = std::size_t;
using NetId = std::size_t;
using PinId = std::size_t;

enum class Orientation { North, South, East, West, FlipNorth, FlipSouth, FlipEast, FlipWest, Unknown };
enum class PinDirection { Input, Output, Bidirectional, Unknown };

Orientation parseOrientation(const std::string& text);
std::string toString(Orientation orientation);
Vec2 transformOffset(Vec2 offset, Orientation orientation);
PinDirection parsePinDirection(const std::string& text);

struct Pin {
    ModuleId module = 0;
    NetId net = 0;
    Vec2 offset;
    PinDirection direction = PinDirection::Unknown;
};

struct Module {
    std::string name;
    double width = 0.0;
    double height = 0.0;
    Vec2 center;
    Orientation orientation = Orientation::North;
    bool is_terminal = false;
    bool is_fixed = false;
    bool is_macro = false;
    std::vector<PinId> pins;

    [[nodiscard]] double area() const { return width * height; }
    [[nodiscard]] Rect rect() const {
        return {{center.x - width * 0.5, center.y - height * 0.5},
                {center.x + width * 0.5, center.y + height * 0.5}};
    }
    [[nodiscard]] Vec2 lowerLeft() const { return rect().ll; }
    void setLowerLeft(Vec2 lower_left) {
        center = {lower_left.x + width * 0.5, lower_left.y + height * 0.5};
    }
};

struct Net {
    std::string name;
    double weight = 1.0;
    std::vector<PinId> pins;
};

struct SiteRow {
    double bottom = 0.0;
    double height = 0.0;
    double site_width = 1.0;
    double site_spacing = 1.0;
    double x_start = 0.0;
    std::size_t site_count = 0;
    Orientation orientation = Orientation::North;

    [[nodiscard]] double xEnd() const { return x_start + static_cast<double>(site_count) * site_spacing; }
    [[nodiscard]] Rect rect() const { return {{x_start, bottom}, {xEnd(), bottom + height}}; }
};

struct DatabaseSummary {
    std::size_t module_count = 0;
    std::size_t movable_count = 0;
    std::size_t fixed_count = 0;
    std::size_t macro_count = 0;
    std::size_t net_count = 0;
    std::size_t pin_count = 0;
    double movable_area = 0.0;
    double fixed_area = 0.0;
};

class PlacementDatabase {
public:
    std::vector<Module> modules;
    std::vector<Pin> pins;
    std::vector<Net> nets;
    std::vector<SiteRow> rows;
    Rect core_region;
    Rect chip_region;

    std::optional<std::size_t> declared_nodes;
    std::optional<std::size_t> declared_terminals;
    std::optional<std::size_t> declared_nets;
    std::optional<std::size_t> declared_pins;

    ModuleId addModule(Module module);
    NetId addNet(Net net);
    PinId addPin(Pin pin);

    [[nodiscard]] const Module* findModule(const std::string& name) const;
    [[nodiscard]] Module* findModule(const std::string& name);
    [[nodiscard]] const Net* findNet(const std::string& name) const;
    [[nodiscard]] Net* findNet(const std::string& name);
    [[nodiscard]] Vec2 pinPosition(PinId pin_id) const;
    [[nodiscard]] const std::vector<ModuleId>& movableModules() const { return movable_modules_; }
    [[nodiscard]] double nominalRowHeight() const { return nominal_row_height_; }
    [[nodiscard]] DatabaseSummary summary() const;

    void refreshDerivedData();
    void clampModuleToCore(ModuleId module_id);

private:
    std::unordered_map<std::string, ModuleId> module_by_name_;
    std::unordered_map<std::string, NetId> net_by_name_;
    std::vector<ModuleId> movable_modules_;
    double nominal_row_height_ = 0.0;
};

}  // namespace myplacement
