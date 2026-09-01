#include "myplacement/model/PlacementDatabase.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>

namespace myplacement {
namespace {

std::string normalized(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::toupper(character));
    });
    return value;
}

void expandToInclude(Rect& target, const Rect& value) {
    target.ll.x = std::min(target.ll.x, value.ll.x);
    target.ll.y = std::min(target.ll.y, value.ll.y);
    target.ur.x = std::max(target.ur.x, value.ur.x);
    target.ur.y = std::max(target.ur.y, value.ur.y);
}

}  // namespace

Orientation parseOrientation(const std::string& text) {
    const std::string value = normalized(text);
    if (value == "N") return Orientation::North;
    if (value == "S") return Orientation::South;
    if (value == "E") return Orientation::East;
    if (value == "W") return Orientation::West;
    if (value == "FN") return Orientation::FlipNorth;
    if (value == "FS") return Orientation::FlipSouth;
    if (value == "FE") return Orientation::FlipEast;
    if (value == "FW") return Orientation::FlipWest;
    return Orientation::Unknown;
}

std::string toString(Orientation orientation) {
    switch (orientation) {
        case Orientation::North: return "N";
        case Orientation::South: return "S";
        case Orientation::East: return "E";
        case Orientation::West: return "W";
        case Orientation::FlipNorth: return "FN";
        case Orientation::FlipSouth: return "FS";
        case Orientation::FlipEast: return "FE";
        case Orientation::FlipWest: return "FW";
        case Orientation::Unknown: return "UNKNOWN";
    }
    return "UNKNOWN";
}

Vec2 transformOffset(Vec2 offset, Orientation orientation) {
    switch (orientation) {
        case Orientation::North: return offset;
        case Orientation::South: return {-offset.x, -offset.y};
        case Orientation::East: return {-offset.y, offset.x};
        case Orientation::West: return {offset.y, -offset.x};
        case Orientation::FlipNorth: return {-offset.x, offset.y};
        case Orientation::FlipSouth: return {offset.x, -offset.y};
        case Orientation::FlipEast: return {offset.y, offset.x};
        case Orientation::FlipWest: return {-offset.y, -offset.x};
        case Orientation::Unknown: return offset;
    }
    return offset;
}

PinDirection parsePinDirection(const std::string& text) {
    const std::string value = normalized(text);
    if (value == "I" || value == "INPUT") return PinDirection::Input;
    if (value == "O" || value == "OUTPUT") return PinDirection::Output;
    if (value == "B" || value == "INOUT") return PinDirection::Bidirectional;
    return PinDirection::Unknown;
}

ModuleId PlacementDatabase::addModule(Module module) {
    if (module.name.empty()) {
        throw std::invalid_argument("A module must have a non-empty name.");
    }
    if (module.width < 0.0 || module.height < 0.0) {
        throw std::invalid_argument("A module cannot have a negative size: " + module.name);
    }
    if (module_by_name_.count(module.name) != 0U) {
        throw std::invalid_argument("Duplicate module name: " + module.name);
    }
    const ModuleId id = modules.size();
    module_by_name_.emplace(module.name, id);
    modules.push_back(std::move(module));
    return id;
}

NetId PlacementDatabase::addNet(Net net) {
    if (net.name.empty()) {
        net.name = "net_" + std::to_string(nets.size());
    }
    if (net_by_name_.count(net.name) != 0U) {
        throw std::invalid_argument("Duplicate net name: " + net.name);
    }
    const NetId id = nets.size();
    net_by_name_.emplace(net.name, id);
    nets.push_back(std::move(net));
    return id;
}

PinId PlacementDatabase::addPin(Pin pin) {
    if (pin.module >= modules.size() || pin.net >= nets.size()) {
        throw std::out_of_range("Pin refers to an unknown module or net.");
    }
    const PinId id = pins.size();
    pins.push_back(pin);
    modules[pin.module].pins.push_back(id);
    nets[pin.net].pins.push_back(id);
    return id;
}

const Module* PlacementDatabase::findModule(const std::string& name) const {
    const auto iterator = module_by_name_.find(name);
    return iterator == module_by_name_.end() ? nullptr : &modules[iterator->second];
}

Module* PlacementDatabase::findModule(const std::string& name) {
    const auto iterator = module_by_name_.find(name);
    return iterator == module_by_name_.end() ? nullptr : &modules[iterator->second];
}

const Net* PlacementDatabase::findNet(const std::string& name) const {
    const auto iterator = net_by_name_.find(name);
    return iterator == net_by_name_.end() ? nullptr : &nets[iterator->second];
}

Net* PlacementDatabase::findNet(const std::string& name) {
    const auto iterator = net_by_name_.find(name);
    return iterator == net_by_name_.end() ? nullptr : &nets[iterator->second];
}

Vec2 PlacementDatabase::pinPosition(PinId pin_id) const {
    const Pin& pin = pins.at(pin_id);
    const Module& module = modules.at(pin.module);
    return module.center + transformOffset(pin.offset, module.orientation);
}

DatabaseSummary PlacementDatabase::summary() const {
    DatabaseSummary result;
    result.module_count = modules.size();
    result.net_count = nets.size();
    result.pin_count = pins.size();
    for (const Module& module : modules) {
        if (module.is_fixed) {
            ++result.fixed_count;
            result.fixed_area += module.area();
        } else {
            ++result.movable_count;
            result.movable_area += module.area();
        }
        if (module.is_macro) ++result.macro_count;
    }
    return result;
}

void PlacementDatabase::refreshDerivedData() {
    movable_modules_.clear();

    if (!rows.empty()) {
        std::vector<double> heights;
        heights.reserve(rows.size());
        for (const SiteRow& row : rows) heights.push_back(row.height);
        std::sort(heights.begin(), heights.end());
        nominal_row_height_ = heights[heights.size() / 2U];

        core_region = rows.front().rect();
        for (const SiteRow& row : rows) expandToInclude(core_region, row.rect());
    } else {
        nominal_row_height_ = 0.0;
        core_region = {};
    }

    for (ModuleId id = 0; id < modules.size(); ++id) {
        Module& module = modules[id];
        if (module.is_terminal) module.is_fixed = true;
        module.is_macro = !module.is_terminal && nominal_row_height_ > kEpsilon &&
                          std::abs(module.height - nominal_row_height_) > kEpsilon;
        if (!module.is_fixed) movable_modules_.push_back(id);
    }

    bool has_region = !rows.empty();
    if (has_region) {
        chip_region = core_region;
    }
    for (const Module& module : modules) {
        const Rect module_rect = module.rect();
        if (!has_region) {
            chip_region = module_rect;
            has_region = true;
        } else {
            expandToInclude(chip_region, module_rect);
        }
    }
}

void PlacementDatabase::clampModuleToCore(ModuleId module_id) {
    if (core_region.area() <= kEpsilon) return;
    Module& module = modules.at(module_id);
    const double lower_x = core_region.ll.x + module.width * 0.5;
    const double upper_x = core_region.ur.x - module.width * 0.5;
    const double lower_y = core_region.ll.y + module.height * 0.5;
    const double upper_y = core_region.ur.y - module.height * 0.5;
    module.center.x = lower_x <= upper_x ? clamp(module.center.x, lower_x, upper_x)
                                         : core_region.center().x;
    module.center.y = lower_y <= upper_y ? clamp(module.center.y, lower_y, upper_y)
                                         : core_region.center().y;
}

}  // namespace myplacement
