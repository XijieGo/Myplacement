#include "myplacement/io/BookshelfParser.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace myplacement {
namespace {

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1U);
}

std::string withoutComment(const std::string& line) {
    return trim(line.substr(0, line.find('#')));
}

bool beginsWith(const std::string& text, const std::string& prefix) {
    return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

std::string upper(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::toupper(character));
    });
    return value;
}

std::vector<double> extractNumbers(std::string value) {
    for (char& character : value) {
        if (character == ':' || character == ',' || character == '\t') character = ' ';
    }
    std::istringstream stream(value);
    std::vector<double> values;
    std::string token;
    while (stream >> token) {
        try {
            std::size_t consumed = 0;
            const double number = std::stod(token, &consumed);
            if (consumed == token.size()) values.push_back(number);
        } catch (const std::exception&) {
            // Non-numeric labels are expected in BookShelf records.
        }
    }
    return values;
}

std::ifstream openFile(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw ParseError("Unable to open BookShelf file: " + path.string());
    return input;
}

void parseNodes(const std::filesystem::path& path, PlacementDatabase& database) {
    std::ifstream input = openFile(path);
    std::string raw_line;
    std::size_t line_number = 0;
    while (std::getline(input, raw_line)) {
        ++line_number;
        const std::string line = withoutComment(raw_line);
        if (line.empty() || beginsWith(line, "UCLA")) continue;

        const std::vector<double> numbers = extractNumbers(line);
        if (beginsWith(line, "NumNodes")) {
            if (numbers.empty()) throw ParseError(path.string() + ": invalid NumNodes at line " +
                                                   std::to_string(line_number));
            database.declared_nodes = static_cast<std::size_t>(numbers.front());
            continue;
        }
        if (beginsWith(line, "NumTerminals")) {
            if (numbers.empty()) throw ParseError(path.string() + ": invalid NumTerminals at line " +
                                                   std::to_string(line_number));
            database.declared_terminals = static_cast<std::size_t>(numbers.front());
            continue;
        }

        std::istringstream stream(line);
        Module module;
        if (!(stream >> module.name >> module.width >> module.height)) {
            throw ParseError(path.string() + ": invalid node at line " + std::to_string(line_number));
        }
        std::string attribute;
        while (stream >> attribute) {
            if (upper(attribute).find("TERMINAL") != std::string::npos) {
                module.is_terminal = true;
                module.is_fixed = true;
            }
        }
        database.addModule(std::move(module));
    }
}

void parseScl(const std::filesystem::path& path, PlacementDatabase& database) {
    std::ifstream input = openFile(path);
    std::string raw_line;
    bool inside_row = false;
    double bottom = 0.0;
    double height = 0.0;
    double site_width = 1.0;
    double site_spacing = 1.0;
    Orientation orientation = Orientation::North;
    std::size_t line_number = 0;

    while (std::getline(input, raw_line)) {
        ++line_number;
        const std::string line = withoutComment(raw_line);
        if (line.empty() || beginsWith(line, "UCLA") || beginsWith(line, "NumRows")) continue;
        if (beginsWith(line, "CoreRow")) {
            inside_row = true;
            bottom = 0.0;
            height = 0.0;
            site_width = 1.0;
            site_spacing = 1.0;
            orientation = Orientation::North;
            continue;
        }
        if (!inside_row) continue;
        if (line == "End") {
            inside_row = false;
            continue;
        }

        const std::vector<double> numbers = extractNumbers(line);
        if (beginsWith(line, "Coordinate")) {
            if (!numbers.empty()) bottom = numbers.front();
        } else if (beginsWith(line, "Height")) {
            if (!numbers.empty()) height = numbers.front();
        } else if (beginsWith(line, "Sitewidth")) {
            if (!numbers.empty()) site_width = numbers.front();
        } else if (beginsWith(line, "Sitespacing")) {
            if (!numbers.empty()) site_spacing = numbers.front();
        } else if (beginsWith(line, "Siteorient")) {
            std::istringstream stream(line);
            std::string label;
            std::string separator;
            std::string value;
            stream >> label >> separator >> value;
            if (!value.empty()) orientation = parseOrientation(value);
        } else if (beginsWith(line, "SubrowOrigin")) {
            if (numbers.size() < 2U || height <= kEpsilon || site_spacing <= kEpsilon) {
                throw ParseError(path.string() + ": invalid SubrowOrigin at line " +
                                 std::to_string(line_number));
            }
            SiteRow row;
            row.bottom = bottom;
            row.height = height;
            row.site_width = site_width;
            row.site_spacing = site_spacing;
            row.x_start = numbers[0];
            row.site_count = static_cast<std::size_t>(numbers[1]);
            row.orientation = orientation;
            database.rows.push_back(row);
        }
    }
    if (database.rows.empty()) throw ParseError(path.string() + ": contains no valid SiteRow records");
}

void parsePl(const std::filesystem::path& path, PlacementDatabase& database) {
    std::ifstream input = openFile(path);
    std::string raw_line;
    std::size_t line_number = 0;
    while (std::getline(input, raw_line)) {
        ++line_number;
        const std::string line = withoutComment(raw_line);
        if (line.empty() || beginsWith(line, "UCLA")) continue;

        std::istringstream stream(line);
        std::string name;
        double x = 0.0;
        double y = 0.0;
        if (!(stream >> name >> x >> y)) continue;
        Module* module = database.findModule(name);
        if (module == nullptr) {
            throw ParseError(path.string() + ": position references unknown module '" + name + "' at line " +
                             std::to_string(line_number));
        }

        std::string orientation_token;
        if (stream >> orientation_token) {
            if (orientation_token == ":") {
                stream >> orientation_token;
            } else if (!orientation_token.empty() && orientation_token.front() == ':') {
                orientation_token.erase(orientation_token.begin());
            }
            if (!orientation_token.empty()) module->orientation = parseOrientation(orientation_token);
        }

        std::string attribute;
        while (stream >> attribute) {
            if (upper(attribute).find("FIXED") != std::string::npos) module->is_fixed = true;
        }
        module->setLowerLeft({x, y});
    }
}

void parseNets(const std::filesystem::path& path, PlacementDatabase& database) {
    std::ifstream input = openFile(path);
    std::string raw_line;
    std::size_t line_number = 0;

    auto nextDataLine = [&]() -> std::pair<std::string, std::size_t> {
        while (std::getline(input, raw_line)) {
            ++line_number;
            const std::string candidate = withoutComment(raw_line);
            if (!candidate.empty()) return {candidate, line_number};
        }
        return {{}, line_number};
    };

    while (std::getline(input, raw_line)) {
        ++line_number;
        const std::string line = withoutComment(raw_line);
        if (line.empty() || beginsWith(line, "UCLA")) continue;
        const std::vector<double> numbers = extractNumbers(line);
        if (beginsWith(line, "NumNets")) {
            if (!numbers.empty()) database.declared_nets = static_cast<std::size_t>(numbers.front());
            continue;
        }
        if (beginsWith(line, "NumPins")) {
            if (!numbers.empty()) database.declared_pins = static_cast<std::size_t>(numbers.front());
            continue;
        }
        if (!beginsWith(line, "NetDegree")) continue;

        std::istringstream header(line);
        std::string label;
        std::string separator;
        std::size_t degree = 0;
        std::string net_name;
        if (!(header >> label >> separator >> degree >> net_name)) {
            throw ParseError(path.string() + ": invalid NetDegree at line " + std::to_string(line_number));
        }
        Net net;
        net.name = net_name;
        const NetId net_id = database.addNet(std::move(net));
        for (std::size_t pin_index = 0; pin_index < degree; ++pin_index) {
            const auto [pin_line, pin_line_number] = nextDataLine();
            if (pin_line.empty()) {
                throw ParseError(path.string() + ": unexpected end of file while reading net '" + net_name + "'");
            }
            std::string normalized_line = pin_line;
            std::replace(normalized_line.begin(), normalized_line.end(), ':', ' ');
            std::istringstream pin_stream(normalized_line);
            std::string module_name;
            std::string direction;
            double offset_x = 0.0;
            double offset_y = 0.0;
            if (!(pin_stream >> module_name >> direction >> offset_x >> offset_y)) {
                throw ParseError(path.string() + ": invalid pin at line " +
                                 std::to_string(pin_line_number));
            }
            Module* module = database.findModule(module_name);
            if (module == nullptr) {
                throw ParseError(path.string() + ": net '" + net_name + "' references unknown module '" +
                                 module_name + "' at line " + std::to_string(pin_line_number));
            }
            Pin pin;
            pin.module = static_cast<ModuleId>(module - database.modules.data());
            pin.net = net_id;
            pin.offset = {offset_x, offset_y};
            pin.direction = parsePinDirection(direction);
            database.addPin(pin);
        }
    }
}

void parseWeights(const std::filesystem::path& path, PlacementDatabase& database) {
    if (!std::filesystem::exists(path)) return;
    std::ifstream input = openFile(path);
    std::string raw_line;
    while (std::getline(input, raw_line)) {
        const std::string line = withoutComment(raw_line);
        if (line.empty() || beginsWith(line, "UCLA")) continue;
        std::istringstream stream(line);
        std::string net_name;
        double weight = 1.0;
        if (!(stream >> net_name >> weight)) continue;
        if (Net* net = database.findNet(net_name); net != nullptr) net->weight = weight;
    }
}

void validateCounts(const PlacementDatabase& database, const std::filesystem::path& source) {
    const auto mismatch = [&](const char* label, std::size_t expected, std::size_t actual) {
        throw ParseError(source.string() + ": declared " + label + "=" + std::to_string(expected) +
                         ", parsed " + std::to_string(actual));
    };
    if (database.declared_nodes && *database.declared_nodes != database.modules.size()) {
        mismatch("NumNodes", *database.declared_nodes, database.modules.size());
    }
    if (database.declared_terminals) {
        const std::size_t actual = static_cast<std::size_t>(std::count_if(
            database.modules.begin(), database.modules.end(), [](const Module& module) { return module.is_terminal; }));
        if (*database.declared_terminals != actual) mismatch("NumTerminals", *database.declared_terminals, actual);
    }
    if (database.declared_nets && *database.declared_nets != database.nets.size()) {
        mismatch("NumNets", *database.declared_nets, database.nets.size());
    }
    if (database.declared_pins && *database.declared_pins != database.pins.size()) {
        mismatch("NumPins", *database.declared_pins, database.pins.size());
    }
}

}  // namespace

PlacementDatabase BookshelfParser::parseAux(const std::filesystem::path& aux_path) const {
    std::ifstream input = openFile(aux_path);
    const std::filesystem::path directory = aux_path.parent_path();
    std::unordered_map<std::string, std::filesystem::path> files;
    std::string raw_line;
    while (std::getline(input, raw_line)) {
        const std::string line = withoutComment(raw_line);
        const std::size_t colon = line.find(':');
        if (line.empty() || colon == std::string::npos) continue;
        std::istringstream stream(line.substr(colon + 1U));
        std::string file_name;
        while (stream >> file_name) {
            const std::string extension = std::filesystem::path(file_name).extension().string();
            if (!extension.empty()) files.emplace(extension, directory / file_name);
        }
    }

    const auto required = [&](const std::string& extension) -> const std::filesystem::path& {
        const auto iterator = files.find(extension);
        if (iterator == files.end()) {
            throw ParseError(aux_path.string() + ": missing required '" + extension + "' file in AUX record");
        }
        return iterator->second;
    };

    PlacementDatabase database;
    parseNodes(required(".nodes"), database);
    parseScl(required(".scl"), database);
    parsePl(required(".pl"), database);
    parseNets(required(".nets"), database);
    const auto weight_file = files.find(".wts");
    if (weight_file != files.end()) parseWeights(weight_file->second, database);
    database.refreshDerivedData();
    validateCounts(database, aux_path);
    return database;
}

}  // namespace myplacement
