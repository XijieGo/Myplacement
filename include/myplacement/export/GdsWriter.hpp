#pragma once

#include <filesystem>

#include <string>

#include "myplacement/model/PlacementDatabase.hpp"

namespace myplacement {

struct GdsWriteOptions {
    std::string library_name = "MYPLACEMENT";
    std::string structure_name = "PLACEMENT";
    double database_units_per_layout_unit = 100.0;
};

class GdsWriter {
public:
    void write(const PlacementDatabase& database, const std::filesystem::path& path,
               const GdsWriteOptions& options = {}) const;
};

}  // namespace myplacement
