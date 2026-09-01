#pragma once

#include <filesystem>

#include "myplacement/model/PlacementDatabase.hpp"

namespace myplacement {

struct RenderOptions {
    int width = 1600;
    int margin = 24;
    bool draw_rows = true;
    bool draw_fixed = true;
    bool draw_macros = true;
};

class Renderer {
public:
    void writeBitmap(const PlacementDatabase& database, const std::filesystem::path& path,
                     const RenderOptions& options = {}) const;
};

}  // namespace myplacement
