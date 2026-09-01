#define cimg_display 0
#include "CImg.h"

#include "myplacement/export/Renderer.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace myplacement {
namespace {

using cimg_library::CImg;

int toPixelX(double x, const Rect& region, double scale, int margin) {
    return margin + static_cast<int>(std::lround((x - region.ll.x) * scale));
}

int toPixelY(double y, const Rect& region, double scale, int margin) {
    return margin + static_cast<int>(std::lround((region.ur.y - y) * scale));
}

}  // namespace

void Renderer::writeBitmap(const PlacementDatabase& database, const std::filesystem::path& path,
                           const RenderOptions& options) const {
    Rect region = database.chip_region.area() > kEpsilon ? database.chip_region : database.core_region;
    if (region.area() <= kEpsilon || options.width <= 0 || options.margin < 0) {
        throw std::invalid_argument("Cannot render an invalid placement region.");
    }
    const int content_width = options.width;
    const int content_height = std::max(80, static_cast<int>(std::lround(
        static_cast<double>(content_width) * region.height() / region.width())));
    const double scale_x = static_cast<double>(content_width) / region.width();
    const double scale_y = static_cast<double>(content_height) / region.height();
    const int margin = options.margin;

    CImg<unsigned char> image(content_width + 2 * margin, content_height + 2 * margin, 1, 3, 255);
    const unsigned char core_color[] = {60, 60, 60};
    const unsigned char row_color[] = {210, 210, 210};
    const unsigned char movable_color[] = {210, 55, 55};
    const unsigned char macro_color[] = {35, 135, 65};
    const unsigned char fixed_color[] = {45, 95, 205};

    const auto drawRect = [&](const Rect& rectangle, const unsigned char* color, float opacity) {
        int x1 = toPixelX(rectangle.ll.x, region, scale_x, margin);
        int x2 = toPixelX(rectangle.ur.x, region, scale_x, margin);
        int y1 = toPixelY(rectangle.ur.y, region, scale_y, margin);
        int y2 = toPixelY(rectangle.ll.y, region, scale_y, margin);
        if (x1 > x2) std::swap(x1, x2);
        if (y1 > y2) std::swap(y1, y2);
        image.draw_rectangle(x1, y1, x2, y2, color, opacity);
    };

    if (options.draw_rows) {
        for (const SiteRow& row : database.rows) {
            const int y = toPixelY(row.bottom, region, scale_y, margin);
            image.draw_line(margin, y, margin + content_width, y, row_color, 0.45F);
        }
    }
    const int core_x1 = toPixelX(database.core_region.ll.x, region, scale_x, margin);
    const int core_x2 = toPixelX(database.core_region.ur.x, region, scale_x, margin);
    const int core_y1 = toPixelY(database.core_region.ur.y, region, scale_y, margin);
    const int core_y2 = toPixelY(database.core_region.ll.y, region, scale_y, margin);
    image.draw_line(core_x1, core_y1, core_x2, core_y1, core_color, 1.0F);
    image.draw_line(core_x2, core_y1, core_x2, core_y2, core_color, 1.0F);
    image.draw_line(core_x2, core_y2, core_x1, core_y2, core_color, 1.0F);
    image.draw_line(core_x1, core_y2, core_x1, core_y1, core_color, 1.0F);

    for (const Module& module : database.modules) {
        if (module.is_fixed && !options.draw_fixed) continue;
        if (module.is_macro && !options.draw_macros) continue;
        const unsigned char* color = module.is_fixed ? fixed_color : (module.is_macro ? macro_color : movable_color);
        drawRect(module.rect(), color, module.is_fixed ? 0.82F : 0.68F);
    }

    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    image.save_bmp(path.string().c_str());
}

}  // namespace myplacement
