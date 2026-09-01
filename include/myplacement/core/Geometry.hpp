#pragma once

#include <algorithm>
#include <cmath>

namespace myplacement {

constexpr double kEpsilon = 1e-9;

struct Vec2 {
    double x = 0.0;
    double y = 0.0;

    Vec2() = default;
    Vec2(double x_value, double y_value) : x(x_value), y(y_value) {}

    Vec2& operator+=(const Vec2& other) {
        x += other.x;
        y += other.y;
        return *this;
    }
    Vec2& operator-=(const Vec2& other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }
};

inline Vec2 operator+(Vec2 left, const Vec2& right) {
    left += right;
    return left;
}

inline Vec2 operator-(Vec2 left, const Vec2& right) {
    left -= right;
    return left;
}

inline Vec2 operator*(Vec2 value, double scale) {
    return {value.x * scale, value.y * scale};
}

inline Vec2 operator*(double scale, Vec2 value) {
    return value * scale;
}

inline double squaredNorm(const Vec2& value) {
    return value.x * value.x + value.y * value.y;
}

inline double norm(const Vec2& value) {
    return std::sqrt(squaredNorm(value));
}

inline double clamp(double value, double lower, double upper) {
    return std::max(lower, std::min(value, upper));
}

struct Rect {
    Vec2 ll;
    Vec2 ur;

    Rect() = default;
    Rect(Vec2 lower_left, Vec2 upper_right) : ll(lower_left), ur(upper_right) {}

    [[nodiscard]] double width() const { return ur.x - ll.x; }
    [[nodiscard]] double height() const { return ur.y - ll.y; }
    [[nodiscard]] double area() const {
        return std::max(0.0, width()) * std::max(0.0, height());
    }
    [[nodiscard]] Vec2 center() const { return {(ll.x + ur.x) * 0.5, (ll.y + ur.y) * 0.5}; }
    [[nodiscard]] bool valid() const { return width() >= -kEpsilon && height() >= -kEpsilon; }

    [[nodiscard]] Rect intersection(const Rect& other) const {
        return {{std::max(ll.x, other.ll.x), std::max(ll.y, other.ll.y)},
                {std::min(ur.x, other.ur.x), std::min(ur.y, other.ur.y)}};
    }

    [[nodiscard]] double overlapArea(const Rect& other) const {
        return intersection(other).area();
    }

    [[nodiscard]] bool overlaps(const Rect& other) const {
        return overlapArea(other) > kEpsilon;
    }
};

}  // namespace myplacement
