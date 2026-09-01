#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

namespace myplacement::test {

inline void expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

inline void expectNear(double actual, double expected, double tolerance, const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message + " (expected " + std::to_string(expected) + ", got " +
                                 std::to_string(actual) + ")");
    }
}

}  // namespace myplacement::test
