#pragma once

#include <cstdint>
#include <algorithm>
#include <concepts>
#include "Vec3.h"

// Forward declarations
class ColorRGB;
class ColorDBL;

// Color class for high-precision calculations during ray tracing
class ColorDBL {
public:
    // Data members (using Vec3d internally for SIMD optimization)
    Vec3d color;

    // Constructors
    constexpr ColorDBL() noexcept : color(0.0, 0.0, 0.0) {}
    constexpr ColorDBL(double r, double g, double b) noexcept : color(r, g, b) {}
    explicit constexpr ColorDBL(const Vec3d& v) noexcept : color(v) {}
    
    // Add copy operations explicitly
    constexpr ColorDBL(const ColorDBL&) noexcept = default;
    constexpr ColorDBL& operator=(const ColorDBL&) noexcept = default;
    
    // Convert from ColorRGB (0-255) to ColorDBL (0.0-1.0)
    explicit ColorDBL(const ColorRGB& rgb) noexcept;

    // Accessors
    [[nodiscard]] constexpr double r() const noexcept { return color.r(); }
    [[nodiscard]] constexpr double g() const noexcept { return color.g(); }
    [[nodiscard]] constexpr double b() const noexcept { return color.b(); }

    // Color operations
    [[nodiscard]] constexpr ColorDBL clamp() const noexcept {
        return ColorDBL(
            std::clamp(color.r(), 0.0, 1.0),
            std::clamp(color.g(), 0.0, 1.0),
            std::clamp(color.b(), 0.0, 1.0)
        );
    }

    // Apply gamma correction
    [[nodiscard]] ColorDBL gamma(double gamma) const noexcept {
        return ColorDBL(
            std::pow(color.r(), 1.0 / gamma),
            std::pow(color.g(), 1.0 / gamma),
            std::pow(color.b(), 1.0 / gamma)
        );
    }

    // Operators
    constexpr ColorDBL& operator+=(const ColorDBL& other) noexcept {
        color += other.color;
        return *this;
    }

    constexpr ColorDBL& operator*=(const ColorDBL& other) noexcept {
        color = hadamard(color, other.color);
        return *this;
    }

    constexpr ColorDBL& operator*=(double scalar) noexcept {
        color *= scalar;
        return *this;
    }

    constexpr ColorDBL& operator/=(double scalar) noexcept {
        color /= scalar;
        return *this;
    }
};

// Color class for final image representation
class ColorRGB {
public:
    // Data members
    uint8_t r{0}, g{0}, b{0};

    // Constructors
    constexpr ColorRGB() noexcept = default;
    constexpr ColorRGB(uint8_t r_, uint8_t g_, uint8_t b_) noexcept : r(r_), g(g_), b(b_) {}
    
    // Convert from ColorDBL (0.0-1.0) to ColorRGB (0-255)
    explicit ColorRGB(const ColorDBL& dbl) noexcept {
        auto clamped = dbl.clamp();
        r = static_cast<uint8_t>(clamped.r() * 255.0 + 0.5);
        g = static_cast<uint8_t>(clamped.g() * 255.0 + 0.5);
        b = static_cast<uint8_t>(clamped.b() * 255.0 + 0.5);
    }

    // Convert to 32-bit packed RGB (useful for image output)
    [[nodiscard]] constexpr uint32_t toRGB32() const noexcept {
        return (static_cast<uint32_t>(r) << 16) |
               (static_cast<uint32_t>(g) << 8) |
               static_cast<uint32_t>(b);
    }
};

// Implementation of ColorDBL's conversion constructor
inline ColorDBL::ColorDBL(const ColorRGB& rgb) noexcept
    : color(rgb.r / 255.0, rgb.g / 255.0, rgb.b / 255.0) {}

// Non-member operators for ColorDBL
[[nodiscard]] constexpr ColorDBL operator+(const ColorDBL& a, const ColorDBL& b) noexcept {
    return ColorDBL(a.color + b.color);
}

[[nodiscard]] constexpr ColorDBL operator*(const ColorDBL& a, const ColorDBL& b) noexcept {
    return ColorDBL(hadamard(a.color, b.color));
}

[[nodiscard]] constexpr ColorDBL operator*(const ColorDBL& c, double scalar) noexcept {
    return ColorDBL(c.color * scalar);
}

[[nodiscard]] constexpr ColorDBL operator*(double scalar, const ColorDBL& c) noexcept {
    return c * scalar;
}

[[nodiscard]] constexpr ColorDBL operator/(const ColorDBL& c, double scalar) noexcept {
    return ColorDBL(c.color / scalar);
}

// Common color constants
namespace Colors {
    constexpr ColorDBL Black(0.0, 0.0, 0.0);
    constexpr ColorDBL White(1.0, 1.0, 1.0);
    constexpr ColorDBL Red(1.0, 0.0, 0.0);
    constexpr ColorDBL Green(0.0, 1.0, 0.0);
    constexpr ColorDBL Blue(0.0, 0.0, 1.0);
} 