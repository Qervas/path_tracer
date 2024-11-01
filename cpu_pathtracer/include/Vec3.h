#pragma once

#include <cmath>
#include <concepts>
#include <format>
#include <iostream>

template<typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

template<FloatingPoint T>
class Vec3 {
public:
    T x{0}, y{0}, z{0};  

    // Constructors
    constexpr Vec3() noexcept = default;
    constexpr Vec3(T value) noexcept : x(value), y(value), z(value) {}
    constexpr Vec3(T x_, T y_, T z_) noexcept : x(x_), y(y_), z(z_) {}
    constexpr Vec3(const Vec3& other) noexcept = default;
    constexpr Vec3& operator=(const Vec3& other) noexcept = default;

    // Color component accessors
    [[nodiscard]] constexpr T& r() noexcept { return x; }
    [[nodiscard]] constexpr T& g() noexcept { return y; }
    [[nodiscard]] constexpr T& b() noexcept { return z; }
    [[nodiscard]] constexpr const T& r() const noexcept { return x; }
    [[nodiscard]] constexpr const T& g() const noexcept { return y; }
    [[nodiscard]] constexpr const T& b() const noexcept { return z; }

    // Array access
    [[nodiscard]] constexpr T& operator[](size_t i) noexcept {
        return (&x)[i];  // Use pointer arithmetic since members are contiguous
    }
    [[nodiscard]] constexpr const T& operator[](size_t i) const noexcept {
        return (&x)[i];
    }

    // Vector operations
    [[nodiscard]] constexpr T length_squared() const noexcept {
        return x * x + y * y + z * z;
    }

    [[nodiscard]] T length() const noexcept {
        return std::sqrt(length_squared());
    }

    constexpr Vec3& normalize() noexcept {
        T len = length();
        if (len > 0) {
            T inv_len = 1 / len;
            x *= inv_len;
            y *= inv_len;
            z *= inv_len;
        }
        return *this;
    }

    [[nodiscard]] constexpr Vec3 normalized() const noexcept {
        Vec3 result = *this;
        result.normalize();
        return result;
    }

    // Arithmetic operators
    constexpr Vec3& operator+=(const Vec3& v) noexcept {
        x += v.x; y += v.y; z += v.z;
        return *this;
    }

    constexpr Vec3& operator-=(const Vec3& v) noexcept {
        x -= v.x; y -= v.y; z -= v.z;
        return *this;
    }

    constexpr Vec3& operator*=(T scalar) noexcept {
        x *= scalar; y *= scalar; z *= scalar;
        return *this;
    }

    constexpr Vec3& operator/=(T scalar) noexcept {
        T inv_scalar = 1 / scalar;
        return *this *= inv_scalar;
    }

    [[nodiscard]] constexpr Vec3 operator-() const noexcept {
        return Vec3(-x, -y, -z);
    }
};

// Type aliases - only for floating point types
using Vec3f = Vec3<float>;
using Vec3d = Vec3<double>;
using Point3f = Vec3<float>;
using Point3d = Vec3<double>;

// Non-member functions
template<FloatingPoint T>
[[nodiscard]] constexpr Vec3<T> operator+(const Vec3<T>& u, const Vec3<T>& v) noexcept {
    return Vec3<T>(u.x + v.x, u.y + v.y, u.z + v.z);
}

template<FloatingPoint T>
[[nodiscard]] constexpr Vec3<T> operator-(const Vec3<T>& u, const Vec3<T>& v) noexcept {
    return Vec3<T>(u.x - v.x, u.y - v.y, u.z - v.z);
}

template<FloatingPoint T>
[[nodiscard]] constexpr Vec3<T> operator*(const Vec3<T>& v, T scalar) noexcept {
    return Vec3<T>(v.x * scalar, v.y * scalar, v.z * scalar);
}

template<FloatingPoint T>
[[nodiscard]] constexpr Vec3<T> operator*(T scalar, const Vec3<T>& v) noexcept {
    return v * scalar;
}

template<FloatingPoint T>
[[nodiscard]] constexpr Vec3<T> operator/(const Vec3<T>& v, T scalar) noexcept {
    return v * (1 / scalar);
}

template<FloatingPoint T>
[[nodiscard]] constexpr Vec3<T> hadamard(const Vec3<T>& u, const Vec3<T>& v) noexcept {
    return Vec3<T>(u.x * v.x, u.y * v.y, u.z * v.z);
}

template<FloatingPoint T>
[[nodiscard]] constexpr T dot(const Vec3<T>& u, const Vec3<T>& v) noexcept {
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

template<FloatingPoint T>
[[nodiscard]] constexpr Vec3<T> cross(const Vec3<T>& u, const Vec3<T>& v) noexcept {
    return Vec3<T>(
        u.y * v.z - u.z * v.y,
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x
    );
}

template<FloatingPoint T>
[[nodiscard]] constexpr Vec3<T> reflect(const Vec3<T>& v, const Vec3<T>& n) noexcept {
    return v - 2 * dot(v, n) * n;
}

template<FloatingPoint T>
[[nodiscard]] Vec3<T> refract(const Vec3<T>& uv, const Vec3<T>& n, T etai_over_etat) noexcept {
    T cos_theta = std::min(dot(-uv, n), T(1));
    Vec3<T> r_out_perp = etai_over_etat * (uv + cos_theta * n);
    Vec3<T> r_out_parallel = -std::sqrt(std::abs(1 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

// I/O operations
template<FloatingPoint T>
std::ostream& operator<<(std::ostream& out, const Vec3<T>& v) {
    return out << std::format("({}, {}, {})", v.x, v.y, v.z);
}

template<FloatingPoint T>
[[nodiscard]] constexpr Vec3<T> min(const Vec3<T>& a, const Vec3<T>& b) noexcept {
    return Vec3<T>(
        std::min(a.x, b.x),
        std::min(a.y, b.y),
        std::min(a.z, b.z)
    );
}

template<FloatingPoint T>
[[nodiscard]] constexpr Vec3<T> max(const Vec3<T>& a, const Vec3<T>& b) noexcept {
    return Vec3<T>(
        std::max(a.x, b.x),
        std::max(a.y, b.y),
        std::max(a.z, b.z)
    );
}