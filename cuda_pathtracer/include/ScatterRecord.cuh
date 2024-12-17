#pragma once

#include "Ray.cuh"
#include "Color.cuh"

struct ScatterRecord_t {
    Ray_t scattered_ray;
    Color_t attenuation;
    float pdf;
    bool is_specular;

    __device__ ScatterRecord_t()
        : scattered_ray()
        , attenuation()
        , pdf(0.0f)
        , is_specular(false)
    {}
}; 