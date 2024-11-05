#pragma once

#include <cuda_runtime.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <X11/XKBlib.h>
#include <cstdint>
#include "Vec3.cuh"
#include "Color.cuh"
#include "Error.cuh"



// Move the kernel function outside the class
__global__ void convertToRGBAKernel(const float4* input, uint32_t* output, uint32_t width, uint32_t height) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    uint32_t idx = y * width + x;
    float4 pixel = input[idx];
    
    uint8_t r = static_cast<uint8_t>(min(max(pixel.x * 255.0f, 0.0f), 255.0f));
    uint8_t g = static_cast<uint8_t>(min(max(pixel.y * 255.0f, 0.0f), 255.0f));
    uint8_t b = static_cast<uint8_t>(min(max(pixel.z * 255.0f, 0.0f), 255.0f));
    
    output[idx] = (r << 16) | (g << 8) | b;
}

class Window_t {
private:
    Display* display_;
    ::Window window_;
    GC gc_;
    XImage* image_;
    int screen_;
    uint32_t width_, height_;
    uint32_t* d_buffer_;     // Device buffer
    uint32_t* h_buffer_;     // Host buffer
    bool running_;
    int center_x_, center_y_;
    bool mouse_captured_;
    bool key_states_[256];   // Simplified key state tracking

public:
    __host__ Window_t(uint32_t width, uint32_t height, const char* title) 
        : width_(width)
        , height_(height)
        , running_(true)
        , mouse_captured_(false)
    {
        // Initialize X11 window
        display_ = XOpenDisplay(nullptr);
        if (!display_) {
            throw "Failed to open X display";
        }

        screen_ = DefaultScreen(display_);
        window_ = XCreateSimpleWindow(display_, DefaultRootWindow(display_),
            0, 0, width_, height_, 2,
            BlackPixel(display_, screen_),
            WhitePixel(display_, screen_));

        XSelectInput(display_, window_,
            ExposureMask | KeyPressMask | KeyReleaseMask |
            PointerMotionMask | ButtonPressMask | ButtonReleaseMask);

        XStoreName(display_, window_, title);
        gc_ = XCreateGC(display_, window_, 0, nullptr);

        // Allocate buffers with error checking
        h_buffer_ = new uint32_t[width_ * height_];
        if (!h_buffer_) {
            throw "Failed to allocate host memory";
        }

        CUDA_CHECK(cudaMalloc(&d_buffer_, width_ * height_ * sizeof(uint32_t)));
        if (!d_buffer_) {
            delete[] h_buffer_;
            throw "Failed to allocate device memory";
        }

        image_ = XCreateImage(display_, DefaultVisual(display_, screen_),
            24, ZPixmap, 0, (char*)h_buffer_,
            width_, height_, 32, 0);

        XMapWindow(display_, window_);
        center_x_ = width_ / 2;
        center_y_ = height_ / 2;

        memset(key_states_, 0, sizeof(key_states_));
    }

    __host__ ~Window_t() {
        if (image_) XDestroyImage(image_);
        if (gc_) XFreeGC(display_, gc_);
        if (window_) XDestroyWindow(display_, window_);
        if (display_) XCloseDisplay(display_);
        if (h_buffer_) delete[] h_buffer_;
        if (d_buffer_) cudaFree(d_buffer_);
    }

    __host__ void toggleMouseCapture() {
        mouse_captured_ = !mouse_captured_;
        if (mouse_captured_) {
            // Create and set invisible cursor
            Cursor invisibleCursor;
            Pixmap bitmapNoData;
            XColor black;
            static char noData[] = { 0,0,0,0,0,0,0,0 };
            black.red = black.green = black.blue = 0;

            bitmapNoData = XCreateBitmapFromData(display_, window_, noData, 8, 8);
            invisibleCursor = XCreatePixmapCursor(display_, bitmapNoData, bitmapNoData,
                                                &black, &black, 0, 0);
            XDefineCursor(display_, window_, invisibleCursor);
            XFreeCursor(display_, invisibleCursor);
            XFreePixmap(display_, bitmapNoData);
            
            XWarpPointer(display_, None, window_, 0, 0, 0, 0, center_x_, center_y_);
        } else {
            XUndefineCursor(display_, window_);
        }
        XFlush(display_);
    }

    __host__ bool processEvents(float& mouse_dx, float& mouse_dy) {
        XEvent event;
        mouse_dx = mouse_dy = 0;

        while (XPending(display_)) {
            XNextEvent(display_, &event);
            switch (event.type) {
                case KeyPress: {
                    KeySym key = XkbKeycodeToKeysym(display_, event.xkey.keycode, 0, 0);
                    if (key == XK_Escape) running_ = false;
                    if (key == XK_Tab) toggleMouseCapture();
                    if (key < 256) key_states_[key] = true;
                    break;
                }
                case KeyRelease: {
                    KeySym key = XkbKeycodeToKeysym(display_, event.xkey.keycode, 0, 0);
                    if (key < 256) key_states_[key] = false;
                    break;
                }
                case MotionNotify: {
                    if (mouse_captured_) {
                        mouse_dx = static_cast<float>(event.xmotion.x - center_x_);
                        mouse_dy = static_cast<float>(event.xmotion.y - center_y_);
                        XWarpPointer(display_, None, window_, 0, 0, 0, 0, center_x_, center_y_);
                    }
                    break;
                }
            }
        }
        return running_;
    }

    // method to call the kernel
    void convertToRGBA(const float4* input, uint32_t* output) {
        dim3 block(16, 16);
        dim3 grid((width_ + block.x - 1) / block.x, 
                 (height_ + block.y - 1) / block.y);
        
        convertToRGBAKernel<<<grid, block>>>(input, output, width_, height_);
    }

    __host__ void updateScreen(const float4* d_pixels) {
        // Convert float4 to uint32_t on GPU
        convertToRGBA(d_pixels, d_buffer_);
        
        // Copy to host
        cudaMemcpy(h_buffer_, d_buffer_, width_ * height_ * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        
        // Update X11 window
        XPutImage(display_, window_, gc_, image_, 0, 0, 0, 0, width_, height_);
        XFlush(display_);
    }

    __host__ uint32_t getWidth() const { return width_; }
    __host__ uint32_t getHeight() const { return height_; }

    __host__ void setTitle(const char* title) {
        XStoreName(display_, window_, title);
        XFlush(display_);
    }

    __host__ bool isKeyPressed(KeySym key) const {
        return key < 256 && key_states_[key];
    }

    // Get device buffer pointer for direct rendering
    __host__ uint32_t* getDeviceBuffer() const { return d_buffer_; }
}; 