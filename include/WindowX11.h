#pragma once

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <X11/XKBlib.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>
#include "Color.h"
#include <unordered_map>

class WindowX11 {
private:
    Display* display_{nullptr};
    ::Window window_{0};
    GC gc_{0};
    XImage* image_{nullptr};
    int screen_{0};
    unsigned int width_{0}, height_{0};
    std::vector<uint32_t> buffer_;
    bool running_{true};
    int center_x_{0}, center_y_{0};
    bool mouse_captured_{false};
    std::unordered_map<KeySym, bool> key_states_;

public:
    WindowX11(unsigned int width, unsigned int height, const std::string& title) 
        : width_(width), height_(height), buffer_(width * height) {
        
        display_ = XOpenDisplay(nullptr);
        if (!display_) {
            throw std::runtime_error("Failed to open X display");
        }

        screen_ = DefaultScreen(display_);
        unsigned long black = BlackPixel(display_, screen_);
        unsigned long white = WhitePixel(display_, screen_);

        window_ = XCreateSimpleWindow(display_, DefaultRootWindow(display_),
            0, 0, width_, height_, 2, black, white);

        // Enable required events
        XSelectInput(display_, window_,
            ExposureMask | KeyPressMask | KeyReleaseMask |
            PointerMotionMask | ButtonPressMask | ButtonReleaseMask);

        // Set window title
        XStoreName(display_, window_, title.c_str());

        // Create GC for drawing
        gc_ = XCreateGC(display_, window_, 0, nullptr);

        // Create image for direct buffer drawing
        image_ = XCreateImage(display_, DefaultVisual(display_, screen_),
            24, ZPixmap, 0, (char*)buffer_.data(),
            width_, height_, 32, 0);

        XMapWindow(display_, window_);
        center_x_ = width_ / 2;
        center_y_ = height_ / 2;
    }

    ~WindowX11() {
        if (image_) XDestroyImage(image_);
        if (gc_) XFreeGC(display_, gc_);
        if (window_) XDestroyWindow(display_, window_);
        if (display_) XCloseDisplay(display_);
    }

    void toggleMouseCapture() {
        mouse_captured_ = !mouse_captured_;
        if (mouse_captured_) {
            // Hide cursor and capture mouse
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
            
            // Warp pointer to center
            XWarpPointer(display_, None, window_, 0, 0, 0, 0, center_x_, center_y_);
        } else {
            // Show cursor
            XUndefineCursor(display_, window_);
        }
        XFlush(display_);
    }

    bool processEvents(double& mouse_dx, double& mouse_dy) {
        XEvent event;
        mouse_dx = mouse_dy = 0;

        while (XPending(display_)) {
            XNextEvent(display_, &event);
            switch (event.type) {
                case KeyPress: {
                    KeySym key = XkbKeycodeToKeysym(display_, event.xkey.keycode, 0, 0);
                    if (key == XK_Escape) running_ = false;
                    if (key == XK_Tab) toggleMouseCapture();
                    key_states_[key] = true;
                    break;
                }
                case KeyRelease: {
                    KeySym key = XkbKeycodeToKeysym(display_, event.xkey.keycode, 0, 0);
                    key_states_[key] = false;
                    break;
                }
                case MotionNotify: {
                    if (mouse_captured_) {
                        mouse_dx = event.xmotion.x - center_x_;
                        mouse_dy = event.xmotion.y - center_y_;
                        XWarpPointer(display_, None, window_, 0, 0, 0, 0, center_x_, center_y_);
                    }
                    break;
                }
            }
        }
        return running_;
    }

    void updateScreen(const std::vector<ColorDBL>& pixels) {
        for (size_t i = 0; i < pixels.size(); ++i) {
            const auto& pixel = pixels[i];
            // Ensure proper color conversion and clamping
            uint8_t r = static_cast<uint8_t>(std::clamp(pixel.r() * 255.0, 0.0, 255.0));
            uint8_t g = static_cast<uint8_t>(std::clamp(pixel.g() * 255.0, 0.0, 255.0));
            uint8_t b = static_cast<uint8_t>(std::clamp(pixel.b() * 255.0, 0.0, 255.0));
            
            buffer_[i] = (r << 16) | (g << 8) | b;
        }
        XPutImage(display_, window_, gc_, image_, 0, 0, 0, 0, width_, height_);
        XFlush(display_);
    }

    [[nodiscard]] unsigned int getWidth() const { return width_; }
    [[nodiscard]] unsigned int getHeight() const { return height_; }

    void setTitle(const std::string& title) {
        XStoreName(display_, window_, title.c_str());
        XFlush(display_);
    }

    [[nodiscard]] bool isKeyPressed(KeySym key) const {
        auto it = key_states_.find(key);
        return it != key_states_.end() && it->second;
    }
}; 