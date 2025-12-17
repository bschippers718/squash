// Minimal stub for libGL.so.1 to satisfy OpenCV in headless environments
// This provides empty implementations of common GL functions that OpenCV might try to load

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// Stub implementations that return safe defaults
// These functions are called by OpenCV's C extension during initialization
// We provide minimal implementations that prevent crashes

void* glXGetProcAddress(const char* name) { 
    return NULL; 
}

int glXMakeCurrent(void* dpy, void* drawable, void* ctx) { 
    return 0; 
}

void* glXCreateContext(void* dpy, void* vis, void* shareList, int direct) { 
    return NULL; 
}

void glXDestroyContext(void* dpy, void* ctx) {
    // No-op
}

void* glXGetCurrentContext(void) { 
    return NULL; 
}

void* glXGetCurrentDisplay(void) { 
    return NULL; 
}

int glXQueryExtension(void* dpy, int* errorBase, int* eventBase) { 
    if (errorBase) *errorBase = 0;
    if (eventBase) *eventBase = 0;
    return 0; 
}

// Additional common GLX functions that might be referenced
void* glXChooseVisual(void* dpy, int screen, int* attribList) {
    return NULL;
}

int glXSwapBuffers(void* dpy, void* drawable) {
    return 0;
}

void glXWaitGL(void) {
    // No-op
}

void glXWaitX(void) {
    // No-op
}

