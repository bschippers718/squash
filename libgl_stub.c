// Minimal stub for libGL.so.1 to satisfy OpenCV in headless environments
// This provides empty implementations of common GL/GLX functions that OpenCV might try to load
// All symbols are exported to ensure OpenCV can find them

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// Make all functions visible and exportable
#define GLAPI extern
#define GLAPIENTRY

// GLX functions - these are what OpenCV typically needs
GLAPI void* glXGetProcAddress(const char* name) { 
    return NULL; 
}

GLAPI int glXMakeCurrent(void* dpy, void* drawable, void* ctx) { 
    return 0; 
}

GLAPI void* glXCreateContext(void* dpy, void* vis, void* shareList, int direct) { 
    return NULL; 
}

GLAPI void glXDestroyContext(void* dpy, void* ctx) {
    // No-op
}

GLAPI void* glXGetCurrentContext(void) { 
    return NULL; 
}

GLAPI void* glXGetCurrentDisplay(void) { 
    return NULL; 
}

GLAPI int glXQueryExtension(void* dpy, int* errorBase, int* eventBase) { 
    if (errorBase) *errorBase = 0;
    if (eventBase) *eventBase = 0;
    return 0; 
}

GLAPI void* glXChooseVisual(void* dpy, int screen, int* attribList) {
    return NULL;
}

GLAPI int glXSwapBuffers(void* dpy, void* drawable) {
    return 0;
}

GLAPI void glXWaitGL(void) {
    // No-op
}

GLAPI void glXWaitX(void) {
    // No-op
}

GLAPI void* glXGetProcAddressARB(const char* name) {
    return NULL;
}

GLAPI int glXIsDirect(void* dpy, void* ctx) {
    return 0;
}

// Basic GL functions that might be referenced
GLAPI void glBegin(int mode) {}
GLAPI void glEnd(void) {}
GLAPI void glClear(int mask) {}
GLAPI void glFlush(void) {}
GLAPI void glFinish(void) {}

// Library initialization (optional but helps)
__attribute__((constructor))
void init_libgl_stub(void) {
    // Library loaded successfully
}

