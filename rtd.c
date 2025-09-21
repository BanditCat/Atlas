//////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4 et al. //
//////////////////////////////////////////////////////////////////////////////

#include "rtd.h"
#include <stdio.h>
#include <windows.h>
#include <stdio.h>
#include "SDL2/SDL.h"
#include "SDL2/SDL_syswm.h"

// Tell Explorer to spawn the hidden WorkerW behind icons:
static void EnsureRealWorkerW() {
    HWND progman = FindWindow("Progman", "Program Manager");
    if (progman) {
        // 0x052C = "RegisterShellHookWindow" message forcing the creation
        // of a WorkerW behind icons, if not already present
        SendMessage(progman, 0x052C, 0, 0);
    }
}

// Callback to find the WorkerW that is the sibling of SHELLDLL_DefView
static BOOL CALLBACK EnumWindowsProc(HWND topHandle, LPARAM lParam) {
    HWND shellView = FindWindowEx(topHandle, NULL, "SHELLDLL_DefView", NULL);
    if (shellView) {
        // The WorkerW is usually the next window after SHELLDLL_DefView
        HWND workerW = FindWindowEx(NULL, topHandle, "WorkerW", NULL);
        if (workerW) {
            HWND *pOut = (HWND*)lParam;
            *pOut = workerW;
            return FALSE; // stop enumerating
        }
    }
    return TRUE; // keep going
}

// Actually find that real “behind icons” WorkerW
static HWND FindBehindIconsWorkerW() {
    // Make sure Explorer has created the WorkerW
    EnsureRealWorkerW();

    HWND workerW = NULL;
    EnumWindows(EnumWindowsProc, (LPARAM)&workerW);
    if (!workerW) {
        printf("Could not find the 'real' WorkerW behind icons.\n");
    }
    return workerW;
}

/**
 * Creates an SDL window and re-parents it under the real
 * WorkerW (the one behind the desktop icons).
 */
SDL_Window* CreateWorkerWWindow(HINSTANCE hInstance, int screenWidth, int screenHeight)
{
    // 1) Create an SDL window with OPENGL | BORDERLESS, initially hidden
    //    so we can safely adjust styles before showing it
    SDL_Window* sdlWindow = SDL_CreateWindow(
        "Atlas WorkerW",
        0, // X
        0, // Y
        screenWidth,
        screenHeight,
        SDL_WINDOW_OPENGL | SDL_WINDOW_BORDERLESS | SDL_WINDOW_HIDDEN
    );

    if (!sdlWindow) {
        printf("SDL_CreateWindow failed: %s\n", SDL_GetError());
        return NULL;
    }

    // 2) Grab the native HWND from SDL
    SDL_SysWMinfo wmInfo;
    SDL_VERSION(&wmInfo.version);
    if (!SDL_GetWindowWMInfo(sdlWindow, &wmInfo)) {
        printf("SDL_GetWindowWMInfo failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(sdlWindow);
        return NULL;
    }

    HWND hwnd = wmInfo.info.win.window;
    if (!hwnd) {
        printf("Could not retrieve native HWND from SDL window.\n");
        SDL_DestroyWindow(sdlWindow);
        return NULL;
    }

    // 3) Locate the real WorkerW behind icons
    HWND workerw = FindBehindIconsWorkerW();
    if (!workerw) {
        // If this fails, we won't have a real behind-icons WorkerW
        SDL_DestroyWindow(sdlWindow);
        return NULL;
    }

    // 4) Adjust the extended style: WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW, etc.
    LONG_PTR exStyle = GetWindowLongPtr(hwnd, GWL_EXSTYLE);
    exStyle |= (WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW);
    SetWindowLongPtr(hwnd, GWL_EXSTYLE, exStyle);

    // 5) Adjust the normal style: add WS_CHILD, remove WS_POPUP, etc.
    LONG_PTR style = GetWindowLongPtr(hwnd, GWL_STYLE);
    style |= WS_CHILD;
    style &= ~WS_POPUP;
    SetWindowLongPtr(hwnd, GWL_STYLE, style);

    // 6) Re-parent the SDL window under WorkerW
    SetParent(hwnd, workerw);

    // 7) Force style changes to take effect (and position to 0,0 if you wish)
    SetWindowPos(
        hwnd,
        NULL,
        0, 0, screenWidth, screenHeight,
        SWP_FRAMECHANGED | SWP_NOZORDER | SWP_NOOWNERZORDER
    );

    // 8) Finally, show the (now child) SDL window
    SDL_ShowWindow(sdlWindow);

    return sdlWindow;
}


/* // Global variable to store the WorkerW handle */
/* HWND g_WorkerW = NULL; */

/* // Callback function for EnumWindows to find the WorkerW window */
/* BOOL CALLBACK EnumWindowsProc(HWND hwnd, LPARAM lParam) { */
/*     HWND shellViewWin = FindWindowEx(hwnd, NULL, "SHELLDLL_DefView", NULL); */
/*     if (shellViewWin != NULL) { */
/*         // The WorkerW window is the next window after SHELLDLL_DefView */
/*         HWND workerw = FindWindowEx(NULL, hwnd, "WorkerW", NULL); */
/*         if (workerw != NULL) { */
/*             HWND* pWorkerW = (HWND*)lParam; */
/*             *pWorkerW = workerw; */
/*             return FALSE;  // Stop enumeration */
/*         } */
/*     } */
/*     return TRUE;  // Continue enumeration */
/* } */
/* #include <windows.h> */
/* #include <stdio.h> */
/* #include "SDL2/SDL.h" */

/* // Window Procedure */
/* LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) { */
/*     switch (uMsg) { */
/*         case WM_DESTROY: */
/*             PostQuitMessage(0); */
/*             return 0; */
/*         // Handle other messages as needed */
/*         default: */
/*             return DefWindowProc(hwnd, uMsg, wParam, lParam); */
/*     } */
/* } */
/* static HWND FindWorkerW(void) */
/* { */
/*     // Simple direct approach; you may prefer a more robust check */
/*     HWND workerw = FindWindow("WorkerW", NULL); */
/*     if (!workerw) { */
/*         printf("Failed to find WorkerW window.\n"); */
/*     } */
/*     return workerw; */
/* } */

/* SDL_Window* CreateWorkerWWindow(HINSTANCE hInstance, int screenWidth, int screenHeight) */
/* { */
/*     // 1) Create an SDL window with OPENGL | BORDERLESS, initially hidden */
/*     //    so we can safely adjust styles before showing it */
/*     SDL_Window* sdlWindow = SDL_CreateWindow( */
/*         "Atlas WorkerW", */
/*         0,                     // X */
/*         0,                     // Y */
/*         screenWidth, */
/*         screenHeight, */
/*         SDL_WINDOW_OPENGL | SDL_WINDOW_BORDERLESS | SDL_WINDOW_HIDDEN */
/*     ); */

/*     if (!sdlWindow) { */
/*         printf("SDL_CreateWindow failed: %s\n", SDL_GetError()); */
/*         return NULL; */
/*     } */

/*     // 2) Grab the native HWND from SDL */
/*     SDL_SysWMinfo wmInfo; */
/*     SDL_VERSION(&wmInfo.version); */
/*     if (!SDL_GetWindowWMInfo(sdlWindow, &wmInfo)) { */
/*         printf("SDL_GetWindowWMInfo failed: %s\n", SDL_GetError()); */
/*         SDL_DestroyWindow(sdlWindow); */
/*         return NULL; */
/*     } */

/*     HWND hwnd = wmInfo.info.win.window; */
/*     if (!hwnd) { */
/*         printf("Could not retrieve native HWND from SDL window.\n"); */
/*         SDL_DestroyWindow(sdlWindow); */
/*         return NULL; */
/*     } */

/*     // 3) Locate the existing WorkerW */
/*     HWND workerw = FindWorkerW(); */
/*     if (!workerw) { */
/*         SDL_DestroyWindow(sdlWindow); */
/*         return NULL; */
/*     } */

/*     // 4) Adjust the extended style: WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW, etc. */
/*     LONG_PTR exStyle = GetWindowLongPtr(hwnd, GWL_EXSTYLE); */
/*     exStyle |= (WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW); */
/*     SetWindowLongPtr(hwnd, GWL_EXSTYLE, exStyle); */

/*     // 5) Adjust the normal style: add WS_CHILD, remove WS_POPUP, etc. */
/*     LONG_PTR style = GetWindowLongPtr(hwnd, GWL_STYLE); */
/*     style |= WS_CHILD; */
/*     style &= ~WS_POPUP; */
/*     SetWindowLongPtr(hwnd, GWL_STYLE, style); */

/*     // 6) Re-parent the SDL window under WorkerW */
/*     //    This makes it appear "behind" the desktop icons (depending on how WorkerW is used). */
/*     SetParent(hwnd, workerw); */

/*     // 7) Force style changes to take effect (and position to 0,0 if you wish) */
/*     SetWindowPos( */
/*         hwnd, */
/*         NULL, */
/*         0, 0, screenWidth, screenHeight, */
/*         SWP_FRAMECHANGED | SWP_NOZORDER | SWP_NOOWNERZORDER */
/*     ); */

/*     // 8) Finally, show the (now child) SDL window */
/*     SDL_ShowWindow(sdlWindow); */

/*     return sdlWindow; */
/* } */
/* SDL_Window* CreateWorkerWWindow(HINSTANCE hInstance, int screenWidth, int screenHeight) { */
/*     // Register the window class */
/*     WNDCLASSEX wc = {0}; */
/*     wc.cbSize = sizeof(WNDCLASSEX); */
/*     wc.style = CS_HREDRAW | CS_VREDRAW; */
/*     wc.lpfnWndProc = WindowProc; */
/*     wc.hInstance = hInstance; */
/*     wc.hCursor = LoadCursor(NULL, IDC_ARROW); */
/*     wc.lpszClassName = "CustomWindowClass"; */
/*     if (!RegisterClassEx(&wc)) { */
/*         printf("RegisterClassEx failed: %lu\n", GetLastError()); */
/*         return NULL; */
/*     } */


/*         // Find the WorkerW window */
/*     HWND workerw = FindWindow("WorkerW", NULL); */
/*     if (!workerw) { */
/*       printf("Failed to find WorkerW window.\n"); */
/*       return NULL; */
/*     } */

/*     // Create the native window */
/*     HWND hwnd = CreateWindowEx( */
/*         WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW, */
/*         wc.lpszClassName, */
/*         "Custom Window", */
/*         WS_VISIBLE, */
/*         0, 0, */
/*         screenWidth, screenHeight, */
/*         workerw, */
/*         NULL, */
/*         hInstance, */
/*         NULL */
/*     ); */

/*     LONG_PTR style = GetWindowLongPtr(hwnd, GWL_STYLE); */
/*     style |= WS_CHILD; */
/*     SetWindowLongPtr(hwnd, GWL_STYLE, style); */
/*    SetWindowPos( */
/*         hwnd, */
/*         NULL, */
/*         0, 0, 0, 0, */
/*         SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED */
/*     ); */
/*     if (!hwnd) { */
/*         printf("CreateWindowEx failed: %lu\n", GetLastError()); */
/*         return NULL; */
/*     } */
/*     // Associate the native window with SDL */
/*     SDL_Window* sdlWindow = SDL_CreateWindowFrom((void*)hwnd); */
/*     if (!sdlWindow) { */
/*         printf("SDL_CreateWindowFrom failed: %s\n", SDL_GetError()); */
/*         DestroyWindow(hwnd); */
/*         return NULL; */
/*     } */


/*     // Show the SDL window */
/*     SDL_ShowWindow(sdlWindow); */

/*     return sdlWindow; */
/* } */

/* HWND CreateWorkerWWindow(HINSTANCE hInstance, int screenWidth, int screenHeight) */
/* { */
/*     // 1) Initialize COM, if needed */
/*     HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE); */
/*     if (FAILED(hr)) { */
/*         printf("CoInitializeEx failed: %ld\n", hr); */
/*         return NULL; */
/*     } */

/*     // 2) Find or spawn WorkerW */
/*     g_WorkerW = FindWindow("WorkerW", NULL); */
/*     if (g_WorkerW == NULL) { */
/*         HWND progman = FindWindow("Progman", NULL); */
/*         if (!progman) { */
/*             printf("Failed to find Progman window.\n"); */
/*             CoUninitialize(); */
/*             return NULL; */
/*         } */
/*         // Send message 0x052C to make a new WorkerW */
/*         SendMessageTimeout(progman, 0x052C, 0, 0, SMTO_NORMAL, 1000, NULL); */

/*         // Enumerate windows to find the new WorkerW */
/*         EnumWindows(EnumWindowsProc, (LPARAM)&g_WorkerW); */
/*         if (!g_WorkerW) { */
/*             printf("Failed to find WorkerW after sending message.\n"); */
/*             CoUninitialize(); */
/*             return NULL; */
/*         } */
/*     } */

/*     // 3) Create an SDL window with OpenGL + hidden + borderless */
/*     //    so we can manipulate styles before showing it */
/*     SDL_Window* sdlWindow = SDL_CreateWindow( */
/*         "Atlas-WorkerW", */
/*         0, */
/*         0, */
/*         screenWidth, */
/*         screenHeight, */
/*         SDL_WINDOW_OPENGL | SDL_WINDOW_BORDERLESS | SDL_WINDOW_HIDDEN */
/*     ); */
/*     if (!sdlWindow) { */
/*         printf("SDL_CreateWindow error: %s\n", SDL_GetError()); */
/*         CoUninitialize(); */
/*         return NULL; */
/*     } */


/*     // 4) Grab the HWND for style manipulation */
/*     SDL_SysWMinfo wmInfo; */
/*     SDL_VERSION(&wmInfo.version); */
/*     if (!SDL_GetWindowWMInfo(sdlWindow, &wmInfo)) { */
/*         printf("SDL_GetWindowWMInfo error: %s\n", SDL_GetError()); */
/*         SDL_DestroyWindow(sdlWindow); */
/*         CoUninitialize(); */
/*         return NULL; */
/*     } */
/*     HWND nativeHwnd = wmInfo.info.win.window; */
/*     if (!nativeHwnd) { */
/*         printf("Failed to get native HWND from SDL window.\n"); */
/*         SDL_DestroyWindow(sdlWindow); */
/*         CoUninitialize(); */
/*         return NULL; */
/*     } */
/*     // 5) Remove conflicting styles from the SDL-created window */
/*     //    that might block SetParent under certain Windows configs */
/*     LONG style = GetWindowLong(nativeHwnd, GWL_STYLE); */
/*     // Remove caption, sysmenu, etc. We want a plain popup */
/*     style &= ~(WS_CAPTION | WS_SYSMENU | WS_THICKFRAME | WS_MINIMIZE | WS_MAXIMIZE); */
/*     // Typically WS_POPUP remains, or we might ensure it's set */
/*     style = WS_POPUP | WS_VISIBLE; */

/*     SetWindowLong(nativeHwnd, GWL_STYLE, style); */


/*     // Also consider removing or adding extended styles: */
/*     LONG exStyle = GetWindowLong(nativeHwnd, GWL_EXSTYLE); */
/*     // For behind icons, sometimes WS_EX_TOOLWINDOW helps */
/*     exStyle = WS_EX_TOOLWINDOW; */
/*     SetWindowLong(nativeHwnd, GWL_EXSTYLE, exStyle); */

/*     // Force style changes to take effect */
/*     SetWindowPos( */
/*         nativeHwnd, */
/*         NULL, */
/*         0, 0, 0, 0, */
/*         SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_FRAMECHANGED */
/*     ); */

/*     // 6) Now attempt to re-parent the SDL window under WorkerW */
/*     if (!IsWindow(nativeHwnd) || !IsWindow(g_WorkerW) ){ */
/*       printf("One or both window handles are invalid.\n"); */
/*       return NULL; */
/*     } */
/*     HWND oldParent = SetParent(nativeHwnd, g_WorkerW); */
/*     if (!oldParent) { */
/*         DWORD errCode = GetLastError(); */
/*         printf("SetParent failed. GetLastError: %lu\n", errCode); */
/*         SDL_DestroyWindow(sdlWindow); */
/*         CoUninitialize(); */
/*         return NULL; */
/*     } */

/*     // 7) Show the SDL window behind icons */
/*     SDL_ShowWindow(sdlWindow); */

/*     // (Optional) Hide WorkerW itself if you like */
/*     // ShowWindow(g_WorkerW, SW_HIDE); */

/*     // 8) Done with COM if you don’t need it */
/*     CoUninitialize(); */

/*     // 9) Return the newly created native HWND */
/*     return nativeHwnd; */
/* } */
