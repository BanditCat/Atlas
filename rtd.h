//////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4 et al. //
//////////////////////////////////////////////////////////////////////////////

#include <objbase.h>
#include "SDL2/SDL.h"
#include "SDL2/SDL_syswm.h"

// Global variable to store the WorkerW handle
extern HWND g_WorkerW;

enum {
    RTD_EVENT_SWITCH_TO_WORKERW,
    RTD_EVENT_RETURN_TO_NORMAL
};

SDL_Window* CreateWorkerWWindow( HINSTANCE hInstance,
                          int screenWidth,
                          int screenHeight );
