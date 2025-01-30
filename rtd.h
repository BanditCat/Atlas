//////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4 et al. //
//////////////////////////////////////////////////////////////////////////////

#include <objbase.h>
// Global variable to store the WorkerW handle
extern HWND g_WorkerW;

HWND CreateWorkerWWindow( HINSTANCE hInstance,
                          int screenWidth,
                          int screenHeight );
