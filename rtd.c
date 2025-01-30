//////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4 et al. //
//////////////////////////////////////////////////////////////////////////////

#include "rtd.h"
#include <stdio.h>

// Global variable to store the WorkerW handle
HWND g_WorkerW = NULL;


// Callback function for EnumWindows to find the WorkerW window
BOOL CALLBACK EnumWindowsProc( HWND hwnd, LPARAM lParam ){
  HWND shellViewWin = FindWindowEx( hwnd, NULL, "SHELLDLL_DefView", NULL );
  if( shellViewWin != NULL ){
    // Get the WorkerW window which is the next window after SHELLDLL_DefView
    HWND workerw = FindWindowEx( NULL, hwnd, "WorkerW", NULL );
    if( workerw != NULL ){
      HWND* pWorkerW = (HWND*)lParam;
      *pWorkerW = workerw;
      return FALSE;  // Stop enumeration
    }
  }
  return TRUE;  // Continue enumeration
}

// Simple Window Procedure for the child window
LRESULT CALLBACK ChildWndProc( HWND hwnd,
                               UINT msg,
                               WPARAM wParam,
                               LPARAM lParam ){
  switch( msg ){
  case WM_DESTROY:
    PostQuitMessage( 0 );
    return 0;
  // Handle other messages as needed
  default:
    return DefWindowProc( hwnd, msg, wParam, lParam );
  }
}

// Function to create and return the HWND for OpenGL rendering
HWND CreateWorkerWWindow( HINSTANCE hInstance,
                          int screenWidth,
                          int screenHeight ){
  HRESULT hr =
    CoInitializeEx( NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE );
  if( FAILED( hr ) ){
    printf( "CoInitializeEx failed: %ld\n", hr );
    return NULL;
  }

  // Check if a WorkerW window already exists
  g_WorkerW = FindWindow( "WorkerW", NULL );
  if( g_WorkerW == NULL ){
    // Retrieve the Progman window
    HWND progman = FindWindow( "Progman", NULL );
    if( progman == NULL ){
      printf( "Failed to find Progman window.\n" );
      CoUninitialize();
      return NULL;
    }

    // Send message 0x052C to Progman to spawn a WorkerW window
    SendMessageTimeout( progman, 0x052C, 0, 0, SMTO_NORMAL, 1000, NULL );

    // Enumerate windows to find the new WorkerW
    EnumWindows( EnumWindowsProc, (LPARAM)&g_WorkerW );

    if( g_WorkerW == NULL ){
      printf( "Failed to find WorkerW window after sending message.\n" );
      CoUninitialize();
      return NULL;
    }
  }

  // Define and register the window class for the child window
  const char CLASS_NAME[] = "MyWorkerWChildWindowClass";

  WNDCLASS wc = { 0 };
  wc.lpfnWndProc = ChildWndProc;
  wc.hInstance = hInstance;
  wc.lpszClassName = CLASS_NAME;

  if( !RegisterClass( &wc ) ){
    printf( "RegisterClass failed: %ld\n", GetLastError() );
    CoUninitialize();
    return NULL;
  }

  // Create the child window as a child of WorkerW
  HWND hwndChild = CreateWindowEx( 0,
                                   CLASS_NAME,
                                   NULL,
                                   WS_CHILD | WS_VISIBLE,
                                   0,
                                   0,
                                   screenWidth,
                                   screenHeight,
                                   g_WorkerW,
                                   NULL,
                                   hInstance,
                                   NULL );

  if( hwndChild == NULL ){
    printf( "CreateWindowEx failed: %ld\n", GetLastError() );
    UnregisterClass( CLASS_NAME, hInstance );
    CoUninitialize();
    return NULL;
  }

  // Optionally, hide the original WorkerW window if it's visible
  // ShowWindow(g_WorkerW, SW_HIDE);

  // Uninitialize COM (optional here, depending on further COM usage)
  CoUninitialize();

  return hwndChild;
}
