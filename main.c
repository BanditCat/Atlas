////////////////////////////////////////////////////////////////////////////////
// Copyright © 2025 Jon DuBois. Written with the assistance of GPT-4 et al.   //
////////////////////////////////////////////////////////////////////////////////

#include "Atlas.h"

#ifndef __EMSCRIPTEN__
#include "SDL2/SDL_syswm.h"
#include <dwmapi.h>
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

#define STARTTEXT                                       \
  "      N\n"                                           \
  "      |\n"                                           \
  "   NW | NE\n"                                        \
  "     \\|/\n"                                         \
  " W----+----E                   Welcome to Atlas!\n"  \
  "     /|\\\n"                                         \
  "   SW | SE\n"                                        \
  "      |\n"                                           \
  "      S\n"

////////////////////////////////////////////////////////////////////
// Global state

u32 jsWidth = 0;
u32 jsHeight = 0;
bool depthTest = false;
bool additive = false;
GLuint vao = 0;
SDL_GameController* controllers[ MAX_CONTROLLERS ] = { NULL };
SDL_JoystickID joystickIDs[ MAX_CONTROLLERS ] = { -1 };
f32 joysticks[ MAX_CONTROLLERS * 21 ] = { 0 };
u32 buttons = 0;
f32 dx = 0;
f32 dy = 0;
f32 posx = 0;
f32 posy = 0;
f32 mouseWheel = 0;
f32 mouseWheelPos = 0.0;
u8 keys[ SDL_NUM_SCANCODES ] = { 0 };
// We set this in main to a mallocd empty string then also deallocate it in
// main.
char* workspace = NULL;

bool doubleClicks[ 3 ] = { 0 };
bool touchClicks[ 3 ] = { 0 };
float pinchZoom = 0.0;

// Main must define these.

#ifdef DEBUG
MemAlloc* mem_list = NULL;
SDL_mutex* mem_list_mutex = NULL;  // Use SDL mutex
#else
u64 memc = 0;
#endif

SDL_Window* window = NULL;
SDL_GLContext glContext;

// The compiled program to be run.
program* prog;
tensorStack* ts;

// Global variables
GLuint shaderProgram;
GLuint vbo;
u64 curTime = 0;
u64 prevTime = 0;
u64 startTime = 0;
f64 runTime = 0.0;
f64 timeDelta = 0.01;
f64 rawFrameTime = 0.01;
f64 targetFps = 82.5;
#define TEXTINPUTBUFFERSIZE 1048576
#define TEXTBUFFERSIZE 1048576
char* textInputBuffer = NULL;
u64 textInputBufferPos = 0;
char* textBuffer = NULL;
u64 textBufferPos = 0;
u32 fullscreen = 0; // 0 no, 1, maybe, 2, yes

u32 EVENT_PASTE = 0;  // Will be initialized in main

// mouse speed
#ifdef __EMSCRIPTEN__
float getMouseSpeed(){ return 1.0; }
#else
float getMouseSpeed() {
  int speed;
  if( !fullscreen )
    return 1.0;
  SystemParametersInfo( SPI_GETMOUSESPEED, 0, &speed, 0 );
  return (float)( speed ) / 10.0;
}
#endif

// Delay to framerat
void delay( void ){
  f64 ttime = 1000 / targetFps - rawFrameTime;
  if( ttime > 0 )
    SDL_Delay( ttime );  
}


#ifdef __EMSCRIPTEN__


// 1. C Callback: JS calls this when the promise resolves
EMSCRIPTEN_KEEPALIVE
void on_paste_received( char* text ){
  if( !text || !EVENT_PASTE )
    return;

  SDL_Event event;
  SDL_zero( event );
  event.type = EVENT_PASTE;
  // We must copy the string because Emscripten will free 'text' immediately
  // after this returns
  event.user.data1 = SDL_strdup( text );
  SDL_PushEvent( &event );
}

// 2. JS Event Listener (REPLACES trigger_web_paste)
EM_JS( void, setup_browser_paste_listener, (), {
    // Attach to the window so we catch paste events anywhere in the tab
    window.addEventListener(
                            'paste', function( e ){
                              // 1. Get the data
                              var pasteText =
                                ( e.clipboardData || window.clipboardData ).getData( 'text' );
                              if( !pasteText )
                                return;

                              // 2. Prevent the browser from double-handling it (optional, but safer)
                              e.preventDefault();

                              // 3. Send to C
                              var lengthBytes = lengthBytesUTF8( pasteText ) + 1;
                              var stringOnWasmHeap = _malloc( lengthBytes );
                              stringToUTF8( pasteText, stringOnWasmHeap, lengthBytes );

                              Module.ccall( 'on_paste_received', null, ['number'], [stringOnWasmHeap] );

                              _free( stringOnWasmHeap );
                            } );

    console.log( "Paste listener attached." );
  } );
#endif

// Structure to pass argc/argv to the render thread
typedef struct {
  int argc;
  char** argv;
} LaunchArgs;

void loadProg( program** prog, tensorStack** ts, const char* fileName ){
  const char* realName = fileName ? fileName : "main.atl";
  if( !fileExists( realName ) )
    error( "File %s does not exist. Either provide a filename argument, or "
           "provide a main.atl",
           realName );
  char* err = newProgramFromFile( realName, prog );
  if( err )
    error( "%s", err );
  *ts = newStack();
}

// Here we define the touch interface.
#ifdef __EMSCRIPTEN__
#include <emscripten/html5.h>

EMSCRIPTEN_KEEPALIVE
EM_BOOL onTouch( int eventType,
                 const EmscriptenTouchEvent* touchEvent,
                 void* userData ){
  static float oldPinchZoom = 0.0;
  static float newPinchZoom = 0.0;
  if( eventType == EMSCRIPTEN_EVENT_TOUCHEND ||
      eventType == EMSCRIPTEN_EVENT_TOUCHCANCEL ){
    for( int i = 0; i < 3; ++i ){
      touchClicks[ i ] = 0;
    }
    oldPinchZoom = 0.0;
  } else {
    if( touchEvent->numTouches == 1 ){
      touchClicks[ 0 ] = 1;
      oldPinchZoom = 0.0;
    } else
      touchClicks[ 0 ] = 0;
    if( touchEvent->numTouches == 2 ){
      float x1 = touchEvent->touches[ 0 ].clientX;
      float y1 = touchEvent->touches[ 0 ].clientY;
      float x2 = touchEvent->touches[ 1 ].clientX;
      float y2 = touchEvent->touches[ 1 ].clientY;
      newPinchZoom =
        sqrtf( ( x2 - x1 ) * ( x2 - x1 ) + ( y2 - y1 ) * ( y2 - y1 ) );
      if( oldPinchZoom != 0.0 ){
        pinchZoom += ( newPinchZoom - oldPinchZoom ) / 20.0;
      }
      oldPinchZoom = newPinchZoom;
      touchClicks[ 1 ] = 1;
    } else
      touchClicks[ 1 ] = 0;
    if( touchEvent->numTouches == 3 ){
      touchClicks[ 2 ] = 1;
      oldPinchZoom = 0.0;
    } else
      touchClicks[ 2 ] = 0;
  }
  return EM_TRUE;
}

#endif  // touch interface

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
void resizeWindow( int width, int height ){
  // Optionally, you can print a message to verify this function is called:
  // printf( "Resizing SDL window to %d x %d\n", width, height );
  jsWidth = width;
  jsHeight = height;
  // Update the SDL window's size.
  SDL_SetWindowSize( window, width, height );
}
#endif

#ifndef __EMSCRIPTEN__
void APIENTRY openglDebugCallback( GLenum source,
                                   GLenum type,
                                   GLuint id,
                                   GLenum severity,
                                   GLsizei length,
                                   const GLchar* message,
                                   const void* userParam ){
  dbg( "OpenGL Debug Message:%s", "\n" );
  dbg( "Source: %d, Type: %d, ID: %d, Severity: %d\n",
       source,
       type,
       id,
       severity );
  dbg( "Message: %s\n", message );
}

void enableDebugCallback(){
  glEnable( GL_DEBUG_OUTPUT );
  glEnable( GL_DEBUG_OUTPUT_SYNCHRONOUS );
  glDebugMessageCallback( openglDebugCallback, NULL );
}

// Thread synchronization variables
SDL_atomic_t running;
// SDL_mutex* data_mutex = NULL;
void SetDarkTitleBar( SDL_Window* sdlWindow ){
  SDL_SysWMinfo wmInfo;
  SDL_VERSION( &wmInfo.version );

  if( SDL_GetWindowWMInfo( sdlWindow, &wmInfo ) ){
    HWND hwnd = wmInfo.info.win.window;
    BOOL enable = TRUE;

    // Apply dark mode attribute
    HRESULT hr = DwmSetWindowAttribute( hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, &enable, sizeof( enable ) );
    if( SUCCEEDED( hr ) ){
      ShowWindow( hwnd, SW_MINIMIZE );
      ShowWindow( hwnd, SW_RESTORE );
    } else {
      MessageBoxA( NULL, "Failed to set dark title bar!", "Error", MB_ICONERROR );
    }

  } else {
    SDL_Log( "Unable to get window handle: %s", SDL_GetError() );
  }
}
#else
int running;  // Simple integer for the running flag in single-threaded mode
#endif

void mainPoll( void ){
  SDL_Event event;
  // SDL_Delay( 1 ); // Without this delay, the render to desktop code can
  // deadlock for unknown reason.
  while( SDL_PollEvent( &event ) ){

    ////////////////////////////////////////////////////
    // Fullscreen check
#ifdef __EMSCRIPTEN__
    EmscriptenFullscreenChangeEvent fsce;
    emscripten_get_fullscreen_status( &fsce );
    if( !fsce.isFullscreen && fullscreen == 2 ){
      SDL_ShowCursor( SDL_ENABLE );
      SDL_SetRelativeMouseMode( SDL_FALSE );
      fullscreen = 0;
    }
    if( fsce.isFullscreen && fullscreen == 1 )
      fullscreen = 2;
    if( !fsce.isFullscreen && fullscreen == 1 ){
      SDL_ShowCursor( SDL_DISABLE );
      SDL_SetRelativeMouseMode( SDL_TRUE );
      emscripten_request_fullscreen( "#canvas", true );
    }
    if( fsce.isFullscreen && fullscreen == 0 ){
      SDL_ShowCursor( SDL_ENABLE );
      SDL_SetRelativeMouseMode( SDL_FALSE );
      emscripten_exit_fullscreen();
    }
#else
    Uint32 flags = SDL_GetWindowFlags( window );
    if( !( flags & SDL_WINDOW_FULLSCREEN_DESKTOP ) && fullscreen == 2 )
      fullscreen = 0;
    if( ( flags & SDL_WINDOW_FULLSCREEN_DESKTOP ) && fullscreen == 1 )
      fullscreen = 2;
    if( !( flags & SDL_WINDOW_FULLSCREEN_DESKTOP ) && fullscreen ){
      SDL_ShowCursor( SDL_DISABLE );
      SDL_SetRelativeMouseMode( SDL_TRUE );
      SDL_SetWindowFullscreen( window, SDL_WINDOW_FULLSCREEN_DESKTOP );
    }
    if( ( flags & SDL_WINDOW_FULLSCREEN_DESKTOP ) && !fullscreen ){
      SDL_ShowCursor( SDL_ENABLE );
      SDL_SetRelativeMouseMode( SDL_FALSE );
      SDL_SetWindowFullscreen( window, 0 );
    }
#endif
    if( event.type == SDL_QUIT ||
        ( event.type == SDL_WINDOWEVENT &&
          event.window.event == SDL_WINDOWEVENT_CLOSE ) ){
#ifndef __EMSCRIPTEN__
      SDL_AtomicSet( &running, 0 );
#else
      emscripten_cancel_main_loop();
#endif
    } else if( event.type == SDL_MOUSEWHEEL ){
#ifndef __EMSCRIPTEN__
      // SDL_LockMutex( data_mutex );
#endif
      mouseWheel += event.wheel.y;
#ifndef __EMSCRIPTEN__
      // SDL_UnlockMutex( data_mutex );
#endif
    } else if( event.type == SDL_MOUSEMOTION ){
#ifndef __EMSCRIPTEN__
      // SDL_LockMutex( data_mutex );
#endif
      posx = event.motion.x;
      posy = event.motion.y;
      if( abs( event.motion.xrel ) < 1000 && abs( event.motion.yrel ) < 1000 ){
        dx += event.motion.xrel * getMouseSpeed();
        dy += event.motion.yrel * getMouseSpeed();
      }
#ifndef __EMSCRIPTEN__
      // SDL_UnlockMutex( data_mutex );
#endif
    } else if( event.type == SDL_TEXTINPUT ){
#ifndef __EMSCRIPTEN__
      // SDL_LockMutex( data_mutex );
#endif
      s64 remaining = TEXTINPUTBUFFERSIZE - textInputBufferPos - 1;
      if( remaining < 0 )
        remaining = 0;
      strncpy(
              textInputBuffer + textInputBufferPos, event.text.text, remaining );
      textInputBufferPos += strlen( event.text.text );
      textInputBuffer[ TEXTINPUTBUFFERSIZE - 1 ] = 0;

#ifndef __EMSCRIPTEN__
      // SDL_UnlockMutex( data_mutex );
#endif
    } else if( event.type == EVENT_PASTE ){
#ifndef __EMSCRIPTEN__
      // SDL_LockMutex( data_mutex );
#endif

      char* pastedText = (char*)event.user.data1;
      if( pastedText ){
        s64 available = TEXTINPUTBUFFERSIZE - textInputBufferPos - 1;
        if( available > 0 ){
          strncpy(
                  textInputBuffer + textInputBufferPos, pastedText, available );
        }
        textInputBufferPos += strlen( pastedText );
        textInputBuffer[ TEXTINPUTBUFFERSIZE - 1 ] = 0;
        SDL_free( pastedText );  // Clean up the duplicate we made
      }
      // A kludge to deal with firefox
#ifdef __EMSCRIPTEN__
      SDL_Event ev;
      SDL_zero( ev );
      ev.type = SDL_KEYUP;
      ev.key.state = SDL_RELEASED;
      ev.key.repeat = 0;
      ev.key.keysym.mod = KMOD_NONE;
      ev.key.keysym.sym = SDLK_LCTRL;
      ev.key.keysym.scancode = SDL_SCANCODE_LCTRL;
      SDL_PushEvent( &ev );
      ev.key.keysym.sym = SDLK_RCTRL;
      ev.key.keysym.scancode = SDL_SCANCODE_RCTRL;
      SDL_PushEvent( &ev );
      ev.key.keysym.sym = SDLK_LSHIFT;
      ev.key.keysym.scancode = SDL_SCANCODE_LSHIFT;
      SDL_PushEvent( &ev );
      ev.key.keysym.sym = SDLK_RSHIFT;
      ev.key.keysym.scancode = SDL_SCANCODE_RSHIFT;
      SDL_PushEvent( &ev );
      ev.key.keysym.sym = SDLK_v;
      ev.key.keysym.scancode = SDL_SCANCODE_V;
      SDL_PushEvent( &ev );
      ev.key.keysym.sym = SDLK_INSERT;
      ev.key.keysym.scancode = SDL_SCANCODE_INSERT;
      SDL_PushEvent( &ev );
#endif
#ifndef __EMSCRIPTEN__
      // SDL_UnlockMutex( data_mutex );
#endif
    } else if( event.type == SDL_MOUSEBUTTONDOWN ){
#ifndef __EMSCRIPTEN__
      // SDL_LockMutex( data_mutex );
#endif
      if( event.button.clicks == 2 ){
        if( event.button.button == SDL_BUTTON_LEFT )
          doubleClicks[ 0 ] = 1;
        if( event.button.button == SDL_BUTTON_RIGHT )
          doubleClicks[ 1 ] = 1;
        if( event.button.button == SDL_BUTTON_MIDDLE )
          doubleClicks[ 2 ] = 1;
      }
      if( event.button.button == SDL_BUTTON_LEFT )
        buttons |= SDL_BUTTON( SDL_BUTTON_LEFT );
      if( event.button.button == SDL_BUTTON_RIGHT )
        buttons |= SDL_BUTTON( SDL_BUTTON_RIGHT );
      if( event.button.button == SDL_BUTTON_MIDDLE )
        buttons |= SDL_BUTTON( SDL_BUTTON_MIDDLE );
#ifndef __EMSCRIPTEN__
      // SDL_UnlockMutex( data_mutex );
#endif
    } else if( event.type == SDL_MOUSEBUTTONUP ){
#ifndef __EMSCRIPTEN__
      // SDL_LockMutex( data_mutex );
#endif
      if( event.button.button == SDL_BUTTON_LEFT )
        buttons &= ~SDL_BUTTON( SDL_BUTTON_LEFT );
      if( event.button.button == SDL_BUTTON_RIGHT )
        buttons &= ~SDL_BUTTON( SDL_BUTTON_RIGHT );
      if( event.button.button == SDL_BUTTON_MIDDLE )
        buttons &= ~SDL_BUTTON( SDL_BUTTON_MIDDLE );
#ifndef __EMSCRIPTEN__
      // SDL_UnlockMutex( data_mutex );
#endif
    } else if( event.type == SDL_KEYDOWN ){
#ifndef __EMSCRIPTEN__
      // SDL_LockMutex( data_mutex );
#endif

      // Hard coded copy paste
      if( ( event.key.keysym.sym == SDLK_v &&
            ( SDL_GetModState() & KMOD_CTRL ) ) ||
          ( event.key.keysym.sym == SDLK_INSERT &&
            ( SDL_GetModState() & KMOD_SHIFT ) ) ){
        if( SDL_HasClipboardText() ){
          char* clipText = SDL_GetClipboardText();
          if( clipText ){
            u32 currentLen = strlen( textInputBuffer );
            u32 available = TEXTINPUTBUFFERSIZE - currentLen - 1;
            if( available > 0 )
              strncat( textInputBuffer, clipText, available );
            SDL_free( clipText );
          }
        }
      }
      keys[ event.key.keysym.scancode ] = 1;
#ifndef __EMSCRIPTEN__
      // SDL_UnlockMutex( data_mutex );
#endif
    } else if( event.type == SDL_KEYUP ){
#ifndef __EMSCRIPTEN__
      // SDL_LockMutex( data_mutex );
#endif
      keys[ event.key.keysym.scancode ] = 0;
#ifndef __EMSCRIPTEN__
      // SDL_UnlockMutex( data_mutex );
#endif
    } else if( event.type == SDL_CONTROLLERDEVICEADDED ){
#ifndef __EMSCRIPTEN__
      //      SDL_LockMutex( data_mutex );
#endif
      // First check if already attached.
      bool found = false;
      for( int i = 0; i < MAX_CONTROLLERS; ++i ){
        if( controllers[ i ] && joystickIDs[ i ] == event.cdevice.which ){
          found = true;
        }
      }
      // Find the first available slot
      if( !found ){
        for( int i = 0; i < MAX_CONTROLLERS; ++i ){
          if( controllers[ i ] == NULL ){
            if( SDL_IsGameController( event.cdevice.which ) ){
              controllers[ i ] = SDL_GameControllerOpen( event.cdevice.which );
              if( controllers[ i ] ){
                joystickIDs[ i ] = SDL_JoystickInstanceID(
                                                          SDL_GameControllerGetJoystick( controllers[ i ] ) );
              }
            }
            break;  // Stop after adding to one slot
          }
        }
      }
#ifndef __EMSCRIPTEN__
      //      SDL_UnlockMutex( data_mutex );
#endif

    } else if( event.type == SDL_CONTROLLERDEVICEREMOVED ){
#ifndef __EMSCRIPTEN__
      //      SDL_LockMutex( data_mutex );
#endif

      // Identify which controller was removed
      for( int i = 0; i < MAX_CONTROLLERS; ++i ){
        if( controllers[ i ] && joystickIDs[ i ] == event.cdevice.which ){
          SDL_GameControllerClose( controllers[ i ] );
          controllers[ i ] = NULL;
          joystickIDs[ i ] = -1;
        }
      }
#ifndef __EMSCRIPTEN__
      //      SDL_UnlockMutex( data_mutex );
#endif
    } else if( event.type == SDL_CONTROLLERAXISMOTION ){
#ifndef __EMSCRIPTEN__
      // SDL_LockMutex( data_mutex );
#endif
      // Handle axis motion
      for( u32 i = 0; i < MAX_CONTROLLERS; ++i ){
        if( controllers[ i ] && joystickIDs[ i ] == event.caxis.which ){
          f32 axisValue = ( (f32)( event.caxis.value ) / 32767.0 );
          switch( event.caxis.axis ){
          case SDL_CONTROLLER_AXIS_TRIGGERLEFT:
            joysticks[ i * 21 + 0 ] = axisValue;
            break;
          case SDL_CONTROLLER_AXIS_TRIGGERRIGHT:
            joysticks[ i * 21 + 1 ] = axisValue;
            break;
          case SDL_CONTROLLER_AXIS_LEFTX:
            joysticks[ i * 21 + 2 ] = axisValue;
            break;
          case SDL_CONTROLLER_AXIS_LEFTY:
            joysticks[ i * 21 + 3 ] = axisValue;
            break;
          case SDL_CONTROLLER_AXIS_RIGHTX:
            joysticks[ i * 21 + 4 ] = axisValue;
            break;
          case SDL_CONTROLLER_AXIS_RIGHTY:
            joysticks[ i * 21 + 5 ] = axisValue;
            break;
          }
        }
      }
#ifndef __EMSCRIPTEN__
      //      SDL_UnlockMutex( data_mutex );
#endif
    } else if( event.type == SDL_CONTROLLERBUTTONDOWN ||
               event.type == SDL_CONTROLLERBUTTONUP ){
#ifndef __EMSCRIPTEN__
      //      SDL_LockMutex( data_mutex );
#endif
      f32 upordown = 0.0;
      if( event.type == SDL_CONTROLLERBUTTONDOWN )
        upordown = 1.0;
      // Handle axis motion
      for( u32 i = 0; i < MAX_CONTROLLERS; ++i ){
        if( controllers[ i ] && joystickIDs[ i ] == event.cbutton.which ){
          switch( event.cbutton.button ){
          case SDL_CONTROLLER_BUTTON_LEFTSHOULDER:
            joysticks[ i * 21 + 6 ] = upordown;
            break;
          case SDL_CONTROLLER_BUTTON_RIGHTSHOULDER:
            joysticks[ i * 21 + 7 ] = upordown;
            break;
          case SDL_CONTROLLER_BUTTON_GUIDE:
            joysticks[ i * 21 + 8 ] = upordown;
            break;
          case SDL_CONTROLLER_BUTTON_DPAD_UP:
            joysticks[ i * 21 + 9 ] = upordown;
            break;
          case SDL_CONTROLLER_BUTTON_DPAD_RIGHT:
            joysticks[ i * 21 + 10 ] = upordown;
            break;
          case SDL_CONTROLLER_BUTTON_DPAD_DOWN:
            joysticks[ i * 21 + 11 ] = upordown;
            break;
          case SDL_CONTROLLER_BUTTON_DPAD_LEFT:
            joysticks[ i * 21 + 12 ] = upordown;
            break;
          case SDL_CONTROLLER_BUTTON_BACK:
            joysticks[ i * 21 + 13 ] = upordown;
            break;
          case SDL_CONTROLLER_BUTTON_START:
            joysticks[ i * 21 + 14 ] = upordown;
            break;
          case SDL_CONTROLLER_BUTTON_A:
            joysticks[ i * 21 + 15 ] = upordown;
            break;
          case SDL_CONTROLLER_BUTTON_B:
            joysticks[ i * 21 + 16 ] = upordown;
            break;
          case SDL_CONTROLLER_BUTTON_X:
            joysticks[ i * 21 + 17 ] = upordown;
            break;
          case SDL_CONTROLLER_BUTTON_Y:
            joysticks[ i * 21 + 18 ] = upordown;
            break;
          case SDL_CONTROLLER_BUTTON_LEFTSTICK:
            joysticks[ i * 21 + 19 ] = upordown;
            break;
          case SDL_CONTROLLER_BUTTON_RIGHTSTICK:
            joysticks[ i * 21 + 20 ] = upordown;
            break;
          }
        }
      }
#ifndef __EMSCRIPTEN__
      //      SDL_UnlockMutex( data_mutex );
#endif
    }
  }
}

// Vertex Shader Source
const GLchar* vertexSource = "#version 300 es\n"
  "precision highp float;\n"
  "precision highp int;\n"
  "precision highp sampler2DArray;\n"
  "in vec2 position;\n"
  "out vec2 fragCoord;\n"
  "void main(){\n"
  "  fragCoord = position;\n"
  "  gl_Position = vec4( position, 0.0, 1.0 );\n"
  "}\n";


const GLchar* fragmentSource =
  "#version 300 es\n"
  "precision highp float;\n"
  "precision highp int;\n"
  "precision highp sampler2DArray;\n"
  "in vec2 fragCoord;\n"
  "out vec4 fragColor;\n"
  "uniform sampler2DArray tex;\n"
  "uniform vec2 resolution;\n"
  "uniform vec4 strides;\n"
  "uniform vec4 shape;\n"
  "uniform vec2 dims;\n"
  "uniform float toffset;\n"
  "float sampleTensorIndex(vec4 i){\n"
  "  float lindex = dot( floor( i ), strides ) + toffset + 0.25;\n"
  "  float pixel_index = floor( lindex / 4.0 ) + 0.25;\n"
  "  float channel = mod( lindex, 4.0 );\n"
  "  vec2 uv = ( vec2( mod( pixel_index, dims.x ), floor( pixel_index / dims.x "
  ") ) + 0.25 ) / dims;\n"
  "  vec4 texel = textureLod( tex, vec3( uv, 0 ), 0.0 );\n"
  "  return texel[ int( channel ) ];\n"
  "}\n"
  "void main(){\n"
  "  vec4 linear = textureLod( tex, vec3( gl_FragCoord.xy / resolution, 0.0 ), 0.0 );\n"
  "  vec3 gamma = pow( linear.rgb, vec3( 1.0/1.6 ) );\n"

  // 4x4 Bayer matrix threshold
  "  ivec2 p = ivec2( gl_FragCoord.xy ) & 3;\n"  // mod 4
  "  int idx = p.x + p.y * 4;\n"
  "  const float bayer[16] = float[16](\n"
  "     0.0/16.0,  8.0/16.0,  2.0/16.0, 10.0/16.0,\n"
  "    12.0/16.0,  4.0/16.0, 14.0/16.0,  6.0/16.0,\n"
  "     3.0/16.0, 11.0/16.0,  1.0/16.0,  9.0/16.0,\n"
  "    15.0/16.0,  7.0/16.0, 13.0/16.0,  5.0/16.0\n"
  "  );\n"
  "  float threshold = bayer[idx] - 0.5;\n"  // center around 0

  // Add dither before quantizing
  "  float levels = 256.0;\n"
  "  vec3 dithered = gamma + threshold / levels;\n"
  "  fragColor = vec4( floor( dithered * levels ) / levels, 1.0 );\n"
  "}\n";


// Function to compile shaders
GLuint compileShader( GLenum type, const GLchar* source ){
  GLuint shader = glCreateShader( type );
  glShaderSource( shader, 1, &source, NULL );
  glCompileShader( shader );

  // Check for compilation errors
  GLint status;
  glGetShaderiv( shader, GL_COMPILE_STATUS, &status );
  if( status != GL_TRUE ){
    char buffer[ 1024 ];
    glGetShaderInfoLog( shader, sizeof( buffer ), NULL, buffer );
    error( "Shader compilation failed: %s\n", buffer );
  }
  return shader;
}

// Function to create shader program
GLuint createProgram( const GLchar* vertexSource,
                      const GLchar* fragmentSource ){
  GLuint vertexShader = compileShader( GL_VERTEX_SHADER, vertexSource );
  GLuint fragmentShader = compileShader( GL_FRAGMENT_SHADER, fragmentSource );

  GLuint program = glCreateProgram();
  glAttachShader( program, vertexShader );
  glAttachShader( program, fragmentShader );
  glBindAttribLocation( program, 0, "position" );
  glLinkProgram( program );

  // Check for linking errors
  GLint status;
  glGetProgramiv( program, GL_LINK_STATUS, &status );
  if( status != GL_TRUE ){
    char buffer[ 1024 ];
    glGetProgramInfoLog( program, 512, NULL, buffer );
    printf( "Program linking failed: %s\n", buffer );
  }

  glDeleteShader( vertexShader );
  glDeleteShader( fragmentShader );

  return program;
}

#ifndef __EMSCRIPTEN__
// Rendering and computation thread function
int renderThreadFunction( void* data ){
  LaunchArgs* args = (LaunchArgs*)data;

  // Determine filename: If we have >1 arg, the second one is the script
  // argv[0] is usually the executable name
  const char* fileName = ( args->argc >= 2 ) ? args->argv[ 1 ] : NULL;

  // Create the OpenGL context in the render thread
  glContext = SDL_GL_CreateContext( window );
  if( !glContext )
    error( "SDL_GL_CreateContext Error: %s\n", SDL_GetError() );
  // Initialize GLEW
  glewExperimental = GL_TRUE;  // Enable modern OpenGL techniques
  GLenum glewError = glewInit();
  if( glewError != GLEW_OK ){
    printf( "glewInit Error: %s\n", glewGetErrorString( glewError ) );
    SDL_AtomicSet( &running, 0 );
    return 1;
  }
  glGenVertexArrays( 1, &vao );
  glBindVertexArray( vao );

  // enableDebugCallback();
  //  Initialize OpenGL
  int windowWidth, windowHeight;
  SDL_GetWindowSize( window, &windowWidth, &windowHeight );
  glViewport( 0, 0, windowWidth, windowHeight );
  glClearColor( 0.0f, 0.0f, 0.0f, 1.0f );
  SDL_GL_SetSwapInterval( 1 );  // VRR
  SDL_SetHint( SDL_HINT_RENDER_VSYNC, "1" );
  /* SDL_GL_SetSwapInterval( 0 ); */
  /* SDL_SetHint(SDL_HINT_RENDER_VSYNC, "0"); */

  shaderProgram = createProgram( vertexSource, fragmentSource );

  // Set up vertex data
  GLfloat vertices[] = {
    -1.0f,
    -1.0f,  // Bottom-left
    1.0f,
    -1.0f,  // Bottom-right
    -1.0f,
    1.0f,  // Top-left
    1.0f,
    1.0f,  // Top-right
  };

  // Generate VBO
  glGenBuffers( 1, &vbo );
  glBindBuffer( GL_ARRAY_BUFFER, vbo );
  glBufferData( GL_ARRAY_BUFFER, sizeof( vertices ), vertices, GL_STATIC_DRAW );

  // Compile program.
  loadProg( &prog, &ts, fileName );

  if( ts ){
    for( int i = 2; i < args->argc; i++ ){
      push( ts, tensorFromString( args->argv[ i ] ) );
    }
  }

  // Main loop
  while( SDL_AtomicGet( &running ) ){
    // SDL_PumpEvents();
    // mainPoll();
    //  Run the program
    CHECK_GL_ERROR();
    prevTime = curTime;
    curTime = SDL_GetPerformanceCounter();
    rawFrameTime = (f64)( curTime - prevTime ) / (f64)( SDL_GetPerformanceFrequency() );
    timeDelta *= 0.5; timeDelta += 0.5 * rawFrameTime;
      
    runTime =
      (f64)( curTime - startTime ) / (f64)( SDL_GetPerformanceFrequency() );
    bool ret;
    char* msg = runProgram( ts, &prog, 0, &ret );
    if( msg )
      error( "%s", msg );
    if( !ret ){

#ifdef __EMSCRIPTEN__
      // 1. Force the screen to show the X immediately
      emscripten_run_script( "Module.showCrash();" );

      // 2. Kill the loop (SimulateInfiniteLoop exception)
      emscripten_cancel_main_loop();
      return;
#else
      // Native quit
      SDL_AtomicSet( &running, 0 );
#endif
#ifdef DBG
      check_memory_leaks();
#endif
      break;
    }

#ifndef EMSCRIPTEN // only need this for native afaict
    //glFinish();
    //glFlush();
#endif

    // Get current window size
    SDL_GetWindowSize( window, &windowWidth, &windowHeight );

    // Adjust the viewport
    glViewport( 0, 0, windowWidth, windowHeight );

    // Render
    if( !ts->size )
      continue;
    if( !ts->stack[ ts->size - 1 ]->gpu )
      tensorToGPUMemory( ts->stack[ ts->size - 1 ] );
    if( ts->stack[ ts->size - 1 ]->rank != 3 )
      error( "%s", "Display tensor not of rank 3" );
    if( ts->stack[ ts->size - 1 ]->shape[ 2 ] != 4 )
      error( "%s", "Display tensor not a 4 component tensor of rank 3." );
    if( ts->stack[ ts->size - 1 ]->tex.channels != 400 )
      error( "%s", "Display tensor not a 4 channel half float tensor of rank 3." );


    glClear( GL_COLOR_BUFFER_BIT );

    glUseProgram( shaderProgram );
    glViewport( 0, 0, windowWidth, windowHeight );

    GLint texLoc = glGetUniformLocation( shaderProgram, "tex" );
    glUniform1i( texLoc, 0 );  // Texture unit 0

    // Bind the texture to texture unit 0
    glActiveTexture( GL_TEXTURE0 );
    glBindTexture( GL_TEXTURE_2D_ARRAY,
                   ts->stack[ ts->size - 1 ]->tex.texture );

    // Set uniforms
    GLint dimsLoc = glGetUniformLocation( shaderProgram, "dims" );
    glUniform2f( dimsLoc,
                 ts->stack[ ts->size - 1 ]->tex.width,
                 ts->stack[ ts->size - 1 ]->tex.height );

    GLint resolutionLoc = glGetUniformLocation( shaderProgram, "resolution" );
    glUniform2f( resolutionLoc, (float)windowWidth, (float)windowHeight );

    GLint stridesLoc = glGetUniformLocation( shaderProgram, "strides" );
    glUniform4f( stridesLoc,
                 ts->stack[ ts->size - 1 ]->strides[ 0 ],
                 ts->stack[ ts->size - 1 ]->strides[ 1 ],
                 ts->stack[ ts->size - 1 ]->strides[ 2 ],
                 ts->stack[ ts->size - 1 ]->strides[ 3 ] );
    GLint shapeLoc = glGetUniformLocation( shaderProgram, "shape" );
    glUniform4f( shapeLoc,
                 ts->stack[ ts->size - 1 ]->shape[ 0 ],
                 ts->stack[ ts->size - 1 ]->shape[ 1 ],
                 ts->stack[ ts->size - 1 ]->shape[ 2 ],
                 ts->stack[ ts->size - 1 ]->shape[ 3 ] );

    GLint toffsetLoc = glGetUniformLocation( shaderProgram, "toffset" );
    glUniform1f( toffsetLoc, ts->stack[ ts->size - 1 ]->offset );

    // Bind VBO and set vertex attributes
    glBindBuffer( GL_ARRAY_BUFFER, vbo );
    GLint posAttrib = glGetAttribLocation( shaderProgram, "position" );
    glEnableVertexAttribArray( posAttrib );
    glVertexAttribPointer( posAttrib, 2, GL_FLOAT, GL_FALSE, 0, 0 );

    // Draw
    glBindFramebuffer( GL_FRAMEBUFFER, 0 );
    glDrawArrays( GL_TRIANGLE_STRIP, 0, 4 );

    // Cleanup
    glDisableVertexAttribArray( posAttrib );
    glBindTexture( GL_TEXTURE_2D_ARRAY, 0 );

#ifndef EMSCRIPTEN // only need this for native afaict
    //glFinish();
    //glFlush();
#endif
    
    delay();
    SDL_GL_SwapWindow( window );

    // DwmFlush();
  }

  // Cleanup
  glDeleteProgram( shaderProgram );
  glDeleteBuffers( 1, &vbo );

  deleteProgram( prog );
  deleteStack( ts );

  glDeleteVertexArrays( 1, &vao );
  vao = 0;
  SDL_GL_DeleteContext( glContext );

  return 0;
}
#else

// Main loop function for Emscripten
void main_loop( void ){
  // Process events
  SDL_Event event;
  mainPoll();

  // Run the program
  CHECK_GL_ERROR();
  prevTime = curTime;
  curTime = SDL_GetPerformanceCounter();
  timeDelta *= 0.9;
  timeDelta +=
    0.1 * (f64)( curTime - prevTime ) / (f64)( SDL_GetPerformanceFrequency() );
  runTime =
    (f64)( curTime - startTime ) / (f64)( SDL_GetPerformanceFrequency() );
  bool ret;
  char* err = runProgram( ts, &prog, 0, &ret );
  if( err )
    error( "%s", err );
  if( !ret ){

#ifdef __EMSCRIPTEN__
    // 1. Force the screen to show the X immediately
    emscripten_run_script( "Module.showCrash();" );

    // 2. Kill the loop (SimulateInfiniteLoop exception)
    emscripten_cancel_main_loop();
    return;
#else
    // Native quit
    SDL_AtomicSet( &running, 0 );
#endif

    running = 0;
    emscripten_cancel_main_loop();
    return;
  }

  // Rendering code
  // Get current window size
  int windowWidth, windowHeight;
  SDL_GetWindowSize( window, &windowWidth, &windowHeight );

  // Adjust the viewport
  glViewport( 0, 0, windowWidth, windowHeight );

  // Render
  if( !ts->size )
    return;
  if( !ts->stack[ ts->size - 1 ]->gpu )
    tensorToGPUMemory( ts->stack[ ts->size - 1 ] );
  if( ts->stack[ ts->size - 1 ]->rank != 3 )
    error( "%s", "Display tensor not of rank 3" );
  if( ts->stack[ ts->size - 1 ]->shape[ 2 ] != 4 )
    error( "%s", "Display tensor not a 4 component tensor of rank 3." );
  if( ts->stack[ ts->size - 1 ]->tex.channels != 400 )
    error( "%s", "Display tensor not a 4 channel half float tensor of rank 3." );

  glClear( GL_COLOR_BUFFER_BIT );

  glUseProgram( shaderProgram );
  glViewport( 0, 0, windowWidth, windowHeight );

  GLint texLoc = glGetUniformLocation( shaderProgram, "tex" );
  glUniform1i( texLoc, 0 );  // Texture unit 0

  // Bind the texture to texture unit 0
  glActiveTexture( GL_TEXTURE0 );
  glBindTexture( GL_TEXTURE_2D_ARRAY, ts->stack[ ts->size - 1 ]->tex.texture );

  // Set uniforms
  GLint dimsLoc = glGetUniformLocation( shaderProgram, "dims" );
  glUniform2f( dimsLoc,
               ts->stack[ ts->size - 1 ]->tex.width,
               ts->stack[ ts->size - 1 ]->tex.height );

  GLint resolutionLoc = glGetUniformLocation( shaderProgram, "resolution" );
  glUniform2f( resolutionLoc, (float)windowWidth, (float)windowHeight );

  GLint stridesLoc = glGetUniformLocation( shaderProgram, "strides" );
  glUniform4f( stridesLoc,
               ts->stack[ ts->size - 1 ]->strides[ 0 ],
               ts->stack[ ts->size - 1 ]->strides[ 1 ],
               ts->stack[ ts->size - 1 ]->strides[ 2 ],
               ts->stack[ ts->size - 1 ]->strides[ 3 ] );
  GLint shapeLoc = glGetUniformLocation( shaderProgram, "shape" );
  glUniform4f( shapeLoc,
               ts->stack[ ts->size - 1 ]->shape[ 0 ],
               ts->stack[ ts->size - 1 ]->shape[ 1 ],
               ts->stack[ ts->size - 1 ]->shape[ 2 ],
               ts->stack[ ts->size - 1 ]->shape[ 3 ] );

  GLint toffsetLoc = glGetUniformLocation( shaderProgram, "toffset" );

  glUniform1f( toffsetLoc, ts->stack[ ts->size - 1 ]->offset );

  // Bind VBO and set vertex attributes
  glBindBuffer( GL_ARRAY_BUFFER, vbo );
  GLint posAttrib = glGetAttribLocation( shaderProgram, "position" );
  glEnableVertexAttribArray( posAttrib );
  glVertexAttribPointer( posAttrib, 2, GL_FLOAT, GL_FALSE, 0, 0 );

  // Draw
  glBindFramebuffer( GL_FRAMEBUFFER, 0 );
  glDrawArrays( GL_TRIANGLE_STRIP, 0, 4 );

  // Cleanup
  glDisableVertexAttribArray( posAttrib );
  glBindTexture( GL_TEXTURE_2D_ARRAY, 0 );

  delay();
  SDL_GL_SwapWindow( window );
}
#endif

// Main function
#ifndef __EMSCRIPTEN__
int main( int argc, char* argv[] )
#else
  EMSCRIPTEN_KEEPALIVE
  void start( void )
#endif
{
  curTime = SDL_GetPerformanceCounter();
  startTime = curTime;
  prevTime = curTime;
  workspace = mem( 1, char );
  workspace[ 0 ] = '\0';
  textInputBuffer = mem( TEXTINPUTBUFFERSIZE, char );
  textBuffer = mem( TEXTBUFFERSIZE, char );
#ifndef __EMSCRIPTEN__

  // 1. Get the handle
  HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
  DWORD fileType = GetFileType(hOut);
    
  // Check if the handle is valid
  bool isValidHandle = (hOut != NULL && hOut != INVALID_HANDLE_VALUE && fileType != FILE_TYPE_UNKNOWN);

  // 2. If we have ANY valid handle (Pipe, File, or inherited Console), use it.
  // Do NOT assume only Pipes are valid. 
  if (isValidHandle) {
    // Bind the OS handle to the C Runtime's stdout (fd 1)
    int fd = _open_osfhandle((intptr_t)hOut, _O_TEXT);
    if (fd != -1) {
      _dup2(fd, 1);
      // DISABLE BUFFERING: This is the #1 cause of "missing" output in Emacs
      setvbuf(stdout, NULL, _IONBF, 0);
    }

    // Do the same for Stderr
    HANDLE hErr = GetStdHandle(STD_ERROR_HANDLE);
    if (hErr != NULL && hErr != INVALID_HANDLE_VALUE) {
      int fdErr = _open_osfhandle((intptr_t)hErr, _O_TEXT);
      if (fdErr != -1) {
        _dup2(fdErr, 2);
        setvbuf(stderr, NULL, _IONBF, 0);
      }
    }
  } 
  // 3. Only if we have NO handle (Detached GUI app), try to attach to parent
  else {
    if (AttachConsole(ATTACH_PARENT_PROCESS)) {
      // Now we are attached to a physical console (e.g., Bash window)
      // It is safe to reopen CONOUT$ here because we had no handle to begin with.
      freopen("CONOUT$", "w", stdout);
      freopen("CONOUT$", "w", stderr);
            
      // Still disable buffering just in case
      setvbuf(stdout, NULL, _IONBF, 0);
      setvbuf(stderr, NULL, _IONBF, 0);
    }
    // If AttachConsole fails, we are truly detached (double-click launch)
    // and have nowhere to print.
  }

  // Set UTF-8 to handle special chars correctly in Emacs
  SetConsoleOutputCP(CP_UTF8);

  SDL_AtomicSet( &running, 1 );

  
#else
  running = 1;
  emscripten_set_touchstart_callback( "#canvas", NULL, EM_TRUE, onTouch );
  emscripten_set_touchend_callback( "#canvas", NULL, EM_TRUE, onTouch );
  emscripten_set_touchmove_callback( "#canvas", NULL, EM_TRUE, onTouch );
  emscripten_set_touchcancel_callback( "#canvas", NULL, EM_TRUE, onTouch );
#endif

  // Set splash text
  printToBuffer( "%s", STARTTEXT );

  setvbuf( stdout, NULL, _IONBF, 0 );  // Unbuffer stdout
  setvbuf( stderr, NULL, _IONBF, 0 );  // Unbuffer stderr

  // Initialize SDL and create window in the main thread
  if( SDL_Init( SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER ) != 0 )
    error( "SDL_Init Error: %s\n", SDL_GetError() );
  EVENT_PASTE = SDL_RegisterEvents( 1 );
  SDL_GL_SetAttribute( SDL_GL_CONTEXT_MAJOR_VERSION, 2 );
  SDL_GL_SetAttribute( SDL_GL_CONTEXT_MINOR_VERSION, 1 );
  //  SDL_GL_SetAttribute( SDL_GL_CONTEXT_PROFILE_MASK,
  //  SDL_GL_CONTEXT_PROFILE_CORE );
  SDL_GL_SetAttribute( SDL_GL_DOUBLEBUFFER, 1 );

  // Add SDL_WINDOW_RESIZABLE flag
  window = SDL_CreateWindow( "Atlas",
                             SDL_WINDOWPOS_CENTERED,
                             SDL_WINDOWPOS_CENTERED,
                             1024,
                             768,  // Initial window size
                             SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN |
                             SDL_WINDOW_RESIZABLE );
  if( !window )
    error( "SDL_CreateWindow Error: %s\n", SDL_GetError() );
#ifndef __EMSCRIPTEN__
  SetDarkTitleBar( window );
#endif

#ifndef __EMSCRIPTEN__

  // Do not create the OpenGL context here for non-Emscripten builds
#else
  // Emscripten code (single-threaded), create the context here
  glContext = SDL_GL_CreateContext( window );
  if( !glContext )
    error( "SDL_GL_CreateContext Error: %s\n", SDL_GetError() );

  // Initialize OpenGL
  // int windowWidth, windowHeight;
  SDL_SetWindowSize( window, jsWidth, jsHeight );
  glViewport( 0, 0, jsWidth, jsHeight );
  //  glClearColor( 0.0f, 0.0f, 0.0f, 1.0f );

  shaderProgram = createProgram( vertexSource, fragmentSource );

  // Set up vertex data
  GLfloat vertices[] = {
    -1.0f,
    -1.0f,  // Bottom-left
    1.0f,
    -1.0f,  // Bottom-right
    -1.0f,
    1.0f,  // Top-left
    1.0f,
    1.0f,  // Top-right
  };

  // Generate VBO
  glGenBuffers( 1, &vbo );
  glBindBuffer( GL_ARRAY_BUFFER, vbo );
  glBufferData( GL_ARRAY_BUFFER, sizeof( vertices ), vertices, GL_STATIC_DRAW );

  // Compile program.
  loadProg( &prog, &ts, NULL );
#endif

#ifndef __EMSCRIPTEN__
  LaunchArgs args;
  args.argc = argc;
  args.argv = argv;

  SDL_Thread* renderThread =
    SDL_CreateThread( renderThreadFunction, "RenderThread", &args );
  if( renderThread == NULL ){
    error( "%s", "Failed to create rendering thread" );
  }

  // Main thread handles SDL event loop
  while( SDL_AtomicGet( &running ) ){
    mainPoll();
    SDL_Delay( 0 );  // Yield but don't actually sleep
  }
  // Cleanup controllers
  for( int i = 0; i < MAX_CONTROLLERS; ++i ){
    if( controllers[ i ] ){
      SDL_GameControllerClose( controllers[ i ] );
      controllers[ i ] = NULL;
      joystickIDs[ i ] = -1;
    }
  }

  // Wait for rendering thread to finish
  SDL_WaitThread( renderThread, NULL );

  // SDL_DestroyMutex( data_mutex );
#else
  // Set up the main loop for Emscripten
  emscripten_set_main_loop( main_loop, 0, 0 );
  return;
#endif

  // Cleanup
  unmem( workspace );
  unmem( textInputBuffer );
  unmem( textBuffer );
  glDeleteProgram( shaderProgram );
  glDeleteBuffers( 1, &vbo );
#ifdef __EMSCRIPTEN__
  deleteProgram( prog );
  deleteStack( ts );
  SDL_GL_DeleteContext( glContext );
#endif

  SDL_DestroyWindow( window );
  SDL_Quit();

#ifdef DEBUG
  check_memory_leaks();
#else
  if( memc )
    dbg( "mem count %llu", memc );
#endif

#ifndef __EMSCRIPTEN__
  return 0;
#endif
}

float getMaxAnisotropy( void ){
  static float ret = 0.0;
  if( ret )
    return ret;
  // Check if the anisotropic filtering extension is supported
  const char* extensions = (const char*)glGetString( GL_EXTENSIONS );
  if( extensions && strstr( extensions, "EXT_texture_filter_anisotropic" ) ){
    // Retrieve the extension
    GLfloat maxAnisotropy = 1.0f;
    glGetFloatv( GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAnisotropy );
    ret = maxAnisotropy;
  } else {
    ret = 1.0f;  // Default value (no anisotropic filtering)
  }
  return ret;
}
void vPrintToBuffer( const char* format, va_list args ){
  va_list len_args;
  va_copy( len_args, args );
  int needed = vsnprintf( NULL, 0, format, len_args );
  va_end( len_args );

  if( needed < 0 ) return;

  int available = TEXTBUFFERSIZE - textBufferPos - 1;
  if( needed > available ){
    size_t discard_amount = textBufferPos / 2;
    if( needed > (int)( TEXTBUFFERSIZE - ( textBufferPos - discard_amount ) ) ){
      size_t keep = ( TEXTBUFFERSIZE - 1 ) - needed;
      if( keep > textBufferPos ) keep = 0;
      discard_amount = textBufferPos - keep;
    }
    memmove( textBuffer, textBuffer + discard_amount, textBufferPos - discard_amount );
    textBufferPos -= discard_amount;
  }
  vsnprintf( textBuffer + textBufferPos, TEXTBUFFERSIZE - textBufferPos, format, args );
  
  textBufferPos += needed;
  if( textBufferPos >= TEXTBUFFERSIZE ) textBufferPos = TEXTBUFFERSIZE - 1;
  textBuffer[ textBufferPos ] = '\0';
}

void printToBuffer( const char* format, ... ){
  va_list args;
  va_start( args, format );
  vPrintToBuffer( format, args );
  va_end( args );
}

void print( const char* format, ... ){
  va_list args;
  va_start( args, format );
  va_list stdout_args;
  va_copy( stdout_args, args );
  vprintf( format, stdout_args );
  va_end( stdout_args );
  vPrintToBuffer( format, args );
  va_end( args );
}
char* printToString( const char* format, ... ){
  va_list args;
  va_start( args, format );
  va_list args_copy;
  va_copy( args_copy, args );
  int needed = vsnprintf( NULL, 0, format, args_copy );
  va_end( args_copy );
  if( needed < 0 ){
    va_end( args );
    return NULL; 
  }
  char* str = mem( needed + 1, char );
  vsnprintf( str, needed + 1, format, args );
  va_end( args );
  return str;
}
