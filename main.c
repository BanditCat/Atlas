//////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4 et al. //
//////////////////////////////////////////////////////////////////////////////

#include "Atlas.h"

#ifndef __EMSCRIPTEN__
#include "SDL2/SDL_syswm.h"
#include <dwmapi.h>
#include <windows.h>
#endif



// Main must define these.

#ifdef DEBUG
MemAlloc* mem_list = NULL;
SDL_mutex* mem_list_mutex = NULL; // Use SDL mutex
#else
u64 memc = 0;
#endif

s32 mouseWheelDelta = 0;
bool doubleClicks[ 3 ] = { 0 };
bool touchClicks[ 3 ] = { 0 };
float pinchZoom = 0.0;

SDL_Window* window = NULL;
SDL_GLContext glContext;

// The compiled program to be run.
program* prog;
tensorStack* ts;

// Global variables
GLuint shaderProgram;
GLuint vbo;


void start( program** prog, tensorStack** ts, const char* fileName ){
  const char* realName = fileName ? fileName : "main.atl";
  if( !fileExists( realName ) )
    error( "File %s does not exist.", realName );
  *prog = newProgramFromFile( realName );
  *ts = newStack();
}


// Here we define the touch interface.
#ifdef __EMSCRIPTEN__
#include <emscripten/html5.h>
// Globals for dealing with touch.


EMSCRIPTEN_KEEPALIVE
EM_BOOL onTouch( int eventType, const EmscriptenTouchEvent *touchEvent, void *userData ){
  static float oldPinchZoom = 0.0;
  static float newPinchZoom = 0.0;
  if( eventType == EMSCRIPTEN_EVENT_TOUCHEND ){
    for( int i = 0; i < 3; ++i ){
      touchClicks[i] = 0;
    }
    oldPinchZoom = 0.0;
  } else{
    if( touchEvent->numTouches == 1 ){
      touchClicks[ 0 ] = 1;
      oldPinchZoom = 0.0;
    } else
      touchClicks[ 0 ] = 0;
    if( touchEvent->numTouches == 2 ){
      float x1 = touchEvent->touches[0].canvasX;
      float y1 = touchEvent->touches[0].canvasY;
      float x2 = touchEvent->touches[1].canvasX;
      float y2 = touchEvent->touches[1].canvasY;
      newPinchZoom = sqrtf((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
      mouseWheelDelta += newPinchZoom;
      if( oldPinchZoom != 0.0 ){
	mouseWheelDelta = 1;
	pinchZoom = ( newPinchZoom - oldPinchZoom );
      }
      oldPinchZoom = newPinchZoom;
      touchClicks[ 1 ] = 1;
    }else
      touchClicks[ 1 ] = 0;
    if( touchEvent->numTouches == 3 ){
      touchClicks[ 2 ] = 1;
      oldPinchZoom = 0.0;
    }else
      touchClicks[ 2 ] = 0;
  }
  return EM_TRUE;
}

#endif // touch interface



#ifndef __EMSCRIPTEN__
void APIENTRY openglDebugCallback( GLenum source, GLenum type, GLuint id,
				   GLenum severity, GLsizei length,
				   const GLchar* message, const void* userParam ){
  dbg( "OpenGL Debug Message:\n" );
  dbg( "Source: %d, Type: %d, ID: %d, Severity: %d\n", source, type, id, severity );
  dbg( "Message: %s\n", message );
}

void enableDebugCallback() {
  glEnable( GL_DEBUG_OUTPUT );
  glEnable( GL_DEBUG_OUTPUT_SYNCHRONOUS );
  glDebugMessageCallback( openglDebugCallback, NULL );
}


// Thread synchronization variables
SDL_atomic_t running;
SDL_mutex* data_mutex = NULL;
void SetDarkTitleBar( SDL_Window* sdlWindow ){
  SDL_SysWMinfo wmInfo;
  SDL_VERSION( &wmInfo.version );

  if( SDL_GetWindowWMInfo( sdlWindow, &wmInfo ) ){
    HWND hwnd = wmInfo.info.win.window;
    BOOL enable = TRUE;

    // Apply dark mode attribute
    HRESULT hr = DwmSetWindowAttribute( hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, &enable, sizeof(enable));
    if (SUCCEEDED(hr)) {
      ShowWindow(hwnd, SW_MINIMIZE);
      ShowWindow(hwnd, SW_RESTORE);
    } else {
      MessageBoxA(NULL, "Failed to set dark title bar!", "Error", MB_ICONERROR);
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
  while( SDL_PollEvent( &event ) ){
    if( event.type == SDL_QUIT ||
	( event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE ) ){
#ifndef __EMSCRIPTEN__
      SDL_AtomicSet( &running, 0 );
#else      
      emscripten_cancel_main_loop();
#endif      
    } else if( event.type == SDL_MOUSEWHEEL ){
#ifndef __EMSCRIPTEN__
      SDL_LockMutex( data_mutex );
#endif
      mouseWheelDelta += event.wheel.y;
#ifndef __EMSCRIPTEN__
      SDL_UnlockMutex( data_mutex );
#endif      
    } else if( event.type == SDL_MOUSEBUTTONDOWN ){
      if( event.button.clicks == 2 ){
	if( event.button.button & SDL_BUTTON( SDL_BUTTON_LEFT ) )
	  doubleClicks[ 0 ] = 1;
	if( event.button.button & SDL_BUTTON( SDL_BUTTON_RIGHT ) )
	  doubleClicks[ 1 ] = 1;
	if( event.button.button & SDL_BUTTON( SDL_BUTTON_MIDDLE ) )
	  doubleClicks[ 2 ] = 1;
      }
    }
  }
}

// Vertex Shader Source
const GLchar* vertexSource = "#version 300 es\n"
                             "precision highp float;\n"
                             "precision highp int;\n"
                             "precision highp sampler2D;\n"
                             "in vec2 position;\n"
                             "out vec2 fragCoord;\n"
                             "void main(){\n"
                             "  fragCoord = position;\n"
                             "  gl_Position = vec4( position, 0.0, 1.0 );\n"
                             "}\n";

// Fragment Shader Source
const GLchar* fragmentSource =
  "#version 300 es\n"
  "precision highp float;\n"
  "precision highp int;\n"
  "precision highp sampler2D;\n"
  "in vec2 fragCoord;\n"
  "out vec4 fragColor;\n"
  "uniform sampler2D tex;\n"
  "uniform vec2 resolution;\n"
  "uniform vec4 strides;\n"
  "uniform vec4 shape;\n"
  "uniform vec2 dims;\n"
  "uniform float toffset;\n"
  "float sampleTensorIndex(vec4 i){\n"
  "  float lindex = dot( i, strides ) + toffset;\n"
  "  float pixel_index = floor( lindex / 4.0 ) + 0.25;\n"
  "  float channel = mod( lindex, 4.0 );\n"
  "  vec2 uv = ( vec2( mod( pixel_index, dims.x ), floor( pixel_index / dims.x "
  ") ) + 0.25 ) / dims;\n"
  "  vec4 texel = texture( tex, uv );\n"
  "  return texel[ int( channel ) ];\n"
  "}\n"
  "void main(){\n"
  "  vec4 tindex = vec4( floor( ( fragCoord.xy * 0.5 + 0.5 ) * shape.xy ), 0, 0 );\n"
  "  float rcolor = sampleTensorIndex( tindex );\n"
  "  tindex = vec4( floor( ( fragCoord.xy * 0.5 + 0.5 ) * shape.xy ), 1, 0 );\n"
  "  float gcolor = sampleTensorIndex( tindex );\n"
  "  tindex = vec4( floor( ( fragCoord.xy * 0.5 + 0.5 ) * shape.xy ), 2, 0 );\n"
  "  float bcolor = sampleTensorIndex( tindex );\n"
  "  fragColor = vec4( rcolor, gcolor, bcolor, 1.0 );\n"
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

  //enableDebugCallback();
  // Initialize OpenGL
  int windowWidth, windowHeight;
  SDL_GetWindowSize( window, &windowWidth, &windowHeight );
  glViewport( 0, 0, windowWidth, windowHeight );
  glClearColor( 0.0f, 0.0f, 0.0f, 1.0f );
  SDL_GL_SetSwapInterval( 1 );

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
  start( &prog, &ts, data );

  // Main loop
  while( SDL_AtomicGet( &running ) ){
    SDL_PumpEvents();
    // Run the program
    CHECK_GL_ERROR();
    if( !runProgram( ts, &prog ) ){
#ifdef DBG
      check_memory_leaks();
#endif      
      SDL_AtomicSet( &running, 0 );
      break;
    }

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
    if( ts->stack[ ts->size - 1 ]->shape[ 2 ] != 3 )
      error( "%s", "Display tensor not a 3 component tensor of rank 3." );

    glClear( GL_COLOR_BUFFER_BIT );

    glUseProgram( shaderProgram );
    glViewport( 0, 0, windowWidth, windowHeight );

    GLint texLoc = glGetUniformLocation( shaderProgram, "tex" );
    glUniform1i( texLoc, 0 );  // Texture unit 0

    // Bind the texture to texture unit 0
    glActiveTexture( GL_TEXTURE0 );
    glBindTexture( GL_TEXTURE_2D, ts->stack[ ts->size - 1 ]->tex.texture );

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

    SDL_GL_SwapWindow( window );
    DwmFlush();
  }

  // Cleanup
  glDeleteProgram( shaderProgram );
  glDeleteBuffers( 1, &vbo );

  deleteProgram( prog );
  deleteStack( ts );

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
  if( !runProgram( ts, &prog ) ){
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
  if( ts->stack[ ts->size - 1 ]->shape[ 2 ] != 3 )
    error( "%s", "Display tensor not a 3 component tensor of rank 3." );

  glClear( GL_COLOR_BUFFER_BIT );

  glUseProgram( shaderProgram );
  glViewport( 0, 0, windowWidth, windowHeight );

  GLint texLoc = glGetUniformLocation( shaderProgram, "tex" );
  glUniform1i( texLoc, 0 );  // Texture unit 0

  // Bind the texture to texture unit 0
  glActiveTexture( GL_TEXTURE0 );
  glBindTexture( GL_TEXTURE_2D, ts->stack[ ts->size - 1 ]->tex.texture );

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

  SDL_GL_SwapWindow( window );
}
#endif

// Main function
int main( int argc, char* argv[] ){

#ifndef __EMSCRIPTEN__
  // Set the output code page to UTF-8
  SetConsoleOutputCP( CP_UTF8 );
  
  SDL_AtomicSet( &running, 1 );

  // Initialize mutex
  data_mutex = SDL_CreateMutex();
  if( data_mutex == NULL ){
    error( "%s", "Failed to create mutex" );
  }

#else
  running = 1;
  emscripten_set_touchstart_callback( "#canvas", NULL, EM_TRUE, onTouch );
  emscripten_set_touchend_callback( "#canvas", NULL, EM_TRUE, onTouch );
  emscripten_set_touchmove_callback( "#canvas", NULL, EM_TRUE, onTouch );
  emscripten_set_touchcancel_callback( "#canvas", NULL, EM_TRUE, onTouch );
#endif

  setvbuf( stdout, NULL, _IONBF, 0 ); // Unbuffer stdout
  setvbuf( stderr, NULL, _IONBF, 0 ); // Unbuffer stderr

  // Initialize SDL and create window in the main thread
  if( SDL_Init( SDL_INIT_VIDEO ) != 0 )
    error( "SDL_Init Error: %s\n", SDL_GetError() );

  SDL_GL_SetAttribute( SDL_GL_CONTEXT_MAJOR_VERSION, 2 );
  SDL_GL_SetAttribute( SDL_GL_CONTEXT_MINOR_VERSION, 1 );

  // Add SDL_WINDOW_RESIZABLE flag
  window = SDL_CreateWindow( "Atlas",
                             SDL_WINDOWPOS_CENTERED,
                             SDL_WINDOWPOS_CENTERED,
                             768,
                             512,  // Initial window size
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
  int windowWidth, windowHeight;
  SDL_GetWindowSize( window, &windowWidth, &windowHeight );
  glViewport( 0, 0, windowWidth, windowHeight );
  glClearColor( 0.0f, 0.0f, 0.0f, 1.0f );

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
  start( &prog, &ts, NULL );
#endif


#ifndef __EMSCRIPTEN__
  // Create rendering and computation thread
  void* arg = argc == 2 ? argv[ 1 ] : NULL;
  SDL_Thread* renderThread =
    SDL_CreateThread( renderThreadFunction, "RenderThread", arg );
  if( renderThread == NULL ){
    error( "%s", "Failed to create rendering thread" );
  }

  // Main thread handles SDL event loop
  while( SDL_AtomicGet( &running ) ){
    // Process events
    mainPoll();
  }

  // Wait for rendering thread to finish
  SDL_WaitThread( renderThread, NULL );

  SDL_DestroyMutex( data_mutex );
#else
  // Set up the main loop for Emscripten
  emscripten_set_main_loop( main_loop, 0, 1 );
#endif

  // Cleanup
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
  return 0;
}

