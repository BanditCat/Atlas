//////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4 et al. //
//////////////////////////////////////////////////////////////////////////////

#include "Atlas.h"

#ifndef __EMSCRIPTEN__
#include "SDL2/SDL_syswm.h"
#include <dwmapi.h>
#include <windows.h>
#endif

// Main must define this.
u64 memc = 0;

// The compiled program to be run.
program* prog;
tensorStack* ts;

/* char* testProg = "size;if'dstart'\n" */
/*                  "[10];c'i / 20.0' 0;0;r;[[0.1] [0.2] [3] [4]];print;\n" */
/*                  "[3 3 3];c't.x / 3.0 + t.y / 3.0 + b( vec4( t.z, 0.0, 0.0, 0.0 ) ) + a( vec4( 0.0, t.z, 0.0, 0.0 ) )' 2\n" */
/*                  "l'start';print;l'dstart';0;r;[0 1];t\n" */
/*                  "0;r;0;r;\n" */
/*                  "\n"; */

char* testProg = "\
size;if'start'\n\
[600 600 2];c'( t.z == 0.0 ) ?\
	  ( t.x + 0.5 ) * 4.0 / 600.0 - 2.0 :\
	  ( t.y + 0.5 ) * 4.0 / 600.0 - 2.0\
	 ' 0;\
[600 600 2];c'0.0' 0\n\
1;if'skip'\n\
l'mand'\n\
1;dup;\n\
\n\
[600 600 2];c'( t.z == 0.0 ) ?\
pow( b( vec4( t.xy, 0.0, 0.0 ) ), 2.0 ) - pow( b( vec4( t.xy, 1.0, 0.0 ) ), 2.0 ) + a( vec4( t.xy, 0.0, 0.0 ) ): \
2.0 * b( vec4( t.xy, 0.0, 0.0 ) ) * b( vec4( t.xy, 1.0, 0.0 ) ) + a( vec4( t.xy, 1.0, 0.0 ) )' 2\n\
return\n\
l'skip'\n\
call'mand'\n\
call'mand'\n\
call'mand'\n\
call'mand'\n\
call'mand'\n\
call'mand'\n\
l'start'\n\
[600 600 3];c'a( vec4( t.xy, 0.0, 0.0 ) )' 1\n\
";
  
// Global variables
SDL_Window* window = NULL;
SDL_GLContext glContext;
GLuint shaderProgram;
GLuint vbo;

#ifndef __EMSCRIPTEN__
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
    HRESULT hr = DwmSetWindowAttribute(
      hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, &enable, sizeof( enable ) );
    if( SUCCEEDED( hr ) ){
      ShowWindow( hwnd, SW_HIDE );
      ShowWindow( hwnd, SW_SHOW );
    } else {
      MessageBoxA(
        NULL, "Failed to set dark title bar!", "Error", MB_ICONERROR );
    }
  } else {
    SDL_Log( "Unable to get window handle: %s", SDL_GetError() );
  }
}
#else
int running;  // Simple integer for the running flag in single-threaded mode
#endif

// Variables shared between threads
float shared_zoom = 1.0f;
float shared_offsetX = -0.5f;
float shared_offsetY = 0.0f;

// Vertex Shader Source
const GLchar* vertexSource = "#version 300 es\n"
                             "precision highp float;\n"
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
  "in vec2 fragCoord;\n"
  "out vec4 fragColor;\n"
  "uniform float zoom;\n"
  "uniform sampler2D tex;\n"
  "uniform vec2 offset;\n"
  "uniform vec2 resolution;\n"
  "uniform vec4 strides;\n"
  "uniform vec4 shape;\n"
  "uniform vec2 dims;\n"
  "uniform float toffset;\n"
  "float sampleTensorIndex(vec4 i){\n"
  "  float lindex = dot( i, strides ) + toffset;\n"
  "  float pixel_index = floor( lindex / 4.0 );\n"
  "  float channel = mod( lindex, 4.0 );\n"
  "  vec2 uv = ( vec2( mod( pixel_index, dims.x ), floor( pixel_index / dims.x "
  ") ) + 0.5 ) / dims;\n"
  "  vec4 texel = texture( tex, uv );\n"
  "  if( channel < 1.0 )\n"
  "    return texel.r;\n"
  "  else if( channel < 2.0 )\n"
  "    return texel.g;\n"
  "  else if( channel < 3.0 )\n"
  "    return texel.b;\n"
  "  else\n"
  "    return texel.a;\n"
  "}\n"
  "void main(){\n"
  "  vec2 uv = fragCoord;\n"
  "  uv.x *= resolution.x / resolution.y; // Adjust for aspect ratio\n"
  "  vec2 c = uv * zoom + offset;\n"
  "  vec2 z = c;\n"
  "  int iterations = 0;\n"
  "  const int maxIterations = 100;\n"
  "  for (int i = 0; i < maxIterations; i++){\n"
  "    float x = (z.x * z.x - z.y * z.y ) + c.x;\n"
  "    float y = (2.0 * z.x * z.y ) + c.y;\n"
  "    if(( x * x + y * y ) > 4.0 ) break;\n"
  "    z.x = x;\n"
  "    z.y = y;\n"
  "    iterations++;\n"
  "  }\n"
  "  float color = float( iterations ) / float( maxIterations );\n"
  "  vec4 tindex = vec4( floor( ( fragCoord.xy * 0.5 + 0.5 ) * shape.xy ), 0, "
  "0 );\n"
  "  float tcolor = sampleTensorIndex( tindex );\n"
  "  fragColor = vec4( 0.0, 1.0 - color, tcolor, 1.0 );\n"
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
  prog = newProgramFromString( testProg );
  ts = newStack();

  float zoom, offsetX, offsetY;

  // Main loop
  while( SDL_AtomicGet( &running ) ){
    SDL_PumpEvents();
    // Run the program
    if( !runProgram( ts, prog ) ){
      SDL_AtomicSet( &running, 0 );
      break;
    }

    // Lock mutex to read shared variables
    SDL_LockMutex( data_mutex );
    zoom = shared_zoom;
    offsetX = shared_offsetX;
    offsetY = shared_offsetY;
    SDL_UnlockMutex( data_mutex );

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
    GLint zoomLoc = glGetUniformLocation( shaderProgram, "zoom" );
    glUniform1f( zoomLoc, zoom );

    GLint offsetLoc = glGetUniformLocation( shaderProgram, "offset" );
    glUniform2f( offsetLoc, offsetX, offsetY );

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
#include <emscripten/emscripten.h>

// Main loop function for Emscripten
void main_loop(){
  // Process events
  SDL_Event event;
  while( SDL_PollEvent( &event ) ){
    if( event.type == SDL_QUIT ){
      running = 0;
      emscripten_cancel_main_loop();
    } else if( event.type == SDL_WINDOWEVENT ){
      if( event.window.event == SDL_WINDOWEVENT_CLOSE ){
        running = 0;
        emscripten_cancel_main_loop();
      }
    } else if( event.type == SDL_MOUSEWHEEL ){
      if( event.wheel.y > 0 ){
        shared_zoom *= 0.9f;  // Zoom in
      } else if( event.wheel.y < 0 ){
        shared_zoom *= 1.1f;  // Zoom out
      }
    } else if( event.type == SDL_MOUSEMOTION ){
      if( event.motion.state & SDL_BUTTON_LMASK ){
        float deltaX = (float)event.motion.xrel / 600 * shared_zoom * 2.0f;
        float deltaY = (float)event.motion.yrel / 600 * shared_zoom * 2.0f;
        shared_offsetX -= deltaX;
        shared_offsetY += deltaY;
      }
    } else if( event.type == SDL_KEYDOWN ){
      if( event.key.keysym.sym == SDLK_ESCAPE ){
        running = 0;
        emscripten_cancel_main_loop();
      }
    }
  }

  // Run the program
  if( !runProgram( ts, prog ) ){
    running = 0;
    emscripten_cancel_main_loop();
    return;
  }

  // Rendering code
  // Get current window size
  int windowWidth, windowHeight;
  SDL_GetWindowSize( window, &windowWidth, &windowHeight );

  float zoom = shared_zoom;
  float offsetX = shared_offsetX;
  float offsetY = shared_offsetY;

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
  GLint zoomLoc = glGetUniformLocation( shaderProgram, "zoom" );
  glUniform1f( zoomLoc, zoom );

  GLint offsetLoc = glGetUniformLocation( shaderProgram, "offset" );
  glUniform2f( offsetLoc, offsetX, offsetY );

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
                             800,
                             600,  // Initial window size
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
  prog = newProgramFromString( testProg );
  ts = newStack();
#endif

#ifndef __EMSCRIPTEN__
  // Create rendering and computation thread
  SDL_Thread* renderThread =
    SDL_CreateThread( renderThreadFunction, "RenderThread", NULL );
  if( renderThread == NULL ){
    error( "%s", "Failed to create rendering thread" );
  }

  // Main thread handles SDL event loop
  while( SDL_AtomicGet( &running ) ){
    // Process events
    SDL_Event event;
    while( SDL_PollEvent( &event ) ){
      if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_RESIZED) {
	int width, height;
	SDL_GetWindowSize( window, &width, &height );
	glViewport( 0, 0, width, height );
	DwmFlush();
      } else if( event.type == SDL_QUIT ){
        SDL_AtomicSet( &running, 0 );
      } else if( event.type == SDL_WINDOWEVENT ){
        if( event.window.event == SDL_WINDOWEVENT_CLOSE ){
          SDL_AtomicSet( &running, 0 );
        }
        // No need to handle SDL_WINDOWEVENT_RESIZED
      } else if( event.type == SDL_MOUSEWHEEL ){
        SDL_LockMutex( data_mutex );
        if( event.wheel.y > 0 ){
          shared_zoom *= 0.9f;  // Zoom in
        } else if( event.wheel.y < 0 ){
          shared_zoom *= 1.1f;  // Zoom out
        }
        SDL_UnlockMutex( data_mutex );
      } else if( event.type == SDL_MOUSEMOTION ){
        if( event.motion.state & SDL_BUTTON_LMASK ){
          SDL_LockMutex( data_mutex );
          float deltaX = (float)event.motion.xrel / 600 * shared_zoom * 2.0f;
          float deltaY = (float)event.motion.yrel / 600 * shared_zoom * 2.0f;
          shared_offsetX -= deltaX;
          shared_offsetY += deltaY;
          SDL_UnlockMutex( data_mutex );
        }
      } else if( event.type == SDL_KEYDOWN ){
        if( event.key.keysym.sym == SDLK_ESCAPE )
          SDL_AtomicSet( &running, 0 );
        break;
      }
    }
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

  dbg( "mem count %llu", memc );

  return 0;
}

void test( void ){
  // Create the root of the trie
  trieNode* root = newTrieNode( NULL, 0 );

  // Insert some keys and values
  trieInsert( root, "apple", 100 );
  trieInsert( root, "app", 50 );
  trieInsert( root, "banana", 150 );
  trieInsert( root, "band", 75 );
  trieInsert( root, "bandana", 200 );

  // Search for keys
  const char* keysToSearch[] = {
    "apple", "app", "banana", "band", "bandana", "bandit", "apricot", "ban" };
  size_t numKeys = sizeof( keysToSearch ) / sizeof( keysToSearch[ 0 ] );

  for( size_t i = 0; i < numKeys; ++i ){
    u32 value;
    bool found = trieSearch( root, keysToSearch[ i ], &value );
    if( found ){
      printf( "Key '%s' found with value: %u\n", keysToSearch[ i ], value );
    } else {
      printf( "Key '%s' not found.\n", keysToSearch[ i ] );
    }
  }

  // Clean up
  deleteTrieNode( root );
}
