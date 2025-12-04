////////////////////////////////////////////////////////////////////////////////
// Copyright © 2025 Jon DuBois. Written with the assistance of GPT-4 et al.   //
////////////////////////////////////////////////////////////////////////////////

#ifndef ATLAS_H_INCLUDED
#define ATLAS_H_INCLUDED
#define _CRT_SECURE_NO_WARNINGS

#include "SDL2/SDL.h"
#ifndef __EMSCRIPTEN__
#include "GL/glew.h"  // For managing OpenGL extensions
#else
#include <GLES3/gl3.h>
#include <SDL2/SDL_opengles2.h>
#include <emscripten/emscripten.h>
#endif

#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//#include <GL/gl.h>
#include <stdio.h>


// The current workspace, this will be prefixed onto all variables and labels in program.c:addStep.  This may be the empty string "" but must not be null, as no checks are performed.
extern char* workspace;


// count newlines
static inline size_t newlines( const char* str ){
  size_t n = 0;
  for(const char *p = str; *p; ++p)
        n += (*p == '\n');
    return n;
}
// determine if a string is floating point.
static inline bool isfloat(const char *str) {
    char *endptr;
    strtof(str, &endptr);
    
    // Skip trailing whitespace
    while (isspace(*endptr)) endptr++;
    
    // Valid if we parsed something and reached the end
    return endptr != str && *endptr == '\0';
}

#ifdef DEBUG
static inline void checkFramebufferStatus( GLuint framebuffer ){
  glBindFramebuffer( GL_FRAMEBUFFER, framebuffer );

  GLenum status = glCheckFramebufferStatus( GL_FRAMEBUFFER );
  switch( status ){
  case GL_FRAMEBUFFER_COMPLETE:
    printf( "Framebuffer is complete.\n" );
    break;
  case GL_FRAMEBUFFER_UNDEFINED:
    printf( "Framebuffer is undefined.\n" );
    break;
  case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
    printf( "Framebuffer has incomplete attachment.\n" );
    break;
  case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
    printf( "Framebuffer is missing an attachment.\n" );
    break;
  case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
    printf( "Framebuffer has incomplete draw buffer.\n" );
    break;
  case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
    printf( "Framebuffer has incomplete read buffer.\n" );
    break;
  case GL_FRAMEBUFFER_UNSUPPORTED:
    printf( "Framebuffer format is unsupported.\n" );
    break;
  case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
    printf( "Framebuffer has incomplete multisample settings.\n" );
    break;
  case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
    printf( "Framebuffer has incomplete layer targets.\n" );
    break;
  default:
    printf( "Unknown framebuffer status: 0x%x\n", status );
    break;
  }

  // Unbind framebuffer to avoid affecting other operations
  glBindFramebuffer( GL_FRAMEBUFFER, 0 );
}
#else
static inline void checkFramebufferStatus( GLuint framebuffer ){
}
#endif

// Function to translate OpenGL error codes to human-readable strings
static inline const char* GetGLErrorString( GLenum error ){
  switch( error ){
  case GL_NO_ERROR:
    return "No error has been recorded.";
  case GL_INVALID_ENUM:
    return "An unacceptable value is specified for an enumerated argument.";
  case GL_INVALID_VALUE:
    return "A numeric argument is out of range.";
  case GL_INVALID_OPERATION:
    return "The specified operation is not allowed in the current state.";
  case GL_INVALID_FRAMEBUFFER_OPERATION:
    return "The framebuffer object is not complete.";
  case GL_OUT_OF_MEMORY:
    return "There is not enough memory left to execute the command.";
  default:
    return "Unknown OpenGL error.";
  }
}

#ifdef DEBUG
// Macro to check and log OpenGL errors
#define CHECK_GL_ERROR()                                                       \
  do {                                                                         \
    GLenum err;                                                                \
    while( ( err = glGetError() ) != GL_NO_ERROR ){                            \
      fprintf( stderr,                                                         \
               "OpenGL Error: %s (0x%X) at %s:%d\n",                           \
               GetGLErrorString( err ),                                        \
               err,                                                            \
               __FILE__,                                                       \
               __LINE__ );                                                     \
      fprintf( stdout,                                                         \
               "OpenGL Error: %s (0x%X) at %s:%d\n",                           \
               GetGLErrorString( err ),                                        \
               err,                                                            \
               __FILE__,                                                       \
               __LINE__ );                                                     \
    }                                                                          \
  } while( 0 )
#else
#define CHECK_GL_ERROR()                                                       \
  { (void)0; }
#endif

#if ULLONG_MAX != 18446744073709551615ULL
#error bad long size
#endif
#if UINT_MAX != 4294967295
#error bad int size
#endif
#if UCHAR_MAX != 255
#error bad char size
#endif

typedef unsigned long long int u64;
typedef signed long long int s64;
typedef unsigned int u32;
typedef signed int s32;
typedef unsigned char u8;
typedef signed char s8;
typedef float f32;
typedef double f64;

// The main poll function, can be called to get active input.
void mainPoll( void );

////////////////////////////////////////////////////////////////////
// Global state

extern bool depthTest;
extern bool additive;

////////////////////////////////////////////////////////////////////
// Memory instrumentation

#ifdef DEBUG

typedef struct MemAlloc{
  void* ptr;
  size_t size;
  const char* file;
  int line;
  int freed;  // Indicates if the memory has been freed
  struct MemAlloc* next;
} MemAlloc;

extern MemAlloc* mem_list;
extern SDL_mutex* mem_list_mutex;  // Use SDL mutex

static inline void*
mem_track( size_t count, size_t size, const char* file, int line ){
  // Initialize the mutex if it's not already initialized
  if( !mem_list_mutex ){
    mem_list_mutex = SDL_CreateMutex();
    if( !mem_list_mutex ){
      fprintf( stderr, "Failed to create mutex: %s\n", SDL_GetError() );
      exit( EXIT_FAILURE );
    }
  }

  void* ptr = calloc( count, size );
  if( !ptr ){
    fprintf( stderr, "Memory allocation failed at %s:%d\n", file, line );
    exit( EXIT_FAILURE );
  }
  MemAlloc* alloc = calloc( 1, sizeof( MemAlloc ) );
  if( !alloc ){
    fprintf(
      stderr, "Failed to allocate memory for tracking at %s:%d\n", file, line );
    free( ptr );
    exit( EXIT_FAILURE );
  }
  alloc->ptr = ptr;
  alloc->size = count * size;
  alloc->file = file;
  alloc->line = line;
  alloc->freed = 0;  // Set freed flag to 0

  SDL_LockMutex( mem_list_mutex );
  alloc->next = mem_list;
  mem_list = alloc;
  SDL_UnlockMutex( mem_list_mutex );

  return ptr;
}

static inline void unmem_track( void* ptr, const char* file, int line ){
  if( !ptr ){
    fprintf( stderr, "Attempted to free NULL pointer at %s:%d\n", file, line );
    return;
  }

  if( !mem_list_mutex ){
    fprintf( stderr,
             "Memory tracker mutex not initialized when freeing at %s:%d\n",
             file,
             line );
    return;
  }

  SDL_LockMutex( mem_list_mutex );
  MemAlloc* current = mem_list;
  while( current ){
    if( current->ptr == ptr ){
      if( current->freed ){
        // Double free detected
        fprintf(
          stderr,
          "Double free detected at %s:%d (originally allocated at %s:%d)\n",
          file,
          line,
          current->file,
          current->line );
      } else {
        // Mark as freed and free the memory
        current->freed = 1;
        free( ptr );
      }
      SDL_UnlockMutex( mem_list_mutex );
      return;
    }
    current = current->next;
  }
  SDL_UnlockMutex( mem_list_mutex );
  fprintf( stderr,
           "Attempted to free untracked or already freed memory at %s:%d\n",
           file,
           line );
}

static inline void check_memory_leaks( void ){
  if( mem_list_mutex ){
    SDL_LockMutex( mem_list_mutex );
  }
  MemAlloc* current = mem_list;
  int leaks_found = 0;
  while( current ){
    if( !current->freed ){
      fprintf( stderr,
               "Memory leak of %zu bytes allocated at %s:%d\n",
               current->size,
               current->file,
               current->line );
      leaks_found = 1;
    }
    MemAlloc* to_free = current;
    current = current->next;
    free( to_free );  // Free the tracking struct itself
  }
  mem_list = NULL;
  if( mem_list_mutex ){
    SDL_UnlockMutex( mem_list_mutex );
    SDL_DestroyMutex( mem_list_mutex );
    mem_list_mutex = NULL;
  }
  if( !leaks_found ){
    printf( "No memory leaks detected.\n" );
  }
}
// Macros to automatically capture file and line information
#define mem( count, type )                                                     \
  mem_track( count, sizeof( type ), __FILE__, __LINE__ )
#define unmem( ptr ) unmem_track( ptr, __FILE__, __LINE__ )

#else  // NO DEBUG

extern u64 memc;
#define mem(size, T) mem_check(calloc((size), sizeof(T)), (size) * sizeof(T), __FILE__, __LINE__)

static inline void* mem_check(void* ptr, size_t bytes, const char* file, int line) {
    memc++;
    if (!ptr) {
        printf("OOM: %zu bytes at %s:%d\n", bytes, file, line);
        exit(1);
    }
    return ptr;
}
#define unmem( F ) ( memc--, free( F ) )

#endif  // DEBUG

#define error( msg, ... )                                                      \
  do {                                                                         \
    char* formatted_msg = mem( 1048576, char );                                \
    snprintf( formatted_msg, 1048576, ( msg ), __VA_ARGS__ );                  \
    printf( ( msg ), __VA_ARGS__ );                                            \
    SDL_ShowSimpleMessageBox(                                                  \
      SDL_MESSAGEBOX_ERROR, "Error", formatted_msg, NULL );                    \
    unmem( formatted_msg );                                                    \
    exit( 1 );                                                                 \
  } while( 0 )

#define dbg( msg, ... )                                                        \
  do {                                                                         \
    char* formatted_msg = mem( 1048576, char );                                \
    snprintf( formatted_msg, 1048576, ( msg ), __VA_ARGS__ );                  \
    printf( ( msg ), __VA_ARGS__ );                                            \
    SDL_ShowSimpleMessageBox(                                                  \
      SDL_MESSAGEBOX_INFORMATION, "Debug", formatted_msg, NULL );              \
    unmem( formatted_msg );                                                    \
  } while( 0 )

float getMaxAnisotropy( void );

// Globals
extern SDL_Window* window;
extern SDL_GLContext glContext;
extern f32 mouseWheel;
extern f32 mouseWheelPos;
extern u32 buttons;
extern f32 dx, dy, posx, posy;
extern bool doubleClicks[ 3 ];
extern bool touchClicks[ 3 ];
extern float pinchZoom;
extern u8 keys[ SDL_NUM_SCANCODES ];
extern f64 timeDelta;
extern f64 runTime;
extern GLuint vao;
#define MAX_CONTROLLERS 8
extern SDL_GameController* controllers[ MAX_CONTROLLERS ];
extern SDL_JoystickID joystickIDs[ MAX_CONTROLLERS ];
extern f32 joysticks[ MAX_CONTROLLERS * 21 ];
#ifndef __EMSCRIPTEN__
//extern SDL_mutex* data_mutex;
#endif

#include "tensor.h"
#include "program.h"
#include "trie.h"

bool fileExists( const char* filename );

#ifndef __EMSCRIPTEN__
void reqSwitchToWorkerW( void );
void reqReturnToNormalWindow( void );
#endif  // __EMSCRIPTEN__

#endif  // ATLAS_H_INCLUDED
