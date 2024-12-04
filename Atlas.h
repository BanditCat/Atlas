////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4-o1.      //
////////////////////////////////////////////////////////////////////////////////
#ifndef ATLAS_H_INCLUDED
#define ATLAS_H_INCLUDED
#define _CRT_SECURE_NO_WARNINGS


#include "SDL2/SDL.h"
#ifndef __EMSCRIPTEN__
#include "GL/glew.h" // For managing OpenGL extensions
#else
#include <SDL2/SDL_opengles2.h>
#include <GLES3/gl3.h>
#include <emscripten/emscripten.h>
#endif 
#include <stdio.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>
#include <stdbool.h>
#include <math.h>


// Function to translate OpenGL error codes to human-readable strings
static inline const char* GetGLErrorString(GLenum error) {
    switch (error) {
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

/* // Macro to check and log OpenGL errors */
/* #define CHECK_GL_ERROR()                                 \ */
/*     do {                                                 \ */
/*         GLenum err;                                      \ */
/*         while ((err = glGetError()) != GL_NO_ERROR) {   \ */
/*             fprintf(stderr, "OpenGL Error: %s (0x%X) at %s:%d\n", \ */
/*                     GetGLErrorString(err), err, __FILE__, __LINE__); \ */
/*             fprintf(stdout, "OpenGL Error: %s (0x%X) at %s:%d\n", \ */
/*                     GetGLErrorString(err), err, __FILE__, __LINE__); \ */
/*             /\* You can choose to exit or handle the error here *\/ \ */
/*         }                                                \ */
/*     } while (0) */
#define CHECK_GL_ERROR() { (void)0; }

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

extern u64 memc;
#define mem( size, T ) ( memc++, calloc( ( size ), sizeof( T ) ) )
#define unmem( F ) ( memc--, free( F ) ) 
#define error( msg, ... ) ( fflush( stdout ),\
			    fprintf( stdout, ( msg ), __VA_ARGS__ ), fprintf( stdout, "\n" ),\
			    fflush( stdout ), exit( 1 ) )
#define dbg( msg, ... ) ( fflush( stdout ),\
			  fprintf( stdout, ( msg ), __VA_ARGS__ ), fprintf( stdout, "\n" ),\
			  fflush( stdout ) )

// Global for ease.
extern SDL_Window* window;
extern SDL_GLContext glContext;
extern s32 mouseWheelDelta;
#ifndef __EMSCRIPTEN__
extern SDL_mutex* data_mutex;
#endif

#include "trie.h"
#include "tensor.h"
#include "program.h"

#endif //ATLAS_H_INCLUDED
