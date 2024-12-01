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

#include "trie.h"
#include "tensor.h"
#include "program.h"

#endif //ATLAS_H_INCLUDED
