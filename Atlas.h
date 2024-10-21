////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4-o1.      //
////////////////////////////////////////////////////////////////////////////////
#ifndef ATLAS_H_INCLUDED
#define ATLAS_H_INCLUDED
#define _CRT_SECURE_NO_WARNINGS

#include "SDL2/SDL.h"
#include "GL/glew.h" // For managing OpenGL extensions
#include <stdio.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

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
typedef unsigned int u32;
typedef unsigned char u8;
typedef u8 bool;
typedef float f32;
typedef double f64;
#define true ( 1 )
#define false ( 0 )

extern u64 memc;
#define mem( size, T ) ( memc++, malloc( sizeof( T ) * ( size ) ) )
#define unmem( F ) ( memc--, free( F ) ) 
#define error( msg ) ( fprintf( stderr, "%s", ( msg ) ), exit( 1 ) )

#include "tensor.h"

#endif //ATLAS_H_INCLUDED
