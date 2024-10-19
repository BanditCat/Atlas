#ifndef ATLAS_H_INCLUDED
#define ATLAS_H_INCLUDED

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
#define true ( 1 )
#define false ( 0 )

u32 memc = 0;
#define mem( size, T ) ( memc++, malloc( sizeof( T ) * ( size ) ) )
#define unmem( F ) ( memc--, free( F ) ) 
#define error( msg ) ( fprintf( stderr, "%s", ( msg ) ), exit( 1 ) )

#include "tensor.h"

#endif //ATLAS_H_INCLUDED
