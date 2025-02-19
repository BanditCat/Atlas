////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4 et al.   //
////////////////////////////////////////////////////////////////////////////////


// This is the number of popped gpu tensors to keep cached.
// THIS LIST IS CRAWLED THROUGH LINEARLY, DONT make it huge

#define TENSOR_CACHE 6


typedef struct{
  u32 rank;              // Rank of the tensor (0 to 4)
  u32 size;              // Total number of elements
  u32 shape[ 4 ];          // Dimensions of the tensor
  s32 strides[ 4 ];        // Strides for indexing
  s32 offset;
  bool gpu;              // True if in gpu memory, false if in cpu memory.
  union{
    f32* data;
    struct{
      GLuint texture;        // OpenGL texture for reading/writing operations
      GLuint framebuffer;    // Framebuffer for rendering into the texture
      GLuint depthbuffer;    // Depth buffer for depth testing.
      u32 width, height, channels; // If channels is non-zero, its a texture.
    } tex;
  };
  bool ownsData;
} tensor;

typedef struct{
  GLuint program;
  GLuint dimsLocation;
  GLuint stridesLocation;
  u32 argCount;
  u32 retCount; 
  GLuint argDimsLocation[ 4 ];
  GLuint argStridesLocation[ 4 ];
  GLuint argToffsetLocation[ 4 ];
  GLuint argTexLocation[ 4 ];
  GLuint* uniformLocs;
  u32 channels;
} compute;

typedef struct{
  u32 allocSize;
  tensor** stack;
  u32 size;
  tensor* cache[ TENSOR_CACHE ];
} tensorStack;

#include "program.h"

void takeOwnership( tensor* t );
tensor* copyTensor( const tensor* t );
void tensorToHostMemory( tensor* t );
void tensorToGPUMemory( tensor* t );
tensorStack* newStack( void );
// Warning! this takes ownership of data and will deallocate it.
tensor* newTensor( u32 rank, const u32* shape, f32* data );
compute* makeCompute( const program* prog, const char* uniforms, const char* vglslpre, const char* glslpre,
		      const char* vglsl, const char* glsl, u32 argCount, u32 retCount, u32 channels );
void deleteCompute( compute* i );
tensor** newTensorsInitialized( program* p, tensorStack* ts, u32 rank, u32* shape,
				const compute* initializer, u32 vertCount );
tensor* tensorFromImageFile( const char* fileName );
tensor* tensorFromString( const char* string );
void deleteTensor( tensor* t );
void deleteStack( tensorStack* ts );
void push( tensorStack* ts, tensor* t );
void tensorReshape( tensorStack* ts, u32 index, u32 newRank, u32* newShape );  // BUGBUG
void tensorTranspose( tensorStack* ts, u32 index, u32 axis1, u32 axis2 );
void tensorReverse( tensorStack* ts, u32 index, u32 axis );
void tensorCat( tensorStack* ts, u32 index1, u32 index2, u32 axis );
void tensorSlice( tensorStack* ts, u32 index, u32 axis, s32 start, s32 end );
void tensorTakeFirst( tensorStack* ts, u32 index );
void tensorTakeLast( tensorStack* ts, u32 index );
void tensorRepeat( tensorStack* ts, u32 index, u32 count );
void tensorEnclose( tensor* t );
void tensorExtrude( tensor* t );
void tensorUnextrude( tensor* t );
// Multiplies matrices in host memory.
void tensorMultiply( tensorStack* ts );
void pop( tensorStack* ts );
// Functions for printing tensors. These put the tensor in cpu memory if not already there.
char* formatTensorData( tensor* t );
void printStack( tensorStack* ts );
bool tensorIsContiguous( const tensor* t );
void tensorEnsureContiguous( tensor* t );
// Returns a string that needs to be deallocated or NULL for nonvectors.
char* tensorToString( tensor* t );
void tensorRotate( tensorStack* ts, u32 index, u32 angleIndex );
void tensorTranslate( tensorStack* ts, u32 index );
void tensorProject( tensorStack* ts, u32 index );

