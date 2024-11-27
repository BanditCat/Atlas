////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4 et al.   //
////////////////////////////////////////////////////////////////////////////////


typedef struct{
  u32 rank;              // Rank of the tensor (0 to 4)
  u32 size;              // Total number of elements
  u32 shape[4];          // Dimensions of the tensor
  s32 strides[4];        // Strides for indexing
  u32 offset;
  bool gpu;              // True if in gpu memory, false if in cpu memory.
  union{
    f32* data;
    struct{
      GLuint texture;        // OpenGL texture for reading/writing operations
      GLuint framebuffer;    // Framebuffer for rendering into the texture
      u32 width, height;
    } tex;
  };
  bool ownsData;
} tensor;

typedef struct{
  GLuint program;
  GLuint dimsLocation;
  GLuint stridesLocation;
  GLuint VBO;
} compute;

typedef struct{
  u32 size;
  tensor** stack;
  u32 top;
} tensorStack;


tensor* copyTensor( const tensor* t );
void tensorToHostMemory( tensor* t );
void tensorToGPUMemory( tensor* t );
tensorStack* newStack( void );
tensor* newTensor( u32 rank, u32* shape, f32* data );
compute* makeCompute( const char* glsl, u32 argCount );
void deleteCompute( compute* i );
tensor* newTensorInitialized( u32 rank, u32* shape, const compute* initializer );
void deleteTensor( tensor* t );
void deleteStack( tensorStack* ts );
void push( tensorStack* ts, tensor* t );
void tensorReshape( tensorStack* ts, u32 index, u32 newRank, u32* newShape );  // BUGBUG
void tensorTranspose( tensorStack* ts, u32 index, u32 axis1, u32 axis2 );
void tensorReverse( tensorStack* ts, u32 index, u32 axis );
void pop( tensorStack* ts );
// Functions for printing tensors. These put the tensor in cpu memory if not already there.
char* formatTensorData( tensor* t );
void printStack( tensorStack* ts );


  
