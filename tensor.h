////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4-o1 etc.  //
////////////////////////////////////////////////////////////////////////////////


typedef struct {
  u32 rank;              // Rank of the tensor (0 to 4)
  u32 size;              // Total number of elements
  u32 shape[4];          // Dimensions of the tensor
  u32 strides[4];        // Strides for indexing
  GLuint texture;        // OpenGL texture for reading/writing operations
  GLuint framebuffer;    // Framebuffer for rendering into the texture
} tensor;

typedef struct{
  u32 size;
  tensor** stack;
  u32 top;
} tensorStack;


f32* tensorToHostMemory( const tensor* t );
tensorStack* newStack( void );
tensor* newTensor( u32 rank, u32* shape, f32* data );
void deleteStack( tensorStack* ts );
void push( tensorStack* ts, u32 rank, u32* shape, f32* data );
void tensorReshape( tensorStack* ts, u32 index, u32 newRank, u32* newShape );
void pop( tensorStack* ts );
// Functions for printing tensors.
char* formatTensorData( const tensor* t );
void printStack( const tensorStack* ts );


  
