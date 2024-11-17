////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4-o1 etc.  //
////////////////////////////////////////////////////////////////////////////////


typedef struct{
  u32 rank;
  u32 size;
  u32 shape[ 4 ];
  u32 strides[ 4 ];
  f32* data;
} tensor;


typedef struct{
  u32 size;
  tensor** stack;
  u32 top;
} tensorStack;


tensorStack* newStack( void );
tensor* newTensor( u32 rank, u32* shape, f32* data );
void deleteStack( tensorStack* ts );
void push( tensorStack* ts, u32 rank, u32* shape, f32* data );
void tensorReshape( tensorStack* ts, u32 index, u32 newRank, u32* newShape );
void pop( tensorStack* ts );
// Functions for printing tensors.
char* formatTensorData( const tensor* t );
void printStack( const tensorStack* ts );


  
