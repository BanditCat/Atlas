////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4-o1 etc.  //
////////////////////////////////////////////////////////////////////////////////


typedef struct{
  u64 rank;
  u64 size;
  u64* shape;
  u64* strides;
  u8* data;
} tensor;


typedef struct{
  u64 size;
  tensor** stack;
  u64 top;
} tensorStack;


tensorStack* newStack( void );
tensor* newTensor( u64 rank, u64* shape, u8* data );
void deleteStack( tensorStack* ts );
void push( tensorStack* ts, u32 rank, u64* shape, u8* data );
// Push the 8 bit add tensor
void pushAdd( tensorStack* ts );
void tensorIndex( tensorStack* ts, u64 indexIndex, u64 tIndex );
void tensorReshape( tensorStack* ts, u64 index, u64 newRank, u64* newShape );
void pop( tensorStack* ts );
// Functions for printing tensors.
char* formatTensorData( const tensor* t );
void printStack( const tensorStack* ts );


  
