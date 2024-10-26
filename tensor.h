////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4-o1 etc.  //
////////////////////////////////////////////////////////////////////////////////


typedef struct{
  u32 rank;
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
tensor* newTensor( u32 rank, u64* shape, u8* data );
void deleteStack( tensorStack* ts );
void push( tensorStack* ts, u32 rank, u64* shape, u8* data );
void tensorIndex( tensorStack* ts, u64 indexIndex, u64 tIndex );
void pop( tensorStack* ts );
// Functions for printing tensors.
char* formatTensorData( const tensor* t );
void printStack( const tensorStack* ts );


  
