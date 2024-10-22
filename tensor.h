////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4-o1 etc.  //
////////////////////////////////////////////////////////////////////////////////


typedef struct{
  bool ownsData;
  u32 rank;
  u64 size;
  u64* shape;
  u64* strides;
  u8* data;
} tensor;



tensor* newTensor( u32 rank, u64 size, u64* shape, u8* data );
void deleteTensor( tensor* t );
// Function for printing tensors.
char* formatTensorData( const tensor* t );


  
