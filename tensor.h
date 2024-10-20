////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4-o1.      //
////////////////////////////////////////////////////////////////////////////////


typedef struct{
  u32 rank;
  u64 size;
  u64* shape;
  u8* data;
  u64 references;
} tensor;



tensor* newTensor( u32 rank, u64 size, u64* shape, u8* data );
void deleteTensor( tensor* t );
// Function for printing tensors.
char* formatTensorData( const tensor* t );


  
