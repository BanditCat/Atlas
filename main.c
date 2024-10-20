////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4-o1.      //
////////////////////////////////////////////////////////////////////////////////


#include "Atlas.h" 


u64 memc = 0;
// Example usage
int main( ){
  u8 data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
  u64 shapeArg[] = { 2, 2, 2, 2 };
  tensor* t = newTensor( 4, 16, shapeArg, data );
  char* output = formatTensorData( t );
  deleteTensor( t );
  printf( "%s\n\n Malloc count: %lld", output, memc - 1 );
  unmem( output ); // Free the allocated memory
  return 0;
}
