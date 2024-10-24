////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4-o1.      //
////////////////////////////////////////////////////////////////////////////////



#include "Atlas.h" 



void push( tensorStack* ts, u32 rank, u64* shape, u8* data ){
  // Grow stack if necessary. 
  if( ts->top >= ts->size ){
    ts->size *= 2;
    tensor* ns = mem( ts->size, tensor );
    memcpy( ns, ts->stack, sizeof( tensor ) * ( ts->top - 1 ) );
    unmem( ts->stack );
    ts->stack = ns;
  }

  tensor* ret = &(ts->stack[ts->top++]);

  if( rank ){
    if( !shape )
      error( "A tensor with non-zero rank was given with no shape." );
    ret->rank = rank;
    ret->shape = mem( rank, u64 );
    ret->strides = mem( rank, u64 );
    u64 size = 1;
    for( int i = rank - 1; i >= 0; --i ){
      ret->strides[ i ] = size;
      size *= shape[ i ];
    }
    ret->size = size;
    memcpy( ret->shape, shape, sizeof( u64 ) * rank );
    ret->data = mem( size, u8 );
    memcpy( ret->data, data, sizeof( u8 ) * size );
  } else {
    ret->rank = 0;
    ret->size = 1;
    ret->shape = NULL;
    ret->strides = NULL;
    ret->data = mem( 1, u8 );
    *(ret->data) = *data;
  }
  ret->ownsData = true;
}

/* typedef struct{ */
/*   bool ownsData; */
/*   u32 rank; */
/*   u64 size; */
/*   u64* shape; */
/*   u64* strides; */
/*   u8* data; */
/* } tensor; */
// This uses the top of the stack to index the tensor underneath it, replacing
// the top.
void tensorIndex( tensorStack* ts ){
  if( ts->top < 2 )
    error( "Attempt to index with less than two tensors on the stack." );
  tensor* indexTensor = &(ts->stack[ ts->top - 1 ]);
  tensor* t = &(ts->stack[ ts->top - 2 ]);
  if( indexTensor->rank != 1 )
    error( "Incorrect rank for index in tensor indexing." );
  if( t->rank < 1 )
    error( "Cannot index singleton." );
  u64 index = 0;
  u64 multend = 1;
  for( u64 i = 0; i < indexTensor->shape[ 0 ]; ++i ){
    index += multend * indexTensor->data[ i ];
    multend <<= 8;
  }
  if( index >= t->shape[ 0 ] )
    error( "Index out of range." );
  // Replace index with the tensor slice.
  indexTensor->rank = t->rank - 1;
  indexTensor->size = t->size / t->shape[ 0 ];
  if( indexTensor->rank != 0 ){
    unmem( indexTensor->shape );
    indexTensor->shape = mem( indexTensor->rank, u64 );
    for( u64 i = 0; i < indexTensor->rank; ++i )
      indexTensor->shape[ i ] = t->shape[ i + 1 ];
    unmem( indexTensor->strides );
    indexTensor->strides = mem( indexTensor->rank, u64 );
    for( u64 i = 0; i < indexTensor->rank; ++i )
      indexTensor->strides[ i ] = t->strides[ i + 1 ];
  } else {
    if( indexTensor->strides )
      unmem( indexTensor->strides );
    indexTensor->strides = NULL;
    if( indexTensor->shape )
      unmem( indexTensor->shape );
    indexTensor->shape = NULL;
  }
  if( indexTensor->ownsData )
    unmem( indexTensor->data );
  indexTensor->ownsData = false;
  indexTensor->data = t->data + t->strides[ 0 ] * index;
}

void deleteTensor( tensor* t ){
   if( t->data && t->ownsData )
      unmem( t->data );
    if( t->shape )
      unmem( t->shape );
    if( t->strides )
      unmem( t->strides );
 }

void pop( tensorStack* ts ){
  if( !ts->top )
    error( "Atempt to pop an empty stack!" );
  deleteTensor( &(ts->stack[ --ts->top ]) );
}

tensorStack* newStack( void ){
  tensorStack* ret = mem( 1, tensorStack );
  ret->size = 256;
  ret->stack = mem( ret->size, tensor );
  ret->top = 0;
  return ret;
}

void deleteStack( tensorStack* ts ){
  for( u64 i = 0; i < ts->top; ++i )
    deleteTensor( &(ts->stack[ i ]) );
  unmem( ts->stack );
  unmem( ts );
}
  
void printStack( const tensorStack* ts ){
  for( u64 i = ts->top - 1; i < ts->top; --i ){
    tensor* t = &(ts->stack[ i ]);
    char* fd = formatTensorData( t );
    printf( "Tensor %llu\n", i );
    printf( "Shape:" );
    for( u64 j = 0; j < t->rank; ++j )
      printf( " %llu", t->shape[ j ] );
    printf( "\nStrides:" );
    for( u64 j = 0; j < t->rank; ++j )
      printf( " %llu", t->strides[ j ] );
    printf( "\n%s\n\n", fd );
    unmem( fd );
  }
}
