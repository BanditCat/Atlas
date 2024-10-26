////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4-o1.      //
////////////////////////////////////////////////////////////////////////////////



#include "Atlas.h" 


tensor* newTensor( u32 rank, u64* shape, u8* data ){
  tensor* ret = mem( 1, tensor );

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
  return ret;
}

void push( tensorStack* ts, u32 rank, u64* shape, u8* data ){
  // Grow stack if necessary. 
  if( ts->top >= ts->size ){
    ts->size *= 2;
    tensor** ns = mem( ts->size, tensor* );
    memcpy( ns, ts->stack, sizeof( tensor* ) * ( ts->top - 1 ) );
    unmem( ts->stack );
    ts->stack = ns;
  }
  ts->stack[ ts->top++ ] = newTensor( rank, shape, data );
}


// Raises rank by 1 without deallocating data.
void enclose( tensor* t ){
  if( !t->rank ){
    t->rank = 1;
    t->shape = mem( 1, u64 );
    *t->shape = 1;
    t->strides = mem( 1, u64 );
    *t->strides = 1;
  } else {
    t->rank++;
    u64* np = mem( t->rank, u64 );
    memcpy( np + 1, t->shape, sizeof( u64 ) * ( t->rank - 1 ) );
    *np = 1;
    unmem( t->shape );
    t->shape = np;
    np = mem( t->rank, u64 );
    memcpy( np + 1, t->strides, sizeof( u64 ) * ( t->rank - 1 ) );
    *np = *(np + 1);
    unmem( t->strides );
    t->strides = np;
  }
}
    

/* typedef struct{ */
/*   bool ownsData; */
/*   u32 rank; */
/*   u64 size; */
/*   u64* shape; */
/*   u64* strides; */
/*   u8* data; */
/* } tensor; */
// This pushes a new tensor onto the stack that is tensor t indexed by index.
void tensorIndex( tensorStack* ts, u64 indexIndex, u64 tIndex ){
  if( indexIndex >= ts->top || tIndex >= ts->top )
    error( "Stack index out of range in tensorIndex." );
  tensor* indexTensor = ts->stack[ indexIndex ];
  tensor* t = ts->stack[ tIndex ];
  if( indexTensor->rank < 2 )
    error( "Index must be at least rank 2." );
  if( t->rank < 1 )
    error( "Cannot index singleton." );
  
  u64 bytesPerIndex = indexTensor->shape[ indexTensor->rank - 1 ];
  u64 rankReduction = indexTensor->shape[ indexTensor->rank - 2 ];

  if( rankReduction > t->rank )
    error( "Index tensor rank reduction greater than target rank." );

  u64 index = 0;
  
  for( u64 i = 0; i < rankReduction; ++i ){
    u64 iindex = 0;
    u64 multend = 1; 
    for( u64 j = 0; j < bytesPerIndex; ++j ){
      iindex += multend * indexTensor->data[ j + i * bytesPerIndex ]; 
      multend <<= 8;
    }
    iindex *= t->strides[ i ];
    index += iindex;
  }

  


  // Please fill the code in here. Notice that the last dimension for the index tensor
  // is so that byte tensors can index as arbirtrary precision. An index tensor of
  // rank two should select a slice of the target tensor. shape[ 0 ] is the number
  // of ranks that will be consumed by indexing as above. Please adhere to my style
  // as closely as possible.
    
  /* u64 index = 0; */
  /* u64 multend = 1; */
  /* for( u64 i = 0; i < indexTensor->shape[ 0 ]; ++i ){ */
  /* } */
  /* if( index >= t->shape[ 0 ] ) */
  /*   error( "Index out of range." ); */
  /* // Replace index with the tensor slice. */
  /* indexTensor->rank = t->rank - 1; */
  /* indexTensor->size = t->size / t->shape[ 0 ]; */
  /* if( indexTensor->rank != 0 ){ */
  /*   unmem( indexTensor->shape ); */
  /*   indexTensor->shape = mem( indexTensor->rank, u64 ); */
  /*   for( u64 i = 0; i < indexTensor->rank; ++i ) */
  /*     indexTensor->shape[ i ] = t->shape[ i + 1 ]; */
  /*   unmem( indexTensor->strides ); */
  /*   indexTensor->strides = mem( indexTensor->rank, u64 ); */
  /*   for( u64 i = 0; i < indexTensor->rank; ++i ) */
  /*     indexTensor->strides[ i ] = t->strides[ i + 1 ]; */
  /* } else { */
  /*   if( indexTensor->strides ) */
  /*     unmem( indexTensor->strides ); */
  /*   indexTensor->strides = NULL; */
  /*   if( indexTensor->shape ) */
  /*     unmem( indexTensor->shape ); */
  /*   indexTensor->shape = NULL; */
  /* } */
  /* if( indexTensor->ownsData ) */
  /*   unmem( indexTensor->data ); */
  /* indexTensor->ownsData = false; */
  /* indexTensor->data = t->data + t->strides[ 0 ] * index; */
}

void deleteTensor( tensor* t ){
   if( t->data )
     unmem( t->data );
   if( t->shape )
     unmem( t->shape );
   if( t->strides )
     unmem( t->strides );
   unmem( t );
}

void pop( tensorStack* ts ){
  if( !ts->top )
    error( "Atempt to pop an empty stack!" );
  deleteTensor( ts->stack[ --ts->top ] );
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
    deleteTensor( ts->stack[ i ] );
  unmem( ts->stack );
  unmem( ts );
}
  
void printStack( const tensorStack* ts ){
  for( u64 i = ts->top - 1; i < ts->top; --i ){
    tensor* t = ts->stack[ i ];
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
