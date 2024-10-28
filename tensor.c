////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4-o1.      //
////////////////////////////////////////////////////////////////////////////////



#include "Atlas.h" 


tensor* newTensor( u64 rank, u64* shape, u8* data ){
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

void pushAdd( tensorStack* ts ){
  u64 shape[] = { 256, 256, 2 };
  u8 data[ 65536 * 2 ] = {};
  for( u64 x = 0; x < 256; ++x ){
    for( u64 y = 0; y < 256; ++y ){
      u64 res = x + y;
      data[ ( x + y * 256 ) * 2 + 0 ] = res & 255;
      data[ ( x + y * 256 ) * 2 + 1 ] = res >> 8;
    }
  }
  push( ts, 3, shape, data );
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
  tensor* ret = mem( 1, tensor );
  ret->rank = ( t->rank - rankReduction ) + ( indexTensor->rank - 2 );
  ret->shape = mem( ret->rank, u64 );
  for( u64 i = 0; i < ( indexTensor->rank - 2 ); ++i )
    ret->shape[ i ] = indexTensor->shape[ i ];
  for( u64 i = ( indexTensor->rank - 2 ); i < ret->rank; ++i )
    ret->shape[ i ] = t->shape[ ( i - ( indexTensor->rank - 2 ) ) + rankReduction ];
  ret->strides = mem( ret->rank, u64 );
  for( int i = ret->rank - 1, m = 1; i >= 0; m *= ret->shape[ i-- ] )
    ret->strides[ i ] = m;
  ret->size = ret->rank ? ret->strides[ 0 ] * ret->shape[ 0 ] : 1;
  ret->data = mem( ret->size, u8 );

  // Allocate arrays for indices
  u64* ret_indices = malloc( ret->rank * sizeof( u64 ) );
  u64* t_indices = malloc( t->rank * sizeof( u64 ) );
  
  // Iterate over each element in ret->data
  for( u64 idx = 0; idx < ret->size; ++idx ){
    u64 temp_idx = idx;

    for( u64 i = 0; i < ret->rank; ++i ) {
      ret_indices[ i ] = temp_idx / ret->strides[ i ];
      temp_idx %= ret->strides[ i ];
    }

    u64 indexTensor_indices[ indexTensor->rank ];
    for( u64 i = 0; i < indexTensor->rank - 2; ++i ) {
      indexTensor_indices[ i ] = ret_indices[ i ];
    }

    for( u64 k = 0; k < rankReduction; ++k ) {
      u64 index = 0;

      for( u64 b = 0; b < bytesPerIndex; ++b ) {
	// Set the indices for rank reduction and bytes per index dimensions
	indexTensor_indices[ indexTensor->rank - 2 ] = k;  
	indexTensor_indices[ indexTensor->rank - 1 ] = b; 

	// Compute flat index into indexTensor->data
	u64 indexTensor_flat_idx = 0;
	for( u64 i = 0; i < indexTensor->rank; ++i ) {
	  indexTensor_flat_idx += indexTensor_indices[ i ] *
	    indexTensor->strides[ i ];
	}
	u8 byte = indexTensor->data[ indexTensor_flat_idx ];
	index |= ( (u64)byte ) << ( b * 8 );
      }

      t_indices[ k ] = index;
    }
    // Map remaining indices from ret_indices to t_indices
    for( u64 k = rankReduction; k < t->rank; ++k ){
      u64 ret_idx = ( indexTensor->rank - 2 ) + ( k - rankReduction );
      t_indices[ k ] = ret_indices[ ret_idx ];
    }

    u64 t_offset = 0;
    for( u64 i = 0; i < t->rank; ++i ){
      t_offset += t_indices[ i ] * t->strides[ i ];
    } 

    ret->data[ idx ] = t->data[ t_offset ];
  }
  free( ret_indices );
  free( t_indices );

  ts->stack[ ts->top++ ] = ret;
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
    printf( "Tensor %llu\n", i );
    printf( "Shape:" );
    for( u64 j = 0; j < t->rank; ++j )
      printf( " %llu", t->shape[ j ] );
    printf( "\nStrides:" );
    for( u64 j = 0; j < t->rank; ++j )
      printf( " %llu", t->strides[ j ] );
    if( t->size < 256 ){
      char* fd = formatTensorData( t );
      printf( "\n%s\n\n", fd );
      unmem( fd );
    } else
      printf( "\n[large tensor]\n\n" );
  }
}

void tensorReshapeHelper( tensor* t, u64 newRank, u64* newShape ){
  if( !t || !newShape || !newRank || !t->rank )
    error( "Invalid tensor or shape." );
  u64 newSize = 1;
  for( u64 i = 0; i < newRank; ++i )
    newSize *= newShape[ i ];
  if( newSize != t->size )
    error( "New shape size does not match tensor size." );
  
  u64* tp = NULL;
  tp = mem( newRank, u64 );
  memcpy( tp, newShape, sizeof( u64 ) * newRank );
  unmem( t->shape ); t->shape = tp;
  tp = mem( newRank, u64 );
  u64 size = 1;
  for( int i = newRank - 1; i >= 0; --i ){
    tp[ i ] = size;
    size *= newShape[ i ];
  }
  unmem( t->strides ); t->strides = tp;
  t->rank = newRank;
}

void tensorReshape( tensorStack* ts, u64 index, u64 newRank, u64* newShape ){
  tensorReshapeHelper( ts->stack[ index ], newRank, newShape );
}
