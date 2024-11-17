////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4-o1.      //
////////////////////////////////////////////////////////////////////////////////



#include "Atlas.h" 


tensor* newTensor( u32 rank, u32* shape, f32* data ){
  tensor* ret = mem( 1, tensor );

  if( rank ){
    if( !shape )
      error( "A tensor with non-zero rank was given with no shape." );
    ret->rank = rank;
    u32 size = 1;
    for( int i = rank - 1; i >= 0; --i ){
      ret->strides[ i ] = size;
      size *= shape[ i ];
    }
    ret->size = size;
    memcpy( ret->shape, shape, sizeof( u32 ) * rank );
    ret->data = mem( size, f32 );
    memcpy( ret->data, data, sizeof( f32 ) * size );
  } else {
    ret->rank = 0;
    ret->size = 1;
    ret->data = mem( 1, f32 );
    *(ret->data) = *data;
  }
  return ret;
}

void push( tensorStack* ts, u32 rank, u32* shape, f32* data ){
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

/* // Raises rank by 1 without deallocating data. */
/* void enclose( tensor* t ){ */
/*   if( !t->rank ){ */
/*     t->rank = 1; */
/*     t->shape = mem( 1, u32 ); */
/*     *t->shape = 1; */
/*     t->strides = mem( 1, u32 ); */
/*     *t->strides = 1; */
/*   } else { */
/*     t->rank++; */
/*     u32* np = mem( t->rank, u32 ); */
/*     memcpy( np + 1, t->shape, sizeof( u32 ) * ( t->rank - 1 ) ); */
/*     *np = 1; */
/*     unmem( t->shape ); */
/*     t->shape = np; */
/*     np = mem( t->rank, u32 ); */
/*     memcpy( np + 1, t->strides, sizeof( u32 ) * ( t->rank - 1 ) ); */
/*     *np = *(np + 1); */
/*     unmem( t->strides ); */
/*     t->strides = np; */
/*   } */
/* } */
    

/* typedef struct{ */
/*   u32 rank; */
/*   u32 size; */
/*   u32* shape; */
/*   u32* strides; */
/*   f32* data; */
/* } tensor; */
// This pushes a new tensor onto the stack that is tensor t indexed by index.
/* void tensorIndex( tensorStack* ts, u32 indexIndex, u32 tIndex ){ */
/*   if( indexIndex >= ts->top || tIndex >= ts->top ) */
/*     error( "Stack index out of range in tensorIndex." ); */
/*   tensor* indexTensor = ts->stack[ indexIndex ]; */
/*   tensor* t = ts->stack[ tIndex ]; */
/*   if( indexTensor->rank < 2 ) */
/*     error( "Index must be at least rank 2." ); */
/*   if( t->rank < 1 ) */
/*     error( "Cannot index singleton." ); */

/*   u32 bytesPerIndex = indexTensor->shape[ indexTensor->rank - 1 ]; */
/*   u32 rankReduction = indexTensor->shape[ indexTensor->rank - 2 ]; */
/*   if( rankReduction > t->rank ) */
/*     error( "Index tensor rank reduction greater than target rank." ); */
/*   tensor* ret = mem( 1, tensor ); */
/*   ret->rank = ( t->rank - rankReduction ) + ( indexTensor->rank - 2 ); */
/*   for( u32 i = 0; i < ( indexTensor->rank - 2 ); ++i ) */
/*     ret->shape[ i ] = indexTensor->shape[ i ]; */
/*   for( u32 i = ( indexTensor->rank - 2 ); i < ret->rank; ++i ) */
/*     ret->shape[ i ] = t->shape[ ( i - ( indexTensor->rank - 2 ) ) + rankReduction ]; */
/*   for( int i = ret->rank - 1, m = 1; i >= 0; m *= ret->shape[ i-- ] ) */
/*     ret->strides[ i ] = m; */
/*   ret->size = ret->rank ? ret->strides[ 0 ] * ret->shape[ 0 ] : 1; */
/*   ret->data = mem( ret->size, f32 ); */

/*   // Allocate arrays for indices */
/*   u32* ret_indices = malloc( ret->rank * sizeof( u32 ) ); */
/*   u32* t_indices = malloc( t->rank * sizeof( u32 ) ); */
  
/*   // Iterate over each element in ret->data */
/*   for( u32 idx = 0; idx < ret->size; ++idx ){ */
/*     u32 temp_idx = idx; */

/*     for( u32 i = 0; i < ret->rank; ++i ) { */
/*       ret_indices[ i ] = temp_idx / ret->strides[ i ]; */
/*       temp_idx %= ret->strides[ i ]; */
/*     } */

/*     u32 indexTensor_indices[ indexTensor->rank ]; */
/*     for( u32 i = 0; i < indexTensor->rank - 2; ++i ) { */
/*       indexTensor_indices[ i ] = ret_indices[ i ]; */
/*     } */

/*     for( u32 k = 0; k < rankReduction; ++k ) { */
/*       u32 index = 0; */

/*       for( u32 b = 0; b < bytesPerIndex; ++b ) { */
/* 	// Set the indices for rank reduction and bytes per index dimensions */
/* 	indexTensor_indices[ indexTensor->rank - 2 ] = k;   */
/* 	indexTensor_indices[ indexTensor->rank - 1 ] = b;  */

/* 	// Compute flat index into indexTensor->data */
/* 	u32 indexTensor_flat_idx = 0; */
/* 	for( u32 i = 0; i < indexTensor->rank; ++i ) { */
/* 	  indexTensor_flat_idx += indexTensor_indices[ i ] * */
/* 	    indexTensor->strides[ i ]; */
/* 	} */
/* 	f32 byte = indexTensor->data[ indexTensor_flat_idx ]; */
/* 	index |= ( (u32)byte ) << ( b * 8 ); */
/*       } */

/*       t_indices[ k ] = index; */
/*     } */
/*     // Map remaining indices from ret_indices to t_indices */
/*     for( u32 k = rankReduction; k < t->rank; ++k ){ */
/*       u32 ret_idx = ( indexTensor->rank - 2 ) + ( k - rankReduction ); */
/*       t_indices[ k ] = ret_indices[ ret_idx ]; */
/*     } */

/*     u32 t_offset = 0; */
/*     for( u32 i = 0; i < t->rank; ++i ){ */
/*       t_offset += t_indices[ i ] * t->strides[ i ]; */
/*     }  */

/*     ret->data[ idx ] = t->data[ t_offset ]; */
/*   } */
/*   free( ret_indices ); */
/*   free( t_indices ); */

/*   ts->stack[ ts->top++ ] = ret; */
/* } */

void deleteTensor( tensor* t ){
  if( t->data )
    unmem( t->data );
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
  for( u32 i = 0; i < ts->top; ++i )
    deleteTensor( ts->stack[ i ] );
  unmem( ts->stack );
  unmem( ts );
}
  
void printStack( const tensorStack* ts ){
  for( u32 i = ts->top - 1; i < ts->top; --i ){
    tensor* t = ts->stack[ i ];
    printf( "Tensor %u\n", i );
    printf( "Shape:" );
    for( u32 j = 0; j < t->rank; ++j )
      printf( " %u", t->shape[ j ] );
    printf( "\nStrides:" );
    for( u32 j = 0; j < t->rank; ++j )
      printf( " %u", t->strides[ j ] );
    if( t->size < 256 ){
      char* fd = formatTensorData( t );
      printf( "\n%s\n\n", fd );
      unmem( fd );
    } else
      printf( "\n[large tensor]\n\n" );
  }
}

void tensorReshapeHelper( tensor* t, u32 newRank, u32* newShape ){
  if( !t || !newShape || !newRank || !t->rank )
    error( "Invalid tensor or shape." );
  u32 newSize = 1;
  for( u32 i = 0; i < newRank; ++i )
    newSize *= newShape[ i ];
  if( newSize != t->size )
    error( "New shape size does not match tensor size." );
  
  memcpy( t->shape, newShape, sizeof( u32 ) * newRank );
  u32 size = 1;
  for( int i = newRank - 1; i >= 0; --i ){
    t->strides[ i ] = size;
    size *= newShape[ i ];
  }

  t->rank = newRank;
}

void tensorReshape( tensorStack* ts, u32 index, u32 newRank, u32* newShape ){
  tensorReshapeHelper( ts->stack[ index ], newRank, newShape );
}
