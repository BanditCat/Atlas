////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4-o1.      //
////////////////////////////////////////////////////////////////////////////////



#include "Atlas.h" 


// Don't forget to unmem.
f32* tensorToHostMemory( const tensor* t ){
  if( t == NULL )
    error( "Tensor is NULL." );

  // Calculate the smallest square dimensions
  u32 size = t->size;
  u32 width = ceil( sqrt( (double)size ) );
  u32 height = ( size + width - 1 ) / width;

  // Allocate memory for the host data
  f32* hostData = mem( size, f32 );
  if( hostData == NULL )
    error( "Failed to allocate host memory." );

  // Allocate temporary buffer for RGBA texture data
  f32* tempData = mem( width * height * 4, f32 ); // RGBA channels
  if( tempData == NULL ){
    unmem( hostData );
    error( "Failed to allocate temporary buffer." );
  }

  // Bind framebuffer and read pixels from the texture
  glBindFramebuffer( GL_FRAMEBUFFER, t->framebuffer );
  glReadPixels( 0, 0, width, height, GL_RGBA, GL_FLOAT, tempData );
  glBindFramebuffer( GL_FRAMEBUFFER, 0 );

  // Extract the actual tensor data
  memcpy( hostData, tempData, size * sizeof( f32 ) );

  // Free the temporary buffer
  unmem( tempData );

  return hostData;
}


tensor* newTensor( u32 rank, u32* shape, f32* data ){
  tensor* ret = mem( 1, tensor );

  if( rank > 4 )
    error( "Rank exceeds maximum of 4." );

  // Initialize basic properties
  ret->rank = rank;
  ret->size = 1;
  for( u32 i = 0; i < rank; ++i ){
    ret->shape[ i ] = shape[ i ];
    ret->strides[ i ] = ret->size;
    ret->size *= shape[ i ];
  }
  for( u32 i = rank; i < 4; ++i ){
    ret->shape[ i ] = 0;
    ret->strides[ i ] = 0;
  }

  // Compute the smallest square dimensions
  u32 size = ret->size;
  u32 width = ceil( sqrt( (double)size ) ); // Start with a square root estimate
  u32 height = ( size + width - 1 ) / width; // Ensure it fits the data

  // Create OpenGL texture
  glGenTextures( 1, &ret->texture);
  glBindTexture( GL_TEXTURE_2D, ret->texture );
  glTexImage2D( GL_TEXTURE_2D, 0, 34836, width, height, 0, GL_RGBA, GL_FLOAT, NULL );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  glBindTexture( GL_TEXTURE_2D, 0 );

  // Create framebuffer
  glGenFramebuffers( 1, &ret->framebuffer);
  glBindFramebuffer( GL_FRAMEBUFFER, ret->framebuffer );
  glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ret->texture, 0 );

  if( glCheckFramebufferStatus( GL_FRAMEBUFFER ) != GL_FRAMEBUFFER_COMPLETE )
    error( "Framebuffer is not complete." );
  
  glBindFramebuffer( GL_FRAMEBUFFER, 0 );
  
  if( !data )
    error( "Null data!" );

  glBindTexture( GL_TEXTURE_2D, ret->texture );
  
  // Prepare temporary buffer to fit the texture size
  f32* paddedData = (f32*)calloc(width * height * 4, sizeof(f32)); // RGBA channels
  memcpy( paddedData, data, size * sizeof(f32) ); // Copy data to padded buffer
  glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_FLOAT, paddedData );
  free( paddedData );
  glBindTexture( GL_TEXTURE_2D, 0 );

  return ret;
}
void deleteTensor( tensor* t ){
  if( t == NULL )
    return;
  if( t->texture ){
    glDeleteTextures( 1, &t->texture );
    t->texture = 0;
  }
  if( t->framebuffer ){
    glDeleteFramebuffers( 1, &t->framebuffer );
    t->framebuffer = 0;
  }

  // Free the tensor structure memory
  unmem( t );
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
