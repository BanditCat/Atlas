////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4 et al.   //
////////////////////////////////////////////////////////////////////////////////



#include "Atlas.h" 

// DANGER this sets owns data to false, therefore the undelyinng data MUST NOT be destroyed BEFORE the copy
// while the programming is running. At exit cleanup, it shouldn't matter the order of deallocation.
tensor* copyTensor( const tensor* t ){
  tensor* ret = mem( 1, tensor );
  memcpy( ret, t, sizeof( tensor ) );
  ret->ownsData = false;
  return ret;    
}
// This converts a tensor to cpu memory.
void tensorToHostMemory( tensor* t ){
  if( t == NULL )
    error( "%s", "Tensor is NULL in tensorToHostMemory." );
  if( !t->gpu )
    return;

  f32* hostData = mem( t->size, f32 );
  f32* tempData = mem( t->tex.width * t->tex.height * 4, f32 ); // RGBA channels

  CHECK_GL_ERROR();
  glBindFramebuffer( GL_FRAMEBUFFER, t->tex.framebuffer );
  CHECK_GL_ERROR();
  glReadPixels( 0, 0, t->tex.width, t->tex.height, GL_RGBA, GL_FLOAT, tempData );
  CHECK_GL_ERROR();
  glBindFramebuffer( GL_FRAMEBUFFER, 0 );
  CHECK_GL_ERROR();

  memcpy( hostData, tempData, t->size * sizeof( f32 ) );

  unmem( tempData );

  if( t->ownsData ){
    if( t->tex.texture ){
      glDeleteTextures( 1, &t->tex.texture );
      t->tex.texture = 0;
    }
    if( t->tex.framebuffer ){
      glDeleteFramebuffers( 1, &t->tex.framebuffer );
      t->tex.framebuffer = 0;
    }
  }
  
  t->data = hostData;
  t->gpu = false;
  t->ownsData = true;
}
void tensorToGPUMemory( tensor* t ){
  if( t->gpu )
    return;
  if( t == NULL )
    error( "%s", "Tensor is NULL in tensorToGPUMemory." );

  // Calculate texture dimensions for GPU storage
  f32* tdata = t->data;
  u32 pixels = ( t->size + 3 ) / 4; // RGBA = 4 floats per pixel
  u32 twidth = ceilf( sqrtf( (f32)pixels ) );
  u32 theight = ( pixels + twidth - 1 ) / twidth;

  // Prepare padded data for texture upload
  f32* paddedData = mem( twidth * theight * 4, f32 );
  memset( paddedData, 0, twidth * theight * 4 * sizeof( f32 ) );
  memcpy( paddedData, t->data, t->size * sizeof( f32 ) );

  t->tex.width = twidth;
  t->tex.height = theight;
    
  glGenTextures( 1, &t->tex.texture );
  glBindTexture( GL_TEXTURE_2D, t->tex.texture );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, t->tex.width, t->tex.height, 0, GL_RGBA, GL_FLOAT, NULL );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

  glGenFramebuffers( 1, &t->tex.framebuffer );
  glBindFramebuffer( GL_FRAMEBUFFER, t->tex.framebuffer );
  glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t->tex.texture, 0 );

  if( glCheckFramebufferStatus( GL_FRAMEBUFFER ) != GL_FRAMEBUFFER_COMPLETE )
    error( "%s", "Framebuffer is not complete." );

  glBindFramebuffer( GL_FRAMEBUFFER, 0 );


  glBindTexture( GL_TEXTURE_2D, t->tex.texture );
  glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, t->tex.width, t->tex.height, GL_RGBA, GL_FLOAT, paddedData );
  glBindTexture( GL_TEXTURE_2D, 0 );

  unmem( paddedData );
  if( t->ownsData ){
    unmem( tdata );
  }

  t->gpu = true;
  t->ownsData = true;
}
// Warning! this takes ownership of data and will deallocate it.
tensor* newTensor( u32 rank, const u32* shape, f32* data ){
  tensor* ret = mem( 1, tensor );

  if( rank > 4 )
    error( "%s", "Rank exceeds maximum of 4." );

  // Initialize basic properties
  ret->rank = rank;
  ret->size = 1;
  ret->offset = 0;
  ret->ownsData = true;
  for( u32 i = 0; i < rank; ++i ){
    ret->shape[ i ] = shape[ i ];
    ret->strides[ rank - i - 1 ] = ret->size;
    ret->size *= shape[ rank - i - 1 ];
  }
  for( u32 i = rank; i < 4; ++i ){
    ret->shape[ i ] = 1;
    ret->strides[ i ] = 1;
  }

  ret->gpu = false;
  
  if( !data )
    error( "%s", "Null data!" );
  ret->data = data;
  
  return ret;
}
void deleteTensor( tensor* t ){
  if( t == NULL )
    return;
  if( t->ownsData ){
    if( t->gpu ){
      if( t->tex.texture ){
	glDeleteTextures( 1, &t->tex.texture );
	t->tex.texture = 0;
      }
      if( t->tex.framebuffer ){
	glDeleteFramebuffers( 1, &t->tex.framebuffer );
	t->tex.framebuffer = 0;
      }
    } else{
      unmem( t->data );
    }
  }
  unmem( t );
}
compute* makeCompute( const char* uniforms, const char* glsl, u32 argCount ){
  // Vertex shader source (simple pass-through)
  compute* ret = mem( 1, compute );
  ret->argCount = argCount;
  const char* vertexShaderSource = "\
    #version 300 es\n\
    precision highp float;\n\
    in vec2 _a_position;\n\
    void main(){\n\
      gl_Position = vec4( _a_position, 0.0, 1.0);\n\
    }\n\
  ";

  // Fragment shader template
  const char* fragmentShaderTemplate = "\
    #version 300 es\n\
    precision highp float;\n\
    out vec4 _a_fragColor;\n\
    uniform vec2 _a_dims; // Texture dimensions\n\
    uniform vec4 _a_strides; // Tensor shape\n\
    \n\
    uniform vec4 _a_astrides;\n\
    uniform float _a_atoffset;\n\
    uniform vec2 _a_adims;\n\
    uniform sampler2D _a_atex;\n\
    float a( vec4 i ){\n\
      float lindex = dot( i, _a_astrides ) + _a_atoffset;\n\
      float pixel_index = floor( lindex / 4.0 );\n\
      float channel = mod( lindex, 4.0 );\n\
      vec2 uv = ( vec2( mod( pixel_index, _a_adims.x ), \n\
                  floor( pixel_index / _a_adims.x) ) + 0.5 ) / _a_adims;\n\
      vec4 texel = texture( _a_atex, uv );\n\
      return texel[ int( channel ) ];\n\
    }\n\
    uniform vec4 _a_bstrides;\n\
    uniform float _a_btoffset;\n\
    uniform vec2 _a_bdims;\n\
    uniform sampler2D _a_btex;\n\
    float b( vec4 i ){\n\
      float lindex = dot( i, _a_bstrides ) + _a_btoffset;\n\
      float pixel_index = floor( lindex / 4.0 );\n\
      float channel = mod( lindex, 4.0 );\n\
      vec2 uv = ( vec2( mod( pixel_index, _a_bdims.x ), \n\
                  floor( pixel_index / _a_bdims.x) ) + 0.5 ) / _a_bdims;\n\
      vec4 texel = texture( _a_btex, uv );\n\
      return texel[ int( channel ) ];\n\
    }\n\
    uniform vec4 _a_cstrides;\n\
    uniform float _a_ctoffset;\n\
    uniform vec2 _a_cdims;\n\
    uniform sampler2D _a_ctex;\n\
    float c( vec4 i ){\n\
      float lindex = dot( i, _a_cstrides ) + _a_ctoffset;\n\
      float pixel_index = floor( lindex / 4.0 );\n\
      float channel = mod( lindex, 4.0 );\n\
      vec2 uv = ( vec2( mod( pixel_index, _a_cdims.x ), \n\
                  floor( pixel_index / _a_cdims.x) ) + 0.5 ) / _a_cdims;\n\
      vec4 texel = texture( _a_ctex, uv );\n\
      return texel[ int( channel ) ];\n\
    }\n\
    uniform vec4 _a_dstrides;\n\
    uniform float _a_dtoffset;\n\
    uniform vec2 _a_ddims;\n\
    uniform sampler2D _a_dtex;\n\
    float d( vec4 i ){\n\
      float lindex = dot( i, _a_dstrides ) + _a_dtoffset;\n\
      float pixel_index = floor( lindex / 4.0 );\n\
      float channel = mod( lindex, 4.0 );\n\
      vec2 uv = ( vec2( mod( pixel_index, _a_ddims.x ), \n\
                  floor( pixel_index / _a_ddims.x) ) + 0.5 ) / _a_ddims;\n\
      vec4 texel = texture( _a_dtex, uv );\n\
      return texel[ int( channel ) ];\n\
    }\n\
    %s\n\
    vec4 _a_toTensorIndices( float i ) {\n\
      vec4 ret;\n\
      ret.x = floor(i / _a_strides.x);\n\
      i -= ret.x * _a_strides.x;\n\
      ret.y = floor(i / _a_strides.y);\n\
      i -= ret.y * _a_strides.y;\n\
      ret.z = floor(i / _a_strides.z);\n\
      i -= ret.z * _a_strides.z;\n\
      ret.w = i;\n\
      return ret;\n\
    }\n\
    void main() {\n\
      float i = ( ( gl_FragCoord.x - 0.5 ) + ( gl_FragCoord.y - 0.5 ) * _a_dims.x ) * 4.0;\n\
      vec4 t = _a_toTensorIndices( i + 0.5 );\n\
      float _a_r = (%s); ++i;\n\
      t = _a_toTensorIndices( i + 0.5 );\n\
      float _a_g = (%s); ++i;\n\
      t = _a_toTensorIndices( i + 0.5 );\n\
      float _a_b = (%s); ++i;\n\
      t = _a_toTensorIndices( i + 0.5 );\n\
      float _a_a = (%s);\n\
      _a_fragColor = vec4( _a_r, _a_g, _a_b, _a_a );\n\
    }\n\
  ";

  // Buffer to hold the final fragment shader source
  u32 bufsize = 65536;
  char* fragmentShaderSource = mem( bufsize, char ); // Adjust size as needed
  int len = snprintf( fragmentShaderSource, bufsize, fragmentShaderTemplate, uniforms, glsl, glsl, glsl, glsl );
  if( len < 0 || len >= bufsize )
    error( "%s", "Shader source exceeds buffer size." );

  //dbg( "%s", fragmentShaderSource );
  // Compile the vertex shader
  GLuint vertexShader = glCreateShader( GL_VERTEX_SHADER );
  glShaderSource( vertexShader, 1, &vertexShaderSource, NULL );
  glCompileShader( vertexShader );

  // Check for vertex shader compilation errors
  GLint status;
  glGetShaderiv( vertexShader, GL_COMPILE_STATUS, &status );
  if( status != GL_TRUE ){
    static char msg[ 2048 ];
    char log[ 1024 ];
    glGetShaderInfoLog( vertexShader, sizeof( log ), NULL, log );
    snprintf( msg, sizeof( msg ), "Vertex shader compilation failed: %s", log );
    glDeleteShader( vertexShader );
    error( "%s", msg );
  }

  // Compile the fragment shader
  GLuint fragmentShader = glCreateShader( GL_FRAGMENT_SHADER );
  const char* p = fragmentShaderSource;
  glShaderSource( fragmentShader, 1, &p, NULL );
  glCompileShader( fragmentShader );
  unmem( fragmentShaderSource );
  
  // Check for fragment shader compilation errors
  glGetShaderiv( fragmentShader, GL_COMPILE_STATUS, &status );
  if( status != GL_TRUE ){
    static char msg[ 512 ];
    char log[ 512 ];
    glGetShaderInfoLog( fragmentShader, sizeof( log ), NULL, log );
    snprintf( msg, sizeof( msg ), "Fragment shader compilation failed: %s", log );
    glDeleteShader( fragmentShader );
    glDeleteShader( vertexShader );
    error( "%s", msg );
  }

  // Create the program and attach both shaders
  ret->program = glCreateProgram();
  glAttachShader( ret->program, vertexShader );
  glAttachShader( ret->program, fragmentShader );

  // Bind attribute locations (if any)
  glBindAttribLocation( ret->program, 0, "_a_position" );

  // Link the program
  glLinkProgram( ret->program );

  // Check for linking errors
  glGetProgramiv( ret->program, GL_LINK_STATUS, &status );
  if( status != GL_TRUE ){
    static char msg[ 512 ];
    char log[ 512 ];
    glGetProgramInfoLog( ret->program, sizeof( log ), NULL, log );
    snprintf( msg, sizeof( msg ), "Program linking failed: %s", log );
    glDeleteProgram( ret->program );
    glDeleteShader( vertexShader );
    glDeleteShader( fragmentShader );
    error( "%s", msg );
  }


  // Get uniforms.
  char sname[ 12 ] = "_a_astrides";
  char toname[ 12 ] = "_a_atoffset";
  char dname[ 9 ] = "_a_adims";
  char tname[ 8 ] = "_a_atex";
  for( u32 i = 0; i < argCount; ++i ){
    sname[ 3 ] = 'a' + i;
    toname[ 3 ] = 'a' + i;
    dname[ 3 ] = 'a' + i;
    tname[ 3 ] = 'a' + i;
    ret->argStridesLocation[ i ] = glGetUniformLocation( ret->program, sname );
    ret->argToffsetLocation[ i ] = glGetUniformLocation( ret->program, toname );
    ret->argDimsLocation[ i ] = glGetUniformLocation( ret->program, dname );
    ret->argTexLocation[ i ] = glGetUniformLocation( ret->program, tname );
  }

  ret->dimsLocation = glGetUniformLocation( ret->program, "_a_dims" );
  ret->stridesLocation = glGetUniformLocation( ret->program, "_a_strides" );

  ret->uboLoc = glGetUniformBlockIndex( ret->program, "vars" );
  glUniformBlockBinding( ret->program, ret->uboLoc, 0 );

  glGenVertexArrays( 1, &ret->VAO );
  glBindVertexArray( ret->VAO );

  f32 vertices[] = {
    -1.0f, -1.0f,
     1.0f, -1.0f,
    -1.0f,  1.0f,
     1.0f,  1.0f  
  };

  glGenBuffers( 1, &ret->VBO );
  glBindBuffer( GL_ARRAY_BUFFER, ret->VBO );
  glEnableVertexAttribArray( 0 );
  glVertexAttribPointer(
			0,                // Attribute location (must match the shader)
			2,                // Number of components per vertex attribute (e.g., vec2)
			GL_FLOAT,         // Data type
			GL_FALSE,         // Normalized
			0,                // Stride (0 if tightly packed)
			(void*)0          // Offset to the first component
			);

  glBufferData( GL_ARRAY_BUFFER, sizeof( vertices ), vertices, GL_STATIC_DRAW );
  glBindBuffer( GL_ARRAY_BUFFER, 0 );
  glBindVertexArray( 0 );
  
  // Cleanup shaders (they're no longer needed once the program is linked)
  glDeleteShader( vertexShader );
  glDeleteShader( fragmentShader );

  return ret;
}
void deleteCompute( compute* i ){
  glDeleteProgram( i->program );
  glDeleteVertexArrays( 1, &i->VAO );
  glDeleteBuffers( 1, &i->VBO );
  unmem( i );
}
tensor* newTensorInitialized( program* p, tensorStack* ts, u32 rank, u32* shape, const compute* compute ){
  tensor* ret = mem( 1, tensor );
  if( compute->argCount > ts->size )
    error( "A compute was called with %u arguments, but the stack size is only %u.",
	   compute->argCount, ts->size );
  for( u32 i = 0; i < compute->argCount; ++i )
    tensorToGPUMemory( ts->stack[ ( ts->size - 1 ) - i ] );
  if( rank > 4 )
    error( "%s", "Rank exceeds maximum of 4." );

  // Initialize basic properties
  ret->rank = rank;
  ret->size = 1;
  ret->offset = 0;
  ret->ownsData = true;
  ret->gpu = true;
  for( u32 i = 0; i < rank; ++i ){
    ret->shape[ i ] = shape[ i ];
    ret->strides[ rank - i - 1 ] = ret->size;
    ret->size *= shape[ rank - i - 1 ];
  }
  for( u32 i = rank; i < 4; ++i ){
    ret->shape[ i ] = 1;
    ret->strides[ i ] = 1;
  }

  // Compute the smallest square dimensions
  u32 pixels = ( ret->size + 3 ) / 4;
  ret->tex.width = (u32)ceilf( sqrtf( (f32)pixels ) );
  ret->tex.height = ( pixels + ret->tex.width - 1 ) / ret->tex.width;

  CHECK_GL_ERROR();
  // Create OpenGL texture
  glGenTextures( 1, &ret->tex.texture );
  glBindTexture( GL_TEXTURE_2D, ret->tex.texture );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, ret->tex.width, ret->tex.height, 0, GL_RGBA, GL_FLOAT, NULL );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

  CHECK_GL_ERROR();
  // Create framebuffer
  glGenFramebuffers( 1, &ret->tex.framebuffer );
  glBindFramebuffer( GL_FRAMEBUFFER, ret->tex.framebuffer );
  glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ret->tex.texture, 0 );

  if( glCheckFramebufferStatus( GL_FRAMEBUFFER ) != GL_FRAMEBUFFER_COMPLETE )
    error( "%s", "Framebuffer is not complete." );


  CHECK_GL_ERROR();
  // Use the compute program to render to the texture
  glViewport( 0, 0, ret->tex.width, ret->tex.height );

  glUseProgram( compute->program );

  glUniform2f( compute->dimsLocation, ret->tex.width, ret->tex.height );
  glUniform4f( compute->stridesLocation, ret->strides[ 0 ], ret->strides[ 1 ],
	       ret->strides[ 2 ], ret->strides[ 3 ] );
  
  // Bind arguments
  for( u32 i = 0; i < compute->argCount; ++i ){
    glActiveTexture( GL_TEXTURE0 + i );
    const tensor* at = ts->stack[ ( ts->size - 1 ) - i ];
    glBindTexture( GL_TEXTURE_2D, at->tex.texture );
    glUniform1i( compute->argTexLocation[ i ], i );
    glUniform2f( compute->argDimsLocation[ i ], at->tex.width, at->tex.height );
    glUniform4f( compute->argStridesLocation[ i ], at->strides[ 0 ], at->strides[ 1 ],
		 at->strides[ 2 ], at->strides[ 3 ] );
    glUniform1f( compute->argToffsetLocation[ i ], at->offset );
  }

  glBindVertexArray( compute->VAO );
  
  CHECK_GL_ERROR();
  glBindBuffer( GL_ARRAY_BUFFER, compute->VBO );
  glBindBuffer( GL_UNIFORM_BUFFER, p->ubo );
  glUniformBlockBinding( compute->program, compute->uboLoc, 0 );
  glBindBufferBase( GL_UNIFORM_BUFFER, 0, p->ubo );

  CHECK_GL_ERROR();
  // Draw the quad
  // Verify Framebuffer Completeness
  GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (status != GL_FRAMEBUFFER_COMPLETE) {
    fprintf(stderr, "Framebuffer incomplete after glDrawArrays: 0x%x\n", status);
    // Handle error (e.g., cleanup and exit)
  }
  glDrawArrays( GL_TRIANGLE_STRIP, 0, 4 );
  // Verify Framebuffer Completeness
  status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (status != GL_FRAMEBUFFER_COMPLETE) {
    fprintf(stderr, "Framebuffer incomplete after glDrawArrays: 0x%x\n", status);
    // Handle error (e.g., cleanup and exit)
  }
  CHECK_GL_ERROR();
  glBindTexture( GL_TEXTURE_2D, 0 );
  glBindBuffer( GL_UNIFORM_BUFFER, 0 );
  glBindFramebuffer( GL_FRAMEBUFFER, 0 );
  glBindVertexArray( 0 );
  glUseProgram( 0 );
  
  CHECK_GL_ERROR();
  // Pop arguments off the stack
  for( u32 i = 0; i < compute->argCount; ++i )
    pop( ts );
  return ret;
}

void push( tensorStack* ts, tensor* t ){
  // Grow stack if necessary. 
  if( ts->size >= ts->allocSize ){
    ts->allocSize *= 2;
    tensor** ns = mem( ts->allocSize, tensor* );
    memcpy( ns, ts->stack, sizeof( tensor* ) * ts->size );
    unmem( ts->stack );
    ts->stack = ns;
  }
  ts->stack[ ts->size++ ] = t;
}


void pop( tensorStack* ts ){
  if( !ts->size )
    error( "%s", "Atempt to pop an empty stack!" );
  deleteTensor( ts->stack[ --ts->size ] );
}

tensorStack* newStack( void ){
  tensorStack* ret = mem( 1, tensorStack );
  ret->allocSize = 256;
  ret->stack = mem( ret->allocSize, tensor );
  ret->size = 0;
  return ret;
}

void deleteStack( tensorStack* ts ){
  for( u32 i = 0; i < ts->size; ++i )
    deleteTensor( ts->stack[ i ] );
  unmem( ts->stack );
  unmem( ts );
}
  
void printStack( tensorStack* ts ){
  for( u32 i = ts->size - 1; i < ts->size; --i ){
    tensor* t = ts->stack[ i ];
    printf( "Tensor %u\n", i );
    printf( "Shape:" );
    for( u32 j = 0; j < t->rank; ++j )
      printf( " %u", t->shape[ j ] );
    printf( "\nStrides:" );
    for( u32 j = 0; j < t->rank; ++j )
      printf( " %i", t->strides[ j ] );
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
    error( "%s", "Invalid tensor or shape." );
  u32 newSize = 1;
  for( u32 i = 0; i < newRank; ++i )
    newSize *= newShape[ i ];
  if( newSize != t->size )
    error( "%s", "New shape size does not match tensor size." );
  
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

void tensorTransposeHelper( tensor* t, u32 axis1, u32 axis2 ){
  if( axis1 >= t->rank || axis2 >= t->rank )
    error( "%s", "Invalid axes in transpose." );
  u32 tmp = t->shape[ axis1 ];
  t->shape[ axis1 ] = t->shape[ axis2 ];
  t->shape[ axis2 ] = tmp;
  tmp = t->strides[ axis1 ];
  t->strides[ axis1 ] = t->strides[ axis2 ];
  t->strides[ axis2 ] = tmp;
}
void tensorTranspose( tensorStack* ts, u32 index, u32 axis1, u32 axis2 ){
  tensorTransposeHelper( ts->stack[ index ], axis1, axis2 );
}

void tensorReverseHelper( tensor* t, u32 axis ){
  if( axis > 3 )
    error( "%s", "Invalid axis in reverse." );
  t->offset += t->strides[ axis ] * ( t->shape[ axis ] - 1 );
  t->strides[ axis ] = -t->strides[ axis ];
}
void tensorReverse( tensorStack* ts, u32 index, u32 axis ){
  tensorReverseHelper( ts->stack[ index ], axis );
}
void tensorCatHelper( tensor* t, tensor* t2, u32 axis ){
  tensorToHostMemory( t ); 
  tensorToHostMemory( t2 );

  if( t->rank != t2->rank )
    error( "Attempt to concatenate tensors of different rank: %u vs %u", t->rank, t2->rank );

  // Check that shapes are compatible except along the concatenation axis
  u32 new_shape[ 4 ];
  for( u32 i = 0; i < t->rank; ++i ){
    if( i == axis ){
      new_shape[ i ] = t->shape[ i ] + t2->shape[ i ];
    } else{
      if( t->shape[ i ] != t2->shape[ i ] )
        error( "Shapes are not compatible for concatenation along axis %u.", axis );
      new_shape[ i ] = t->shape[ i ];
    }
  }

  // Compute new strides
  u32 new_strides[ 4 ];
  u32 size = 1;
  for( int i = t->rank - 1; i >= 0; --i ){
    new_strides[ i ] = size;
    size *= new_shape[ i ];
  }
  size_t total_elements = size;

  // Allocate new data buffer
  f32* new_data = mem( total_elements, f32 );

  // Initialize indices
  u32 indices[ 4 ] = { 0, 0, 0, 0 };
  u32 indices_t2[ 4 ];

  // Iterate over all elements
  for( size_t count = 0; count < total_elements; ++count ){

    f32 val;

    if( indices[ axis ] < t->shape[ axis ] ){
      // Get data from t
      size_t src_idx = t->offset;
      for( u32 i = 0; i < t->rank; ++i )
        src_idx += indices[ i ] * t->strides[ i ];
      val = t->data[ src_idx ];
    } else{
      // Get data from t2
      for( u32 i = 0; i < t->rank; ++i )
        indices_t2[ i ] = indices[ i ];
      indices_t2[ axis ] -= t->shape[ axis ];
      size_t src_idx = t2->offset;
      for( u32 i = 0; i < t2->rank; ++i )
        src_idx += indices_t2[ i ] * t2->strides[ i ];
      val = t2->data[ src_idx ];
    }

    // Compute dest_idx
    size_t dest_idx = 0;
    for( u32 i = 0; i < t->rank; ++i )
      dest_idx += indices[ i ] * new_strides[ i ];

    new_data[ dest_idx ] = val;

    // Increment indices like in nested loops
    for( int i = t->rank - 1; i >= 0; --i ){
      indices[ i ]++;
      if( indices[ i ] < new_shape[ i ] )
        break;
      else
        indices[ i ] = 0;
    }
  }

  // Free old data if owned
  if( t->ownsData ){
    unmem( t->data );
  }

  t->data = new_data;
  t->ownsData = true;

  // Update shape and strides
  for( u32 i = 0; i < t->rank; ++i ){
    t->shape[ i ] = new_shape[ i ];
    t->strides[ i ] = new_strides[ i ];
  }

  t->size = total_elements;
}
void tensorCat( tensorStack* ts, u32 index1, u32 index2, u32 axis ){
  tensorCatHelper( ts->stack[ index1 ], ts->stack[ index2 ], axis );
}
