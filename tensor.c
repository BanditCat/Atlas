////////////////////////////////////////////////////////////////////////////////
// Copyright © 2025 Jon DuBois. Written with the assistance of GPT-4 et al.   //
////////////////////////////////////////////////////////////////////////////////

#include "Atlas.h"
#include "tensorGltf.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "miniz.h"

void takeOwnership( tensor* t ){
  if( t == NULL )
    error( "%s", "Tensor is NULL in takeOwnership." );

  if( t->ownsData )
    return;  // Already owns data, nothing to do

  if( t->gpu ){
    tensorToHostMemory( t );
    tensorToGPUMemory( t );		    
  }else{
    tensorEnsureContiguous( t );
    
    if( t->ownsData )
      return;  // Already owns data, nothing to do
    
    // Allocate new memory for the data
    f32* newData = mem( t->size, f32 );
    // Copy the data from the original tensor, considering the offset
    memcpy( newData, t->data + t->offset, t->size * sizeof( f32 ) );
    // Reset the offset since data is now at the beginning
    t->offset = 0;
    // Update the data pointer to the new data
    t->data = newData;
    // Mark that the tensor now owns the data
    t->ownsData = true;
  }
};
// DANGER this sets owns data to false, therefore the underlying data MUST NOT
// be destroyed BEFORE the copy while the programming is running. At exit
// cleanup, it shouldn't matter the order of deallocation.
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

  // hostData is allocated for the EXACT tensor size
  f32* hostData = mem( t->size, f32 );

  u64 mult = t->tex.channels;
  if( mult >= 10 )
    mult /= 10;
  if( !mult ) mult = 4; // Generic storage is RGBA (4 floats)
  
  // layerElementCount is the TEXTURE size (which may include padding)
  u32 layerElementCount = t->tex.width * t->tex.height * mult;
  f32* tempData =  mem( layerElementCount, f32 ); 

  CHECK_GL_ERROR();
  glBindFramebuffer( GL_FRAMEBUFFER, t->tex.framebuffer );
  glFramebufferTextureLayer( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, t->tex.texture, 0, 0 );
  CHECK_GL_ERROR();
  
  GLenum format = GL_RGBA;
  if( t->tex.channels == 1 || t->tex.channels == 10 ) format = GL_RED;
  if( t->tex.channels == 2 || t->tex.channels == 20 ) format = GL_RG;
  if( t->tex.channels == 3 || t->tex.channels == 30 ) format = GL_RGB;

  for( u32 i = 0; i < t->tex.layers; ++i ){
    glFramebufferTextureLayer( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, t->tex.texture, 0, i );
    glReadPixels( 0, 0, t->tex.width, t->tex.height, format, GL_FLOAT, tempData );
    
    u64 offset = (u64)i * layerElementCount;
    u64 elementsToCopy = layerElementCount;

    if( offset + elementsToCopy > t->size ){
      if( offset >= t->size ) {
        elementsToCopy = 0;
      } else {
        elementsToCopy = t->size - offset;
      }
    }

    memcpy( hostData + offset, tempData, elementsToCopy * sizeof( f32 ) );
  }
  CHECK_GL_ERROR();

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

  t->offset = 0;
  t->data = hostData;
  t->gpu = false;
  t->ownsData = true;
  u32 stride = 1;
  for( int i = t->rank - 1; i >= 0; --i ){
    t->strides[ i ] = stride;
    stride *= t->shape[ i ];
  }
}
void tensorToGPUMemory( tensor* t ){
  if( t == NULL )
    error( "%s", "Tensor is NULL in tensorToGPUMemory." );
  if( t->gpu )
    return;
  tensorEnsureContiguous( t );

  //  texture dimensions for GPU storage
  f32* tdata = t->data;
  u32 pixels = ( t->size + 3 ) / 4;  // RGBA = 4 floats per pixel
  u32 twidth = ceilf( sqrtf( (f32)pixels ) );
  u32 theight = ( pixels + twidth - 1 ) / twidth;

  // Prepare padded data for texture upload
  f32* paddedData = mem( twidth * theight * 4, f32 );
  memcpy( paddedData, tdata + t->offset, t->size * sizeof( f32 ) );

  t->offset = 0;
  t->tex.width = twidth;
  t->tex.height = theight;
  t->tex.layers = 1;
  t->tex.channels = 0;
  
  
  glGenTextures( 1, &t->tex.texture );
  glBindTexture( GL_TEXTURE_2D_ARRAY, t->tex.texture );
  glTexImage3D( GL_TEXTURE_2D_ARRAY,
                0,
                GL_RGBA32F,
                t->tex.width,
                t->tex.height,
                1,
                0,
                GL_RGBA,
                GL_FLOAT,
                NULL );
  glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
  /* glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT ); */
  /* glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT ); */

  glGenFramebuffers( 1, &t->tex.framebuffer );
  glBindFramebuffer( GL_FRAMEBUFFER, t->tex.framebuffer );
  glFramebufferTextureLayer( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, t->tex.texture, 0, 0 );

  if( glCheckFramebufferStatus( GL_FRAMEBUFFER ) != GL_FRAMEBUFFER_COMPLETE )
    error( "%s", "Framebuffer is not complete." );

  glBindFramebuffer( GL_FRAMEBUFFER, 0 );

  glBindTexture( GL_TEXTURE_2D_ARRAY, t->tex.texture );
  glTexSubImage3D( GL_TEXTURE_2D_ARRAY, 0,
                   0,
                   0,
                   0,
                   t->tex.width,
                   t->tex.height,
                   1,
                   GL_RGBA,
                   GL_FLOAT,
                   paddedData );
  glBindTexture( GL_TEXTURE_2D_ARRAY, 0 );

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
      if( t->tex.depthbuffer ){
        glDeleteRenderbuffers( 1, &t->tex.depthbuffer );
        t->tex.depthbuffer = 0;
      }
      if( t->tex.framebuffer ){
        glDeleteFramebuffers( 1, &t->tex.framebuffer );
        t->tex.framebuffer = 0;
      }
    } else {
      unmem( t->data );
    }
  }
  unmem( t );
}
compute* makeCompute( const char* filename,
                      u32 linenum,
                      u32 commandnum,
                      const program* prog,
                      const char* uniforms,
                      const char* vglslpre,
                      const char* glslpre,
                      const char* vglsl,
                      const char* glsl,
                      u32 argCount,
                      u32 retCount,
                      u32 channels,
                      bool reuse ){
  const char* channelsString;
  switch( channels ){
  case 0:
  case 4: case 40: channelsString = "vec4"; break;
  case 1: case 10: channelsString = "float"; break;
  case 2: case 20: channelsString = "vec2"; break;
  case 3: case 30: channelsString = "vec3"; break;
  }
  // Vertex shader source (simple pass-through)
  compute* ret = mem( 1, compute );
  ret->argCount = argCount;
  ret->retCount = retCount;
  ret->channels = channels;
  ret->reuse = reuse;
  const char* vertexShaderTemplate = "\
    #version 300 es\n\
    precision highp float;\n\
    precision highp int;\n\
    precision highp sampler2DArray;\n\
    in vec2 _a_position;\n\
    %s\n\
    %s\n\
    %s\n\
    const vec4 _a_corners[ 6 ] = vec4[](\n\
      vec4( -1.0, -1.0, 1.0, 1.0),\n\
      vec4(  1.0, -1.0, 1.0, 1.0),\n\
      vec4( -1.0,  1.0, 1.0, 1.0),\n\
      vec4( -1.0,  1.0, 1.0, 1.0),\n\
      vec4(  1.0, -1.0, 1.0, 1.0),\n\
      vec4(  1.0,  1.0, 1.0, 1.0)\n\
    );\n\
    void main(){\n\
      vec4 ret;\n\
      int i = gl_VertexID;\n\
      float ifloat = float( i ) + 0.5;\n\
      %s\n\
      gl_Position = ret;\n\
    }\n\
  ";

  const char* defvshader = "ret = _a_corners[ gl_VertexID ];";
  
  const char* texFunctions = "\
    uniform ivec4 _a_astrides;\n\
    uniform int _a_atoffset;\n\
    uniform ivec2 _a_adims;\n\
    uniform sampler2DArray _a_atex;\n\
    float a( ivec4 i ){\n\
      vec2 a_adims = vec2( _a_adims );\n\
      int lindex = _a_atoffset;\n\
      for( int j = 0; j < 4; ++j )\n\
        lindex += _a_astrides[ j ] * int( i[ j ] );\n\
      int pixel_index = lindex / 4;\n\
      int channel = lindex % 4;\n\
      vec2 uv = ( vec2( pixel_index % _a_adims.x, pixel_index / _a_adims.x ) + 0.5 ) / a_adims;\n\
      vec4 texel = textureLod( _a_atex, vec3( uv, 0.0 ), 0.0 );\n\
      return texel[ int( channel ) ];\n\
    }\n\
    vec4 af( vec2 uv ){\n\
      return texture( _a_atex, vec3( uv / vec2( _a_adims ), 0.0 ) );\n\
    }\n\
    vec4 af( vec3 uv ){\n\
      return texture( _a_atex, vec3( uv.xy / vec2( _a_adims ), uv.z ) );\n\
    }\n\
    vec4 af( vec4 uv ){\n\
      return textureLod( _a_atex, vec3( uv.xy / vec2( _a_adims ), uv.z ), uv.w );\n\
    }\n\
    uniform ivec4 _a_bstrides;\n\
    uniform int _a_btoffset;\n\
    uniform ivec2 _a_bdims;\n\
    uniform sampler2DArray _a_btex;\n\
    float b( ivec4 i ){\n\
      vec2 a_adims = vec2( _a_bdims );\n\
      int lindex = _a_btoffset;\n\
      for( int j = 0; j < 4; ++j )\n\
        lindex += _a_bstrides[ j ] * int( i[ j ] );\n\
      int pixel_index = lindex / 4;\n\
      int channel = lindex % 4;\n\
      vec2 uv = ( vec2( pixel_index % _a_bdims.x, pixel_index / _a_bdims.x ) + 0.5 ) / a_adims;\n\
      vec4 texel = textureLod( _a_btex, vec3( uv, 0.0 ), 0.0 );\n\
      return texel[ int( channel ) ];\n\
    }\n\
    vec4 bf( vec2 uv ){\n\
      return texture( _a_btex, vec3( uv / vec2( _a_bdims ), 0.0 ) );\n\
    }\n\
    vec4 bf( vec3 uv ){\n\
      return texture( _a_btex, vec3( uv.xy / vec2( _a_bdims ), uv.z ) );\n\
    }\n\
    vec4 bf( vec4 uv ){\n\
      return textureLod( _a_btex, vec3( uv.xy / vec2( _a_bdims ), uv.z ), uv.w );\n\
    }\n\
    uniform ivec4 _a_cstrides;\n\
    uniform int _a_ctoffset;\n\
    uniform ivec2 _a_cdims;\n\
    uniform sampler2DArray _a_ctex;\n\
    float c( ivec4 i ){\n\
      vec2 a_adims = vec2( _a_cdims );\n\
      int lindex = _a_ctoffset;\n\
      for( int j = 0; j < 4; ++j )\n\
        lindex += _a_cstrides[ j ] * int( i[ j ] );\n\
      int pixel_index = lindex / 4;\n\
      int channel = lindex % 4;\n\
      vec2 uv = ( vec2( pixel_index % _a_cdims.x, pixel_index / _a_cdims.x ) + 0.5 ) / a_adims;\n\
      vec4 texel = textureLod( _a_ctex, vec3( uv, 0.0 ), 0.0 );\n\
      return texel[ int( channel ) ];\n\
    }\n\
    vec4 cf( vec2 uv ){\n\
      return texture( _a_ctex, vec3( uv / vec2( _a_cdims ), 0.0 ) );\n\
    }\n\
    vec4 cf( vec3 uv ){\n\
      return texture( _a_ctex, vec3( uv.xy / vec2( _a_cdims ), uv.z ) );\n\
    }\n\
    vec4 cf( vec4 uv ){\n\
      return textureLod( _a_ctex, vec3( uv.xy / vec2( _a_cdims ), uv.z ), uv.w );\n\
    }\n\
    uniform ivec4 _a_dstrides;\n\
    uniform int _a_dtoffset;\n\
    uniform ivec2 _a_ddims;\n\
    uniform sampler2DArray _a_dtex;\n\
    float d( ivec4 i ){\n\
      vec2 a_adims = vec2( _a_ddims );\n\
      int lindex = _a_dtoffset;\n\
      for( int j = 0; j < 4; ++j )\n\
        lindex += _a_dstrides[ j ] * int( i[ j ] );\n\
      int pixel_index = lindex / 4;\n\
      int channel = lindex % 4;\n\
      vec2 uv = ( vec2( pixel_index % _a_ddims.x, pixel_index / _a_ddims.x ) + 0.5 ) / a_adims;\n\
      vec4 texel = textureLod( _a_dtex, vec3( uv, 0.0 ), 0.0 );\n\
      return texel[ int( channel ) ];\n\
    }\n\
    vec4 df( vec2 uv ){\n\
      return texture( _a_dtex, vec3( uv / vec2( _a_ddims ), 0.0 ) );\n\
    }\n\
    vec4 df( vec3 uv ){\n\
      return texture( _a_dtex, vec3( uv.xy / vec2( _a_ddims ), uv.z ) );\n\
    }\n\
    vec4 df( vec4 uv ){\n\
      return textureLod( _a_dtex, vec3( uv.xy / vec2( _a_ddims ), uv.z ), uv.w );\n\
    }\n";
  
  // Fragment shader template
  const char* fragmentShaderTemplate = "\
    #version 300 es\n\
    precision highp float;\n\
    precision highp int;\n\
    precision highp sampler2DArray;\n\
    layout(location = 0) out %s _a_fragColor[ %u ];\n\
    uniform ivec2 _a_dims; // Texture dimensions\n\
    uniform ivec4 _a_strides; // Tensor shape\n\
    \n\
    %s\n\
    %s\n\
    ivec4 _a_toTensorIndices( int i ){\n\
      ivec4 ret;\n\
      ret.x = i / _a_strides.x;\n\
      i -= ret.x * _a_strides.x;\n\
      ret.y = i / _a_strides.y;\n\
      i -= ret.y * _a_strides.y;\n\
      ret.z = i / _a_strides.z;\n\
      i -= ret.z * _a_strides.z;\n\
      ret.w = i;\n\
      return ret;\n\
    }\n\
    %s\n\
    %s\n";
  char* tensorFooterTemplate = 
    "void main(){\n\
      int i = ( int( gl_FragCoord.x ) + int( gl_FragCoord.y ) * _a_dims.x ) * 4;\n\
      float ifloat = float( i ) + 0.5;\n\
      ivec4 t = _a_toTensorIndices( i );\n\
      vec4 tf = vec4( t ) + 0.5;\n\
      float ret[ %u ];\n\
      float _a_r[ %u ];\n\
      float _a_g[ %u ];\n\
      float _a_b[ %u ];\n\
      float _a_a[ %u ];\n\
      {%s}\n\
      for( int j = 0; j < %u; ++j ) _a_r[ j ] = ret[ j ];\n\
      ++i; t = _a_toTensorIndices( i ); ++ifloat; tf = vec4( t ) + 0.5;\n\
      {%s}\n\
      for( int j = 0; j < %u; ++j ) _a_g[ j ] = ret[ j ];\n\
      ++i; t = _a_toTensorIndices( i ); ++ifloat; tf = vec4( t ) + 0.5;\n\
      {%s}\n\
      for( int j = 0; j < %u; ++j ) _a_b[ j ] = ret[ j ];\n\
      ++i; t = _a_toTensorIndices( i ); ++ifloat; tf = vec4( t ) + 0.5;\n\
      {%s}\n\
      for( int j = 0; j < %u; ++j ) _a_a[ j ] = ret[ j ];\n";
  char* textureFooterTemplate =
    "void main(){\n\
     %s ret[ %u ];\n\
     vec2 tf = gl_FragCoord.xy;\n\
     float ifloat = floor( tf.y ) * float( _a_dims.x ) + floor( tf.x ) + 0.5;\n\
     int i = int( ifloat );\n\
     {%s}\n";
   
  // Buffer to hold the final fragment shader source
  u32 bufsize = 1048576;
  char* fragmentShaderSource = mem( bufsize, char );
  char* footerSource = mem( bufsize, char );
  u32 headerLineCount = 0;
  u32 headerIncPreambleLineCount = 0;
  int flen;
  if( channels == 0 ){
    flen = snprintf( footerSource, bufsize, tensorFooterTemplate, retCount,
                     retCount, retCount, retCount, retCount, glsl, retCount,
                     glsl, retCount, glsl, retCount, glsl, retCount, retCount );
    headerIncPreambleLineCount += newlines( tensorFooterTemplate );
  } else{
    flen = snprintf( footerSource, bufsize, textureFooterTemplate, channelsString, retCount, glsl );
    headerIncPreambleLineCount += newlines( textureFooterTemplate );
    headerIncPreambleLineCount += newlines( channelsString );
  }
  int len = snprintf( fragmentShaderSource, bufsize, fragmentShaderTemplate, channelsString,
                      retCount, texFunctions, uniforms, glslpre, footerSource );
  headerLineCount += newlines( fragmentShaderTemplate );
  headerLineCount += newlines( channelsString );
  headerLineCount += newlines( texFunctions );
  headerLineCount += newlines( uniforms );
  --headerLineCount;
  headerIncPreambleLineCount += headerLineCount;
  headerIncPreambleLineCount += newlines( glslpre );
  u32 smallbufsize = 65536;
  if( len < 0 || len >= bufsize - smallbufsize || flen < 0 || flen >= bufsize )
    error( "%s", "Shader source exceeds buffer size." );
  if( channels == 0 ){
    for( u32 i = 0; i < retCount; ++i ){
      char* smallbuf = mem( smallbufsize, char );
      snprintf( smallbuf,
                smallbufsize,
                "    _a_fragColor[ %u ] = vec4( _a_r[ %u ], _a_g[ %u ], _a_b[ %u "
                "], _a_a[ %u ] );\n", i, i, i, i, i );
      strncat( fragmentShaderSource, smallbuf, 1000 );
      unmem( smallbuf );
    }
    strncat( fragmentShaderSource, "}", 1000 );
  } else{
    for( u32 i = 0; i < retCount; ++i ){
      char* smallbuf = mem( smallbufsize, char );
      snprintf( smallbuf,
                smallbufsize,
                "    _a_fragColor[ %u ] = ret[ %u ];\n", i, i );
      strncat( fragmentShaderSource, smallbuf, 1000 );
      unmem( smallbuf );
    }
    strncat( fragmentShaderSource, "}", 1000 );
    //dbg( "%s", fragmentShaderSource );
  }
  //  Compile the vertex shader
  char* vertexShaderSource = mem( bufsize, char );
  len = snprintf( vertexShaderSource, bufsize, vertexShaderTemplate, uniforms, texFunctions, vglslpre,
                  strlen( vglsl ) ? vglsl : defvshader );
  u32 vheaderLineCount = 0;
  u32 vheaderIncPreambleLineCount = 0;
  vheaderLineCount += newlines( uniforms );
  vheaderLineCount += newlines( texFunctions );
  vheaderLineCount += 9;  
  vheaderIncPreambleLineCount += vheaderLineCount;
  vheaderIncPreambleLineCount += newlines( vglslpre );
  vheaderIncPreambleLineCount += 12;
  if( len < 0 || len >= bufsize - smallbufsize || flen < 0 || flen >= bufsize )
    error( "%s", "Shader source exceeds buffer size." );
  
  GLuint vertexShader = glCreateShader( GL_VERTEX_SHADER );
  const char* p = vertexShaderSource;
  glShaderSource( vertexShader, 1, &p, NULL );
  glCompileShader( vertexShader );

  // Check for vertex shader compilation errors
  GLint status;
  glGetShaderiv( vertexShader, GL_COMPILE_STATUS, &status );
  if( status != GL_TRUE ){
    static const u32 bufsize = 65536;
    char* msg = mem( bufsize, char );
    char* log = mem( bufsize, char );
    glGetShaderInfoLog( vertexShader, bufsize, NULL, log );
    snprintf( msg, bufsize, "%s:%u command %u:\nVertex shader compilation failed, error line numbers offset by %u for preamble and %u for main body:\n\n %s", filename, linenum, commandnum, vheaderLineCount, vheaderIncPreambleLineCount, log );
    glDeleteShader( vertexShader );
    error( "%s", msg );
  }

  // Compile the fragment shader
  GLuint fragmentShader = glCreateShader( GL_FRAGMENT_SHADER );
  p = fragmentShaderSource;
  glShaderSource( fragmentShader, 1, &p, NULL );
  glCompileShader( fragmentShader );
  

  // Check for fragment shader compilation errors
  glGetShaderiv( fragmentShader, GL_COMPILE_STATUS, &status );
  if( status != GL_TRUE ){
    static const u32 bufsize = 65536;
    char* msg = mem( bufsize, char );
    char* log = mem( bufsize, char );
    glGetShaderInfoLog( fragmentShader, bufsize, NULL, log );
    snprintf( msg, bufsize, "%s:%u command %u:\nFragment shader compilation failed, error line numbers offset by %u for preamble and %u for main body:\n\n %s", filename, linenum, commandnum, headerLineCount, headerIncPreambleLineCount, log );
    glDeleteShader( fragmentShader );
    glDeleteShader( vertexShader );
    error( "%s", msg );
  }

  unmem( vertexShaderSource );
  unmem( fragmentShaderSource );
  unmem( footerSource );

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

  // Get uniforms locations from program.
  ret->uniformLocs = mem( prog->numVars, GLuint );
  for( u32 i = 0; i < prog->numVars; ++i ){
    u32 varlen = strlen( prog->varNames[ i ] );
    char* safeName = mem( varlen + 1, char );
    memcpy( safeName, prog->varNames[ i ], varlen + 1 );
    for( u32 i = 0; i < varlen; ++i )
      if( safeName[ i ] == '.' )
        safeName[ i ] = '_';
    int rv = glGetUniformLocation( ret->program, safeName );
    ret->uniformLocs[ i ] = rv;
    //    if( rv == -1 )
    //      error( "Error getting uniform location %s %s!", safeName, uniforms );
    unmem( safeName );
  }
  // Cleanup shaders (they're no longer needed once the program is linked)
  glDeleteShader( vertexShader );
  glDeleteShader( fragmentShader );

  return ret;
}
void deleteCompute( compute* i ){
  glDeleteProgram( i->program );
  unmem( i->uniformLocs );
  unmem( i );
}
tensor** newTensorsInitialized( program* p, tensorStack* ts, u32 rank, u32* shape, const compute* compute, u32 vertCount ){
  CHECK_GL_ERROR();
  glUseProgram( compute->program );
  if( compute->reuse ){
    if( compute->argCount + compute->retCount > ts->size )
      error( "A compute was called with %u arguments and %u returns, but the stack size is only %u.",
             compute->argCount, compute->retCount, ts->size );
  } else
    if( compute->argCount > ts->size )
      error( "A compute was called with %u arguments, but the stack size is only %u.",
             compute->argCount,
             ts->size );
  tensor** rets = mem( compute->retCount, tensor* );
  tensor* ret;
  for( u32 i = 0; i < compute->argCount; ++i ){
    tensor* cur = ts->stack[ ( ts->size - 1 ) - i ];
    tensorToGPUMemory( cur );
    // Set filtering here so its per tensor not per gl texture. NOTE THAT YES ITS BUGBUG IF YOU HAVE A VIEW both
    // mipmapped and nonmipmapped, so dont do that.
    glBindTexture( GL_TEXTURE_2D_ARRAY, cur->tex.texture );
    if( cur->tex.mipmapped ){
      glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR );
      glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR  );
      glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT );
      glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT );
      if( getMaxAnisotropy() > 1.0 ){
        glTexParameterf(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAX_ANISOTROPY_EXT, getMaxAnisotropy() );
      }
    } else {
      glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
      glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
      glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
      glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
      glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE );
    }
    glBindTexture( GL_TEXTURE_2D_ARRAY, 0 );
  }
  u32 size = 1;
  for( u32 i = 0; i < rank; ++i )
    size *= shape[ i ];
  // Compute the smallest square dimensions BUGBUG TODO move frambuffer into
  // compute and out of tex
  u32 pixels = ( size + 3 ) / 4;
  u32 width = (u32)ceilf( sqrtf( (f32)pixels ) );
  u32 height = ( pixels + width - 1 ) / width;
  if( compute->channels != 0 ){
    width = shape[ 0 ];
    height = shape[ 1 ];
    size = width * height * shape[ 2 ];
  }
  for( u32 reti = 0; reti < compute->retCount; ++reti ){
    if( compute->reuse ){
      tensor* t = ts->stack[ ( ( ts->size - 1 ) - compute->argCount ) - reti ];
      if( !t->gpu || ( t->tex.channels != compute->channels ) )
        error( "%s", "Attempt to return on top of a incompatible tensor (wrong channel count)." );
      if( t->tex.width != width || t->tex.height != height )
        error( "%s", "Attempt to return on top of a incompatible tensor (bad size)." );
      ret = t;
    } else {
      u32 found = TENSOR_CACHE;
      for( u32 i = 0; i < TENSOR_CACHE; ++i )
        if( ts->cache[ i ] && size == ts->cache[ i ]->size && ts->cache[ i ]->tex.channels == compute->channels ){
          found = i;
          break;
        }
      if( found != TENSOR_CACHE ){
        ret = ts->cache[ found ];
        glBindTexture( GL_TEXTURE_2D_ARRAY, ret->tex.texture );
        glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
        glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
        glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE );
        glBindTexture( GL_TEXTURE_2D_ARRAY, 0 );
        ret->tex.mipmapped = false;
        ret->tex.channels = compute->channels;
        ts->cache[ found ] = NULL;
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
      } else {
        ret = mem( 1, tensor );
        ret->tex.channels = compute->channels;
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
        ret->tex.width = width;
        ret->tex.height = height;
        ret->tex.layers = 1;

        CHECK_GL_ERROR();
        // Create OpenGL texture
        glGenTextures( 1, &ret->tex.texture );
        glBindTexture( GL_TEXTURE_2D_ARRAY, ret->tex.texture );
        switch( compute->channels ){
        case 40:
          glTexImage3D( GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, width, height, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL );
          break;
        case 0:
        case 4:
          glTexImage3D( GL_TEXTURE_2D_ARRAY, 0, GL_RGBA32F, width, height, 1, 0, GL_RGBA, GL_FLOAT, NULL );
          break;
        case 10:
          glTexImage3D( GL_TEXTURE_2D_ARRAY, 0, GL_R8, width, height, 1, 0, GL_RED, GL_UNSIGNED_BYTE, NULL );
          break;
        case 1:
          glTexImage3D( GL_TEXTURE_2D_ARRAY, 0, GL_R32F, width, height, 1, 0, GL_RED, GL_FLOAT, NULL );
          break;
        case 2:
          glTexImage3D( GL_TEXTURE_2D_ARRAY, 0, GL_RG32F, width, height, 1, 0, GL_RG, GL_FLOAT, NULL );
          break;
        case 3:
          glTexImage3D( GL_TEXTURE_2D_ARRAY, 0, GL_RGB32F, width, height, 1, 0, GL_RGB, GL_FLOAT, NULL );
          break;
        }       
        glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
        glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
        glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE );
        /* glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT ); */
        /* glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT ); */
        
        CHECK_GL_ERROR();
        // Create framebuffer
        glGenFramebuffers( 1, &ret->tex.framebuffer );
      }
    }
    rets[ reti ] = ret;
  }
  glBindFramebuffer( GL_FRAMEBUFFER, rets[ 0 ]->tex.framebuffer );
  for( u32 i = 0; i < compute->retCount; ++i )
    glFramebufferTextureLayer( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, rets[ i ]->tex.texture, 0, 0 );

  


  CHECK_GL_ERROR();
  glViewport( 0, 0, width, height );

  glUniform2i( compute->dimsLocation, width, height );
  glUniform4i( compute->stridesLocation,
               ret->strides[ 0 ],
               ret->strides[ 1 ],
               ret->strides[ 2 ],
               ret->strides[ 3 ] );

  // Bind arguments
  for( u32 i = 0; i < compute->argCount; ++i ){
    glActiveTexture( GL_TEXTURE0 + i );
    const tensor* at = ts->stack[ ( ts->size - 1 ) - i ];
    glBindTexture( GL_TEXTURE_2D_ARRAY, at->tex.texture );
    glUniform1i( compute->argTexLocation[ i ], i );
    glUniform2i( compute->argDimsLocation[ i ], at->tex.width, at->tex.height );
    glUniform4i( compute->argStridesLocation[ i ],
                 at->strides[ 0 ],
                 at->strides[ 1 ],
                 at->strides[ 2 ],
                 at->strides[ 3 ] );
    glUniform1i( compute->argToffsetLocation[ i ], at->offset );
  }

  CHECK_GL_ERROR();
  // glBindBuffer( GL_UNIFORM_BUFFER, p->ubo );
  // glUniformBlockBinding( compute->program, compute->uboLoc, 0 );
  // glBindBufferBase( GL_UNIFORM_BUFFER, 0, p->ubo );

  GLenum drawBuffers[ 4 ] = { GL_COLOR_ATTACHMENT0,
                              GL_COLOR_ATTACHMENT1,
                              GL_COLOR_ATTACHMENT2,
                              GL_COLOR_ATTACHMENT3 };
  glDrawBuffers( compute->retCount, drawBuffers );

  if( depthTest ){
    if( !ret->tex.depthbuffer )
      glGenRenderbuffers( 1, &ret->tex.depthbuffer );
    glBindRenderbuffer( GL_RENDERBUFFER, ret->tex.depthbuffer );
    if( !compute->reuse )
      glRenderbufferStorage( GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height );
    glFramebufferRenderbuffer( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, ret->tex.depthbuffer );
#ifdef __EMSCRIPTEN__   // I dont know why it only works this way, but it only works this way.
    glDepthFunc( GL_LEQUAL );
    glClearDepthf( 1.0f ); 
    glDepthRangef( 0.0, 1.0 );
#else
    glDepthFunc( GL_GEQUAL );
    glClearDepth( 0.0f ); 
    glDepthRange( 1.0, 0.0 );
#endif
    glViewport( 0, 0, width, height );
    glDepthMask( GL_TRUE );
    if( !compute->reuse )
      glClear( GL_DEPTH_BUFFER_BIT );
    glEnable( GL_DEPTH_TEST );
  } else
    glDisable( GL_DEPTH_TEST );

  if( additive ){
    glEnable( GL_BLEND );
    glBlendFunc( 1.0, 1.0 );
  } else
    glDisable( GL_BLEND );
  
  CHECK_GL_ERROR();
  // Draw the quad
  if( !compute->reuse )
    glClear( GL_COLOR_BUFFER_BIT );
  glDrawArrays( GL_TRIANGLES, 0, vertCount );
  CHECK_GL_ERROR();
  
  for( u32 i = 0; i < compute->retCount; ++i )
    glFramebufferTextureLayer( GL_FRAMEBUFFER,
                               GL_COLOR_ATTACHMENT0 + i,
                               0,
                               0,
                               0 );
  //  glBindTexture( GL_TEXTURE_2D, 0 );
  // glBindBuffer( GL_UNIFORM_BUFFER, 0 );
  // glBindBufferBase( GL_UNIFORM_BUFFER, 0, 0 );
  glBindFramebuffer( GL_FRAMEBUFFER, 0 );
  //glBindVertexArray( 0 );
  // glUseProgram( 0 );

  CHECK_GL_ERROR();
  // Pop arguments off the stack
  for( u32 i = 0; i < compute->argCount; ++i )
    pop( ts );
#ifndef EMSCRIPTEN // only need this for native afaict
  glFinish();
#endif

  if( compute->reuse ){
    unmem( rets );
    return NULL;
  } else
    return rets;
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
    error( "%s", "Attempt to pop an empty stack!" );
  --ts->size;
  if( !ts->stack[ ts->size ]->gpu || !ts->stack[ ts->size ]->ownsData )
    deleteTensor( ts->stack[ ts->size ] );
  else {
    if( ts->cache[ TENSOR_CACHE - 1 ] )
      deleteTensor( ts->cache[ TENSOR_CACHE - 1 ] );
    for( int i = TENSOR_CACHE - 1; i > 0; --i )
      ts->cache[ i ] = ts->cache[ i - 1 ];
    ts->cache[ 0 ] = ts->stack[ ts->size ];
    ts->stack[ ts->size ] = NULL;
  }
}

tensorStack* newStack( void ){
  tensorStack* ret = mem( 1, tensorStack );
  ret->allocSize = 256;
  ret->stack = mem( ret->allocSize, tensor* );
  ret->size = 0;
  return ret;
}

void deleteStack( tensorStack* ts ){
  for( u32 i = 0; i < ts->size; ++i )
    deleteTensor( ts->stack[ i ] );
  for( u32 i = 0; i < TENSOR_CACHE; ++i )
    if( ts->cache[ i ] )
      deleteTensor( ts->cache[ i ] );
  unmem( ts->stack );
  unmem( ts );
}

void printStack( tensorStack* ts ){
  for( u32 i = ts->size - 1; i < ts->size; --i ){
    tensor* t = ts->stack[ i ];
    print( "tensor %u\n", i );
    print( "shape:" );
    for( u32 j = 0; j < t->rank; ++j )
      print( " %u", t->shape[ j ] );
    print( "\nstrides:" );
    for( u32 j = 0; j < t->rank; ++j )
      print( " %i", t->strides[ j ] );
    if( t->size < MAX_TENSOR_DIPLAY_SIZE ){
      char* fd = formatTensorData( t );
      print( "\n%s\n\n", fd );
      unmem( fd );
    } else {
      if( t->tex.channels )
        print( "\n[large texture]\n\n" );        
      else
        print( "\n[large tensor]\n\n" );
    }
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

  // Ensure both tensors own their data
  takeOwnership( t );
  takeOwnership( t2 );

  // Check that shapes are compatible except along the concatenation axis
  u32 new_shape[ 4 ];
  for( u32 i = 0; i < t->rank; ++i ){
    if( i == axis ){
      new_shape[ i ] = t->shape[ i ] + t2->shape[ i ];
    } else {
      if( t->shape[ i ] != t2->shape[ i ] )
        error( "Shapes are not compatible for concatenation along axis %u.",
               axis );
      new_shape[ i ] = t->shape[ i ];
    }
  }

  // Compute new strides for standard contiguous layout
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

  // Iterate over all elements
  for( size_t count = 0; count < total_elements; ++count ){

    f32 val;
    size_t dest_idx = 0;
    for( u32 i = 0; i < t->rank; ++i )
      dest_idx += indices[ i ] * new_strides[ i ];

    if( indices[ axis ] < t->shape[ axis ] ){
      // Get data from t
      size_t src_idx = t->offset;
      for( u32 i = 0; i < t->rank; ++i )
        src_idx += indices[ i ] * t->strides[ i ];
      val = t->data[ src_idx ];
    } else {
      // Get data from t2
      u32 idx_in_t2[ 4 ];
      for( u32 i = 0; i < t->rank; ++i ){
        if( i == axis )
          idx_in_t2[ i ] = indices[ i ] - t->shape[ i ];
        else
          idx_in_t2[ i ] = indices[ i ];
      }
      size_t src_idx = t2->offset;
      for( u32 i = 0; i < t2->rank; ++i )
        src_idx += idx_in_t2[ i ] * t2->strides[ i ];
      val = t2->data[ src_idx ];
    }

    new_data[ dest_idx ] = val;

    // Increment indices
    for( int i = t->rank - 1; i >= 0; --i ){
      indices[ i ]++;
      if( indices[ i ] < new_shape[ i ] )
        break;
      else
        indices[ i ] = 0;
    }
  }

  // Free old data
  if( t->ownsData ){
    unmem( t->data );
  }

  // Update tensor t to be the concatenated tensor
  t->data = new_data;
  t->ownsData = true;
  t->offset = 0;

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
// Returns a fresh tensor which must later be deallocated.
tensor*  tensorMultiplyHelper( tensor* t1, tensor* t2 ){
  tensorToHostMemory( t1 );
  tensorToHostMemory( t2 );
  
  tensor* ret = mem( 1, tensor );
  ret->rank = 2;
  ret->shape[ 2 ] = ret->shape[ 3 ] = 1;
  ret->shape[ 0 ] = t2->shape[ 0 ];
  ret->shape[ 1 ] = t1->shape[ 1 ];
  ret->size = ret->shape[ 0 ] * ret->shape[ 1 ];
  for( u32 i = 0; i < 4; ++i )
    ret->strides[ i ] = i == ret->rank - 1 ? 1 : t1->shape[ 1 ];
  ret->ownsData = true;
  ret->gpu = false;
  ret->data = mem( ret->size, f32 );

  for( u32 i = 0; i < ret->shape[ 0 ]; ++i )
    for( u32 j = 0; j < ret->shape[ 1 ]; ++j ){
      f32 val = 0;
      for( u32 k = 0; k < t2->shape[ 1 ]; ++k )
        val += t1->data[ t1->offset + j * t1->strides[ 1 ] + k * t1->strides[ 0 ] ] *
          t2->data[ t2->offset + k * t2->strides[ 1 ] + i * t2->strides[ 0 ] ];
      ret->data[ i * ret->strides[ 0 ] + j * ret->strides[ 1 ] ] = val;
    }
  
  return ret;
}
void tensorMultiply( tensorStack* ts ){
  tensor* ret = tensorMultiplyHelper( ts->stack[ ts->size - 1 ], ts->stack[ ts->size - 2 ] );
  pop( ts );
  pop( ts );
  push( ts, ret );
}
void tensorSliceHelper( tensor* t, u32 axis, s32 start, s32 end ){
  if( t == NULL )
    error( "%s", "Tensor is NULL in tensorSliceHelper." );
  if( axis >= t->rank )
    error( "Axis %u is out of bounds for tensor of rank %u in slice.", axis, t->rank );

  s32 len = t->shape[ axis ];

  // Adjust negative indices
  if( start < 0 )
    start += len;
  if( end < 0 )
    end += len;

  // Ensure indices are within bounds
  if( start < 0 || end > len || start > end )
    error( "Slice indices out of range: start=%d, end=%d, length=%d",
           start,
           end,
           len );

  s32 new_len = end - start;

  // Adjust the offset based on the stride
  t->offset += t->strides[ axis ] * start;

  // Update the shape
  t->shape[ axis ] = new_len;

  t->size = 1;
  for( u32 i = 0; i < t->rank; ++i )
    t->size *= t->shape[ i ];
}

void tensorSlice( tensorStack* ts, u32 index, u32 axis, s32 start, s32 end ){
  tensorSliceHelper( ts->stack[ index ], axis, start, end );
}

// Function to take the first item of a tensor, reducing its rank by one
void tensorTakeFirstHelper( tensor* t ){
  if( t == NULL )
    error( "%s", "Tensor is NULL in tensorTakeFirstHelper." );
  if( t->rank == 0 )
    error( "%s", "Cannot reduce rank of a tensor with rank 0." );

  // Adjust the offset to point to the first element along axis 0
  t->offset += 0 * t->strides[ 0 ];
  
  // Reduce the rank by one
  t->rank -= 1;

  // Shift the shapes and strides arrays to remove the first axis
  for( u32 i = 0; i <= t->rank; ++i ){
    t->shape[ i ] = t->shape[ i + 1 ];
    t->strides[ i ] = t->strides[ i + 1 ];
  }

  // Update the size of the tensor
  t->size = 1;
  for( u32 i = 0; i < t->rank; ++i )
    t->size *= t->shape[ i ];
}

// Function to apply tensorTakeFirstHelper on a tensor in the stack
void tensorTakeFirst( tensorStack* ts, u32 index ){
  tensorTakeFirstHelper( ts->stack[ index ] );
}
// Function to take the last item of a tensor, reducing its rank by one
void tensorTakeLastHelper( tensor* t ){
  if( t == NULL )
    error( "%s", "Tensor is NULL in tensorTakeLastHelper." );
  if( t->rank == 0 )
    error( "%s", "Cannot reduce rank of a tensor with rank 0." );

  // Adjust the offset to point to the last element along axis 0
  t->offset += ( t->shape[ 0 ] - 1 ) * t->strides[ 0 ];

  // Reduce the rank by one
  t->rank -= 1;

  // Shift the shapes and strides arrays to remove the first axis
  for( u32 i = 0; i <= t->rank; ++i ){
    t->shape[ i ] = t->shape[ i + 1 ];
    t->strides[ i ] = t->strides[ i + 1 ];
  }

  // Update the size of the tensor
  t->size = 1;
  for( u32 i = 0; i < t->rank; ++i )
    t->size *= t->shape[ i ];
}
// Function to apply tensorTakeLastHelper on a tensor in the stack
void tensorTakeLast( tensorStack* ts, u32 index ){
  tensorTakeLastHelper( ts->stack[ index ] );
}
Uint32 getPixel( SDL_Surface* surface, int x, int y ){
  int bpp = surface->format->BytesPerPixel;
  // The start of the pixel row in memory
  Uint8* p = (Uint8*)surface->pixels + y * surface->pitch + x * bpp;

  switch( bpp ){
  case 1:
    return *p;

  case 2:
    return *(Uint16*)p;

  case 3:
    if( SDL_BYTEORDER == SDL_BIG_ENDIAN )
      return ( p[ 0 ] << 16 ) | ( p[ 1 ] << 8 ) | p[ 2 ];
    else
      return p[ 0 ] | ( p[ 1 ] << 8 ) | ( p[ 2 ] << 16 );

  case 4:
    return *(Uint32*)p;

  default:
    // This should never happen for a valid surface
    return 0;
  }
}
tensor* tensorFromImageFile( const char* filename ){
  int w, h, channels;
    
  // Force loading as 4 channels (RGBA), regardless of input format
  unsigned char* pixels = stbi_load( filename, &w, &h, &channels, 4 );
    
  if( !pixels ) {
    error( "Unable to load image file: %s\nReason: %s\n", filename, stbi_failure_reason() );
  }

  tensor* ret = mem( 1, tensor );
  ret->size = w * h * 4;
    
  // Shape: [Width, Height, 4]
  ret->shape[ 0 ] = w;
  ret->shape[ 1 ] = h;
  ret->shape[ 2 ] = 4;
  ret->shape[ 3 ] = 1;
    
  ret->rank = 3;

  // Strides for Column-Major Layout (X is outer dimension)
  // index = (x * h + y) * 4 + c
  ret->strides[ 0 ] = h * 4;
  ret->strides[ 1 ] = 4;
  ret->strides[ 2 ] = 1;
  ret->strides[ 3 ] = 1;

  ret->data = mem( ret->size, f32 );
  ret->ownsData = true;

  // Convert Row-Major (stb) to Column-Major + Y-Flip (Atlas)
  for( u32 x = 0; x < w; ++x ){
    for( u32 y = 0; y < h; ++y ){
      // Source Index (stb is Row-Major: y * w + x)
      // We read 'y' directly (top-down)
      u32 src_idx = ( y * w + x ) * 4;

      // Dest Index (Atlas is Column-Major: x * h + y)
      // We write 'y' flipped (bottom-up) for OpenGL coords
      u32 dest_y = h - 1 - y; 
      u32 dest_idx = ( x * h + dest_y ) * 4;

      // Normalize 0..255 to 0.0..1.0
      // Mapping: R->0, G->1, B->2, A->3 (Standard RGBA)
      ret->data[ dest_idx + 0 ] = pixels[ src_idx + 2 ] / 255.0f; // R
      ret->data[ dest_idx + 1 ] = pixels[ src_idx + 1 ] / 255.0f; // G
      ret->data[ dest_idx + 2 ] = pixels[ src_idx + 0 ] / 255.0f; // B
      ret->data[ dest_idx + 3 ] = pixels[ src_idx + 3 ] / 255.0f; // A
    }
  }

  stbi_image_free( pixels );
  tensorToGPUMemory( ret );
  return ret;
}
tensor* tensorFromString( const char* string ){
  tensor* ret = mem( 1, tensor );
  ret->ownsData = true;
  for( u32 i = 0; i < 4; ++i )
    ret->shape[ i ] = ret->strides[ i ] = 1;
  u32 size = strlen( string );
  ret->shape[ 0 ] = size;
  ret->size = size;
  ret->rank = 1;
  ret->data = mem( size, f32 );
  for( u32 i = 0; i < size; ++i )
    ret->data[ i ] = string[ i ];
  return ret;
}
void tensorRepeatHelper( tensor* t, u32 count ){
  if( !t )
    error( "%s", "Tensor is NULL in tensorRepeatHelper." );
  if( count == 0 )
    error( "%s", "Repeat count must be greater than 0." );
  if( t->rank == 4 )
    error( "%s", "Cannot increase rank of a tensor with rank 4." );

  // Ensure data is on CPU and owned
  tensorToHostMemory( t );
  takeOwnership( t );
  if( !tensorIsContiguous( t ) )
    tensorEnsureContiguous( t );

  // The old rank and size
  u32 old_rank = t->rank;
  u32 old_size = t->size;

  u32 new_rank = old_rank + 1;
  u32 new_size = old_size * count;
  f32* new_data = mem( new_size, f32 );
  for( u32 i = 0; i < count; i++ )
    memcpy(
           new_data + i * old_size, t->data + t->offset, old_size * sizeof( f32 ) );
  if( t->ownsData && t->data )
    unmem( t->data );

  t->data = new_data;
  t->offset = 0;
  t->ownsData = true;
  t->size = new_size;

  t->rank = new_rank;

  for( int i = (int)old_rank - 1; i >= 0; --i )
    t->shape[ i + 1 ] = t->shape[ i ];
  t->shape[ 0 ] = count;

  u32 sz = 1;
  for( int i = (int)new_rank - 1; i >= 0; --i ){
    t->strides[ i ] = sz;
    sz *= t->shape[ i ];
  }
}
void tensorRepeat( tensorStack* ts, u32 index, u32 count ){
  tensorRepeatHelper( ts->stack[ index ], count );
}
void tensorEnclose( tensor* t ){
  for( int i = t->rank; i >= 1; --i ){
    t->shape[ i ] = t->shape[ i - 1 ];
    t->strides[ i ] = t->strides[ i - 1 ];
  }
  t->shape[ 0 ] = 1;
  t->strides[ 0 ] = t->strides[ 1 ];
  t->rank++;
}
void tensorExtrude( tensor* t ){
  t->shape[ t->rank ] = 1;
  t->strides[ t->rank ] = t->rank ? t->strides[ t->rank - 1 ] : 1;
  t->rank++;
}
void tensorUnextrude( tensor* t ){
  t->rank--;
}
bool tensorIsContiguous( const tensor* t ){
  if( t == NULL )
    return false;

  u32 expected_stride = 1;
  for( int i = t->rank - 1; i >= 0; --i ){
    if( t->strides[ i ] != expected_stride )
      return false;
    expected_stride *= t->shape[ i ];
  }
  return true;
}
void tensorEnsureContiguous( tensor* t ){
  if( t == NULL )
    error( "%s", "Tensor is NULL in tensorEnsureContiguous." );

  if( t->gpu )
    tensorToHostMemory( t );

  if( tensorIsContiguous( t ) )
    return;  // Already contiguous, nothing to do.

  
  f32* newData = mem( t->size, f32 );

  u32 std_strides[ 4 ] = { 1, 1, 1, 1 };
  if( t->rank > 0 ){
    std_strides[ t->rank - 1 ] = 1;
    for( int i = t->rank - 2; i >= 0; --i ){
      std_strides[ i ] = std_strides[ i + 1 ] * t->shape[ i + 1 ];
    }
  }

  u32 indices[ 4 ] = { 0, 0, 0, 0 };
  for( u32 i = 0; i < t->size; ++i ){
    // Compute multi-dimensional index based on standard strides.
    u32 tmp = i;
    for( u32 dim = 0; dim < t->rank; ++dim ){
      indices[ dim ] = tmp / std_strides[ dim ];
      tmp %= std_strides[ dim ];
    }

    // Compute source index based on current strides.
    size_t src_idx = t->offset;
    for( u32 dim = 0; dim < t->rank; ++dim ){
      src_idx += (s32)indices[ dim ] * t->strides[ dim ];
    }

    // Copy the element to the new data buffer.
    newData[ i ] = t->data[ src_idx ];
  }

  // Free old data if owned.
  if( t->ownsData ){
    unmem( t->data );
  }

  // Update tensor with new contiguous data.
  t->data = newData;
  t->offset = 0;
  t->ownsData = true;

  // Update strides to standard.
  for( u32 i = 0; i < t->rank; ++i ){
    t->strides[ i ] = std_strides[ i ];
  }
}
char* tensorToString( tensor* t ){
  if( t->rank != 1 )
    return NULL;
  char* ret = mem( t->size + 1, char );
  for( u32 i = 0; i < t->size; ++i )
    ret[ i ] = t->data[ t->offset + i * t->strides[ 0 ] ];
  ret[ t->size ] = 0;
  return ret;
}
void tensorRotate( tensorStack* ts, u32 index, u32 angleIndex ){
  tensor* top = ts->stack[ index ];
  tensor* anglet = ts->stack[ angleIndex ];
  tensorToHostMemory( top );
  tensorToHostMemory( anglet );
  float angle = anglet->data[ anglet->offset ];
  float x = top->data[ top->offset + top->strides[ 0 ] * 0 ];
  float y = top->data[ top->offset + top->strides[ 0 ] * 1 ];
  float z = top->data[ top->offset + top->strides[ 0 ] * 2 ];
  float d = sqrtf( x * x + y * y + z * z );
  x /= d; y /= d; z /= d;
  
  float c = cosf( angle );
  float c1 = 1 - c;
  float s = sinf( angle );
  pop( ts ); pop( ts );
  f32* ret = mem( 16, f32 );
  ret[ 0  ] = c + x * x * c1;      ret[ 1  ] = x * y * c1 - z * s;    ret[ 2  ] = x * z * c1 + y * s;  ret[ 3  ] = 0;
  ret[ 4  ] = y * x * c1 + z * s;  ret[ 5  ] = c + y * y * c1;        ret[ 6  ] = y * z * c1 - x * s;  ret[ 7  ] = 0;
  ret[ 8  ] = z * x * c1 - y * s;  ret[ 9  ] = z * y * c1 + x * s;    ret[ 10 ] = c + z * z * c1;      ret[ 11 ] = 0;
  ret[ 12 ] = 0;                   ret[ 13 ] = 0;                     ret[ 14 ] = 0;                   ret[ 15 ] = 1;
  u32 shape[ 2 ] = { 4, 4 };
  push( ts, newTensor( 2, shape, ret ) );
}
void tensorTranslate( tensorStack* ts, u32 index ){
  tensor* top = ts->stack[ index ];
  tensorToHostMemory( top );
  float x = top->data[ top->offset + top->strides[ 0 ] * 0 ];
  float y = top->data[ top->offset + top->strides[ 0 ] * 1 ];
  float z = top->data[ top->offset + top->strides[ 0 ] * 2 ];

  pop( ts );
  f32* ret = mem( 16, f32 );
  ret[ 0  ] = 1;      ret[ 1  ] = 0;    ret[ 2  ] = 0;  ret[ 3  ] = x;
  ret[ 4  ] = 0;      ret[ 5  ] = 1;    ret[ 6  ] = 0;  ret[ 7  ] = y;
  ret[ 8  ] = 0;      ret[ 9  ] = 0;    ret[ 10 ] = 1;  ret[ 11 ] = z;
  ret[ 12 ] = 0;      ret[ 13 ] = 0;    ret[ 14 ] = 0;  ret[ 15 ] = 1;
  u32 shape[ 2 ] = { 4, 4 };
  push( ts, newTensor( 2, shape, ret ) );
}
void tensorProject( tensorStack* ts, u32 index ){
  tensor* top = ts->stack[ index ];
  tensorToHostMemory( top );
  float fov = tanf( top->data[ top->offset + top->strides[ 0 ] * 0 ] / 2 );
  float width = top->data[ top->offset + top->strides[ 0 ] * 1 ];
  float height = top->data[ top->offset + top->strides[ 0 ] * 2 ];
  float near = top->data[ top->offset + top->strides[ 0 ] * 3 ];
  float far = top->data[ top->offset + top->strides[ 0 ] * 4 ];
  float aspect = sqrtf( width / height );
  
  pop( ts );
  f32* ret = mem( 16, f32 );
  ret[ 0  ] = fov/aspect; ret[ 1  ] = 0;          ret[ 2  ] = 0;                      ret[ 3  ] = 0;
  ret[ 4  ] = 0;          ret[ 5  ] = fov*aspect; ret[ 6  ] = 0;                      ret[ 7  ] = 0;
  ret[ 8  ] = 0;          ret[ 9  ] = 0;          ret[ 10 ] = -(far+near)/(far-near); ret[ 11 ] = -2*far*near/(far-near);
  ret[ 12 ] = 0;          ret[ 13 ] = 0;          ret[ 14 ] = -1;                     ret[ 15 ] = 0;
  u32 shape[ 2 ] = { 4, 4 };
  push( ts, newTensor( 2, shape, ret ) );
}
void tensorOrtho( tensorStack* ts, u32 index ){
  tensor* top = ts->stack[ index ];
  tensorToHostMemory( top );
  float left = top->data[ top->offset + top->strides[ 0 ] * 0 ];
  float right = top->data[ top->offset + top->strides[ 0 ] * 1 ];
  float bottom = top->data[ top->offset + top->strides[ 0 ] * 2 ];
  float topv = top->data[ top->offset + top->strides[ 0 ] * 3 ];
  float near = top->data[ top->offset + top->strides[ 0 ] * 4 ];
  float far = top->data[ top->offset + top->strides[ 0 ] * 5 ];
  float rl = right - left;
  float rlp = right + left;
  float tb = topv - bottom;
  float tbp = topv + bottom;
  float fn = far - near;
  float fnp = far + near;
  
  pop( ts );
  f32* ret = mem( 16, f32 );
  ret[ 0  ] = 2.0 / rl;   ret[ 1  ] = 0;          ret[ 2  ] = 0;                      ret[ 3  ] = -rlp/rl;
  ret[ 4  ] = 0;          ret[ 5  ] = 2.0 / tb;   ret[ 6  ] = 0;                      ret[ 7  ] = -tbp/tb;
  ret[ 8  ] = 0;          ret[ 9  ] = 0;          ret[ 10 ] = -2.0 / fn;              ret[ 11 ] = -fnp/fn;
  ret[ 12 ] = 0;          ret[ 13 ] = 0;          ret[ 14 ] = 0;                      ret[ 15 ] = 1;
  u32 shape[ 2 ] = { 4, 4 };
  push( ts, newTensor( 2, shape, ret ) );
}
void tensorToTextureArray( tensor* t, u32 channels ){
  if( !t ) error( "%s", "Tensor is NULL in tensorToTextureArray." );
  tensorToHostMemory( t );
  tensorEnsureContiguous( t );
  // MUST CACHE DATA BECAUSE WERE MUCKING WITH THE TEXTURES, which are in a union with t->data
  float* data = t->data + t->offset;
  float* dataBase = t->data;
  
  if( t->rank != 4 ) error( "%s", "tensorToTextureArray requires a rank 4 tensor [W, H, Layers, C]." );
  if( !channels || ( t->shape[ 3 ] != channels && t->shape[ 3 ] != channels / 10 ) )
    error( "%s", "tensorToTextureArray called with a bad channel count." );
  u32 layers = t->shape[ 0 ]; 
  u32 height = t->shape[ 1 ];
  u32 width  = t->shape[ 2 ];
  
  GLenum internalFormat, format, type;
  switch( channels ){
  case 40: internalFormat = GL_RGBA8; format = GL_RGBA; type = GL_FLOAT; break;
  case 4:  internalFormat = GL_RGBA32F; format = GL_RGBA; type = GL_FLOAT; break;
  case 10: internalFormat = GL_R8; format = GL_RED; type = GL_FLOAT; break;
  case 1:  internalFormat = GL_R32F; format = GL_RED; type = GL_FLOAT; break;
  default: error( "%s", "Unsupported channel format for textureArray." );
  }

  // 4. Create Texture Array
  glGenTextures( 1, &t->tex.texture );
  glBindTexture( GL_TEXTURE_2D_ARRAY, t->tex.texture );
  
  // Allocation
  glTexImage3D( GL_TEXTURE_2D_ARRAY, 0, internalFormat, width, height, layers, 0, format, type, data );
  
  // Mipmaps & Params
  glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST  );
  glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
  glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE );
  
  glGenFramebuffers( 1, &t->tex.framebuffer );
  glBindFramebuffer( GL_FRAMEBUFFER, t->tex.framebuffer );
  glFramebufferTextureLayer( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, t->tex.texture, 0, 0 );
  glBindTexture( GL_TEXTURE_2D_ARRAY, 0 );

  if( glCheckFramebufferStatus( GL_FRAMEBUFFER ) != GL_FRAMEBUFFER_COMPLETE )
    error( "%s", "Framebuffer is not complete." );

  glBindFramebuffer( GL_FRAMEBUFFER, 0 );

  // 5. Update Tensor State
  // We free the CPU data because we moved it to the GPU
  if( t->ownsData ) unmem( dataBase );
  
  t->gpu = true;
  t->ownsData = true;
  t->tex.width = width;
  t->tex.height = height;
  t->tex.layers = layers;
  t->tex.channels = channels; 
}







// --- KETTLE IMPLEMENTATION ---

typedef struct {
  u32 magic;      // 0x4B544C32 "KTL2"
  u32 count;      // Number of tensors
} KettleHeader;

typedef struct {
  u32 rank;
  u32 shape[4];
  u32 channels;   // 1-4 (f32) or 10-40 (u8)
  u32 isGpu;      // 1 if on GPU, 0 if CPU
  u32 size;       // Element count
  u32 mipmapped;
  u32 layers, width, height;
} KettleMeta;

// Helper to write to a memory buffer and advance the offset
void memWrite(u8* base, u64* offset, void* data, u32 size) {
  memcpy(base + *offset, data, size);
  *offset += size;
}

void textureTensor( tensor* cur ){
  if( !cur->gpu || cur->tex.channels == 0 )
    error( "%s", "Attempt to use an inapropriate tensor as a texture. Must be channeled." );
  glBindTexture( GL_TEXTURE_2D_ARRAY, cur->tex.texture );
  glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR );
  glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR  );
  glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT );
  glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT );
  if( getMaxAnisotropy() > 1.0 ){
    glTexParameterf(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAX_ANISOTROPY_EXT, getMaxAnisotropy() );
  }
  glGenerateMipmap( GL_TEXTURE_2D_ARRAY );
  glBindTexture( GL_TEXTURE_2D_ARRAY, 0 );
  cur->tex.mipmapped = true;
}

void kettle(tensorStack* ts, u32 count, const char* filename) {
  if (count > ts->size) error("Kettle: Stack underflow. Requested %u, has %u.", count, ts->size);

  u32 startIdx = ts->size - count;
    
  // --- PASS 1: Calculate Total Uncompressed Size ---
  u64 totalUncompressedSize = sizeof(KettleHeader);

  // We need to look at the tensors to know their specific storage size (f32 vs u8)
  // Note: We don't modify the tensors yet, just peek at them.
  for (u32 i = 0; i < count; ++i) {
    tensor* t = ts->stack[startIdx + i];
    totalUncompressedSize += sizeof(KettleMeta);

    // Determine effective channels for storage size calculation
    u32 ch = t->gpu ? t->tex.channels : 0; 
        
    bool isQuantized = (ch == 10 || ch == 20 || ch == 30 || ch == 40);
    u32 elementSize = isQuantized ? sizeof(u8) : sizeof(f32);
        
    totalUncompressedSize += (u64)t->size * elementSize;
  }

  // --- PASS 2: Serialize to Memory ---
  u8* rawData = mem(totalUncompressedSize, u8);
  u64 writeHead = 0;

  KettleHeader h = { 0x4B544C31, count }; // compressedSize is 0 in the inner header
  memWrite(rawData, &writeHead, &h, sizeof(KettleHeader));

  for (u32 i = 0; i < count; ++i) {
    tensor* t = ts->stack[startIdx + i];

    // 1. Prepare Meta
    KettleMeta meta = {0};
    meta.rank = t->rank;
    memcpy(meta.shape, t->shape, sizeof(u32) * 4);
    meta.size = t->size;
    meta.isGpu = t->gpu ? 1 : 0;
    meta.channels = t->gpu ? t->tex.channels : 0;

    if (t->gpu) {
      meta.layers = t->tex.layers;
      meta.width = t->tex.width;
      meta.height = t->tex.height;
      meta.mipmapped = t->tex.mipmapped;
    }

    // 2. Fetch Data
    tensorToHostMemory(t);
    tensorEnsureContiguous(t);

    // 3. Write Meta
    memWrite(rawData, &writeHead, &meta, sizeof(KettleMeta));

    // 4. Write Data (Quantized or F32)
    if (meta.channels == 10 || meta.channels == 20 || 
        meta.channels == 30 || meta.channels == 40) {
            
      // We can write directly to the buffer to avoid a temp malloc
      u8* dst = rawData + writeHead;
      for (u32 k = 0; k < meta.size; ++k) {
        f32 val = t->data[k];
        if (val < 0.0f) val = 0.0f;
        if (val > 1.0f) val = 1.0f;
        dst[k] = (u8)(val * 255.0f);
      }
      writeHead += meta.size; // Advance by size * sizeof(u8)
    } else {
      memWrite(rawData, &writeHead, t->data, sizeof(f32) * meta.size);
    }
  }

  // --- PASS 3: Compress and Write to Disk ---
  mz_ulong compressedLen = mz_compressBound(totalUncompressedSize);
  u8* compressedData = mem(compressedLen, u8);

  int status = mz_compress(compressedData, &compressedLen, rawData, (mz_ulong)totalUncompressedSize);
  if (status != MZ_OK) error("Kettle: Compression failed with error %d", status);

  FILE* f = fopen(filename, "wb");
  if (!f) error("Kettle: Could not open %s for writing.", filename);

  // Write file container format: [UncompressedSize (u32)] [CompressedSize (u32)] [Data...]
  u32 uSize = (u32)totalUncompressedSize;
  u32 cSize = (u32)compressedLen;

  fwrite(&uSize, sizeof(u32), 1, f);
  fwrite(&cSize, sizeof(u32), 1, f);
  fwrite(compressedData, 1, cSize, f);

  fclose(f);
    
  // Cleanup
  unmem(rawData);
  unmem(compressedData);

  printf("Cooked %u tensors into %s (Ratio: %.2f%%)\n", count, filename, (float)cSize/uSize * 100.0f);
}

void unkettle( tensorStack* ts, const char* filename ){
  FILE* f = fopen( filename, "rb" );
  if( !f ) error( "Unkettle: Could not open %s.", filename );
  u32 uSize, cSize;
  fread( &uSize, sizeof( u32 ), 1, f );
  fread( &cSize, sizeof( u32 ), 1, f );
  
  // 1. Read Compressed Data from File
  u8* compressedData = mem( cSize, u8 );
  if( fread( compressedData, 1, cSize, f ) != cSize ){
    fclose( f );
    error( "%s", "Unkettle: File truncated." );
  }
  fclose( f ); // File is closed here. Do not close again at bottom.

  // 2. Decompress
  u8* decompressedBuffer = mem( uSize, u8 ); // Renamed for clarity vs macro
  mz_ulong destLen = uSize;
  int status = mz_uncompress( decompressedBuffer, &destLen, compressedData, cSize );
    
  if( status != MZ_OK ) error( "Unkettle: Decompression failed %d", status );
    
  unmem( compressedData );
  
  // 3. Setup a pointer for the parsing loop
  u64 readHead = 0;

  // Macro uses 'decompressedBuffer' specifically now
#define READ_MEM( dest, size ) { memcpy( dest, decompressedBuffer + readHead, size ); readHead += size; }
    
  KettleHeader h;
  READ_MEM( &h, sizeof(KettleHeader) );
  if( h.magic != 0x4B544C31 ){
    error( "Unkettle: Invalid file format in %s.", filename );
  }
  
  for( u32 i = 0; i < h.count; ++i ){
    KettleMeta meta;
    READ_MEM( &meta, sizeof( KettleMeta ) );

    tensor* t = mem( 1, tensor );
    t->rank = meta.rank;
    t->size = meta.size;
    memcpy( t->shape, meta.shape, sizeof(u32)*4 );
    
    // Reconstruct strides
    u32 stride = 1;
    for( int k = t->rank - 1; k >= 0; --k ){
      t->strides[ k ] = stride;
      stride *= t->shape[ k ];
    }
    for( u32 k = t->rank; k < 4; ++k ){
      t->shape[k] = 1;
      t->strides[k] = 1;
    }

    t->offset = 0;
    t->ownsData = true;
    t->gpu = ( meta.isGpu != 0 );

    if( t->gpu ){
      // --- GPU UPLOAD PATH ---
      t->tex.channels = meta.channels;
      t->tex.layers = meta.layers;
      t->tex.height = meta.height;
      t->tex.width  = meta.width;

      GLenum internalFormat = GL_RGBA32F;
      GLenum format = GL_RGBA;
      GLenum type = GL_FLOAT;
      
      // FIXED: Rename this so it doesn't shadow 'decompressedBuffer' 
      // or confuse the macro logic
      void* texData = NULL; 
      
      u32 bytesToRead = 0;
      bool isU8 = false;

      // Determine GL formats based on Atlas Channel Logic
      if( meta.channels == 40 ){
        internalFormat = GL_RGBA8; format = GL_RGBA; type = GL_UNSIGNED_BYTE; isU8 = true;
      } else if( meta.channels == 30 ){
        internalFormat = GL_RGB8;  format = GL_RGB;  type = GL_UNSIGNED_BYTE; isU8 = true;
      } else if( meta.channels == 20 ){
        internalFormat = GL_RG8;   format = GL_RG;   type = GL_UNSIGNED_BYTE; isU8 = true;
      } else if( meta.channels == 10 ){
        internalFormat = GL_R8;    format = GL_RED;  type = GL_UNSIGNED_BYTE; isU8 = true;
      } else if( meta.channels == 4 ){
        internalFormat = GL_RGBA32F; format = GL_RGBA; type = GL_FLOAT;
      } else if( meta.channels == 3 ){
        internalFormat = GL_RGB32F;  format = GL_RGB;  type = GL_FLOAT;
      } else if( meta.channels == 2 ){
        internalFormat = GL_RG32F;   format = GL_RG;   type = GL_FLOAT;
      } else if( meta.channels == 1 ){
        internalFormat = GL_R32F;    format = GL_RED;  type = GL_FLOAT;
      }

      bytesToRead = meta.size * ( isU8 ? 1 : 4 );
      u32 channelMult = 4;
      if( meta.channels > 10 )
        channelMult = meta.channels / 10;
      else if( meta.channels )
        channelMult = meta.channels;
      
      // Allocate temp buffer for upload
      texData = mem( t->tex.width * t->tex.height * t->tex.layers * ( isU8 ? 1 : 4 ) * channelMult, u8 );
      
      // FIXED: Read from the global decompressedBuffer INTO texData
      READ_MEM( texData, bytesToRead );

      glGenTextures( 1, &t->tex.texture );
      glBindTexture( GL_TEXTURE_2D_ARRAY, t->tex.texture );
      
      glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
      glTexImage3D( GL_TEXTURE_2D_ARRAY, 0, internalFormat, 
                    t->tex.width, t->tex.height, t->tex.layers, 
                    0, format, type, texData );

      if( meta.mipmapped ){
        glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR );
        glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
        glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT );
        glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT );
        if( getMaxAnisotropy() > 1.0 )
          glTexParameterf(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAX_ANISOTROPY_EXT, getMaxAnisotropy() );
        glGenerateMipmap( GL_TEXTURE_2D_ARRAY );
        t->tex.mipmapped = true;
      }else{
        glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
        glTexParameteri( GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
      }

      glGenFramebuffers( 1, &t->tex.framebuffer );
      glBindFramebuffer( GL_FRAMEBUFFER, t->tex.framebuffer );
      glFramebufferTextureLayer( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, t->tex.texture, 0, 0 );
      
      if( glCheckFramebufferStatus( GL_FRAMEBUFFER ) != GL_FRAMEBUFFER_COMPLETE )
        error( "%s", "Unkettle: Framebuffer incomplete." );
          
      glBindFramebuffer( GL_FRAMEBUFFER, 0 );
      glBindTexture( GL_TEXTURE_2D_ARRAY, 0 );

      unmem( texData ); // Free the temp buffer

    } else {
      // CPU
      t->data = mem( meta.size, f32 );
      
      if( meta.channels >= 10 ) {
        u8* tmp = mem( meta.size, u8 );
        READ_MEM( tmp, meta.size );
        for( u32 k=0; k<meta.size; ++k ) t->data[k] = (f32)tmp[k] / 255.0f;
        unmem( tmp );
      } else {
        READ_MEM( t->data, sizeof(f32) * meta.size );
      }
    }
    push( ts, t );
  }

  unmem( decompressedBuffer );
}
tensor* textBufferView( u32 width, u32 height, u32 scrollUp ){
  u32 shape[ 2 ] = { height, width };
  u32 totalCells = width * height;
  f32* view = mem( totalCells, f32 ); 

  for( u32 i = 0; i < totalCells; ++i ){
    view[ i ] = 32.0f; 
  }

  s64 bufIdx = textBufferPos - 1;
  s64 row = height - 1;           
  
  char* lineCache = mem( width, char ); 
  u32 lineLen = 0;

  u32 linesSkipped = 0; 

  while( bufIdx >= 0 && row >= 0 ){ 
    char c = textBuffer[ bufIdx-- ];

    if( c == '\n' ){
      if( linesSkipped >= scrollUp ){
        if( row < height ){
          for( u32 k = 0; k < lineLen; ++k ){
            view[ row * width + k ] = (f32)lineCache[ lineLen - 1 - k ];
          }
        }
        row--; 
      } else {
        linesSkipped++;
      }
      
      lineLen = 0;
      continue;
    }

    if( lineLen >= width ){
      if( linesSkipped >= scrollUp ){
        if( row < height ){
          for( u32 k = 0; k < lineLen; ++k ){
            view[ row * width + k ] = (f32)lineCache[ lineLen - 1 - k ];
          }
        }
        row--;
      } else {
        linesSkipped++;
      }
      lineLen = 0;
    }

    lineCache[ lineLen++ ] = c;
  }

  if( lineLen > 0 ){
    if( linesSkipped >= scrollUp && row >= 0 && row < height ){
      for( u32 k = 0; k < lineLen; ++k ){
        view[ row * width + k ] = (f32)lineCache[ lineLen - 1 - k ];
      }
    }
  }

  unmem( lineCache );
  return newTensor( 2, shape, view );
}
