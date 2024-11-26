////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4 et al.   //
////////////////////////////////////////////////////////////////////////////////
#include "Atlas.h" 
// Main must define this.
u64 memc = 0;

// The compiled program to be run.
program* prog;
tensorStack* ts; 

char* testProg = 
  "size;if'start'\n"
  "[3 5 3];i't.y / 5.0'\n"
  "print\n"
  "l'start';1;r;[0 1];t\n"
  "\n";



// Global variables
SDL_Window* window = NULL;
SDL_GLContext glContext; 
GLuint shaderProgram; 
GLuint vbo;
float zoom = 1.0f;
float offsetX = -0.5f, offsetY = 0.0f; // Centered on Mandelbrot set
int windowWidth = 800, windowHeight = 600; 
bool running = true;

// Vertex Shader Source
const GLchar* vertexSource = 
  "precision highp float;\n"
  "attribute vec2 position;\n"
  "varying vec2 fragCoord;\n"
  "void main() {\n"
  "  fragCoord = position;\n"
  "  gl_Position = vec4( position, 0.0, 1.0 );\n"
  "}\n";

// Fragment Shader Source
const GLchar* fragmentSource = 
  "precision highp float;\n"
  "varying vec2 fragCoord;\n"
  "uniform float zoom;\n"
  "uniform sampler2D tex;\n"
  "uniform vec2 offset;\n"
  "uniform vec2 resolution;\n"
  "uniform vec4 strides;\n"
  "uniform vec4 shape;\n"
  "uniform vec2 dims;\n"
  "uniform float toffset;\n"
  "float sampleTensorIndex(vec4 i) {\n"
  "  float lindex = dot( i, strides ) + toffset;\n"
  "  float pixel_index = floor( lindex / 4.0 );\n"
  "  float channel = mod( lindex, 4.0 );\n"
  "  vec2 uv = ( vec2( mod( pixel_index, dims.x ), floor( pixel_index / dims.x ) ) + 0.5 ) / dims;\n"
  "  vec4 texel = texture2D( tex, uv );\n"
  "  if( channel < 1.0 )\n"
  "    return texel.r;\n"
  "  else if( channel < 2.0 )\n"
  "    return texel.g;\n"
  "  else if( channel < 3.0 )\n"
  "    return texel.b;\n"
  "  else\n"
  "    return texel.a;\n"
  "}\n"
  "void main(){\n"
  "  vec2 uv = fragCoord;\n"
  "  uv.x *= resolution.x / resolution.y; // Adjust for aspect ratio\n"
  "  vec2 c = uv * zoom + offset;\n"
  "  vec2 z = c;\n"
  "  int iterations = 0;\n"
  "  const int maxIterations = 100;\n"
  "  for (int i = 0; i < maxIterations; i++) {\n"
  "    float x = (z.x * z.x - z.y * z.y ) + c.x;\n"
  "    float y = (2.0 * z.x * z.y ) + c.y;\n"
  "    if(( x * x + y * y ) > 4.0 ) break;\n"
  "    z.x = x;\n"
  "    z.y = y;\n"
  "    iterations++;\n"
  "  }\n"
  "  float color = float( iterations ) / float( maxIterations );\n"
  "  vec4 tindex = vec4( floor( ( fragCoord.xy * 0.5 + 0.5 ) * shape.xy ), 0, 0 );\n"
  "  float tcolor = sampleTensorIndex( tindex );\n"
  "  gl_FragColor = vec4( tcolor, 1.0 - color, 0.0, 1.0 );\n"
  "}\n";

// Function to compile shaders
GLuint compileShader( GLenum type, const GLchar* source ){
  GLuint shader = glCreateShader( type );
  glShaderSource( shader, 1, &source, NULL );
  glCompileShader( shader );

  // Check for compilation errors
  GLint status;
  glGetShaderiv( shader, GL_COMPILE_STATUS, &status );
  if( status != GL_TRUE ) {
    char buffer[ 1024 ];
    glGetShaderInfoLog( shader, sizeof( buffer ), NULL, buffer );
    error( "Shader compilation failed: %s\n", buffer );
  }
  return shader;
}

// Function to create shader program
GLuint createProgram( const GLchar* vertexSource,
		      const GLchar* fragmentSource ) {
  GLuint vertexShader = compileShader( GL_VERTEX_SHADER, vertexSource );
  GLuint fragmentShader = compileShader( GL_FRAGMENT_SHADER, fragmentSource );

  GLuint program = glCreateProgram();
  glAttachShader( program, vertexShader );
  glAttachShader( program, fragmentShader );
  glBindAttribLocation( program, 0, "position");
  glLinkProgram( program );

  // Check for linking errors
  GLint status;
  glGetProgramiv( program, GL_LINK_STATUS, &status );
  if( status != GL_TRUE ) {
    char buffer[ 1024 ];
    glGetProgramInfoLog( program, 512, NULL, buffer );
    printf("Program linking failed: %s\n", buffer );
  }

  glDeleteShader( vertexShader );
  glDeleteShader( fragmentShader );

  return program;
}

// Main loop function
void mainLoop(){
  if( !runProgram( ts, prog ) ){
    running = false;
    return;
  }
  // Handle events
  SDL_Event event;
  while( SDL_PollEvent(&event ) ){
    if( event.type == SDL_QUIT ) {
      running = false;
    } else if( event.type == SDL_MOUSEWHEEL ){
      if( event.wheel.y > 0 ){
	zoom *= 0.9f; // Zoom in
      } else if( event.wheel.y < 0 ){
	zoom *= 1.1f; // Zoom out
      }
    } else if( event.type == SDL_MOUSEMOTION ){
      if( event.motion.state & SDL_BUTTON_LMASK ){
	float deltaX =
	  (float)event.motion.xrel / windowWidth * zoom * 2.0f;
	float deltaY =
	  (float)event.motion.yrel / windowHeight * zoom * 2.0f;
	offsetX -= deltaX;
	offsetY += deltaY;
      }
    } else if( event.type == SDL_KEYDOWN ){
      if( event.key.keysym.sym == SDLK_ESCAPE )
	running = false;
      break;
    }
  }
  // Render
  if( !ts->top )
    return;
  if( !ts->stack[ ts->top - 1 ]->gpu )
    tensorToGPUMemory( ts->stack[ ts->top - 1 ] );
  if( ts->stack[ ts->top - 1 ]->rank != 3 )
    error( "%s", "Display tensor not of rank 3" );
  if( ts->stack[ ts->top - 1 ]->shape[ 2 ] != 3 )
    error( "%s", "Display tensor not a 3 component tensor of rank 3." );

  glClear( GL_COLOR_BUFFER_BIT );
  
  glUseProgram( shaderProgram );
  glViewport( 0, 0, windowWidth, windowHeight );
  
  GLint texLoc = glGetUniformLocation( shaderProgram, "tex");
  glUniform1i( texLoc, 0 ); // Texture unit 0

  // Bind the texture to texture unit 0
  glActiveTexture( GL_TEXTURE0 );
  glBindTexture( GL_TEXTURE_2D, ts->stack[ ts->top - 1 ]->tex.texture );
  
  // Set uniforms
  GLint zoomLoc = glGetUniformLocation( shaderProgram, "zoom");
  glUniform1f( zoomLoc, zoom );
  
  GLint offsetLoc = glGetUniformLocation( shaderProgram, "offset");
  glUniform2f( offsetLoc, offsetX, offsetY );

  GLint dimsLoc = glGetUniformLocation( shaderProgram, "dims" );
  glUniform2f( dimsLoc, ts->stack[ ts->top - 1 ]->tex.width, ts->stack[ ts->top - 1 ]->tex.height );

  GLint resolutionLoc = glGetUniformLocation( shaderProgram, "resolution");
  glUniform2f( resolutionLoc, (float)windowWidth, (float)windowHeight );

  GLint stridesLoc = glGetUniformLocation( shaderProgram, "strides");
  glUniform4f( stridesLoc, ts->stack[ ts->top - 1 ]->strides[ 0 ], ts->stack[ ts->top - 1 ]->strides[ 1 ],
	       ts->stack[ ts->top - 1 ]->strides[ 2 ], ts->stack[ ts->top - 1 ]->strides[ 3 ] );
  GLint shapeLoc = glGetUniformLocation( shaderProgram, "shape");
  glUniform4f( shapeLoc, ts->stack[ ts->top - 1 ]->shape[ 0 ], ts->stack[ ts->top - 1 ]->shape[ 1 ],
	       ts->stack[ ts->top - 1 ]->shape[ 2 ], ts->stack[ ts->top - 1 ]->shape[ 3 ] );

  GLint toffsetLoc = glGetUniformLocation( shaderProgram, "toffset");
  glUniform1f( toffsetLoc, ts->stack[ ts->top - 1 ]->offset );

  
  // Bind VBO and set vertex attributes
  glBindBuffer( GL_ARRAY_BUFFER, vbo );
  GLint posAttrib = glGetAttribLocation( shaderProgram, "position");
  glEnableVertexAttribArray( posAttrib );
  glVertexAttribPointer( posAttrib, 2, GL_FLOAT, GL_FALSE, 0, 0 );
  
  // Draw
  glBindFramebuffer( GL_FRAMEBUFFER, 0 );
  glDrawArrays( GL_TRIANGLE_STRIP, 0, 4 );
  
  // Cleanup
  glDisableVertexAttribArray( posAttrib );
  
  SDL_GL_SwapWindow( window );
}

void test( void );
int main( int argc, char* argv[] ){
  if( SDL_Init( SDL_INIT_VIDEO ) != 0 )
    error( "SDL_Init Error: %s\n", SDL_GetError() );

  SDL_GL_SetAttribute( SDL_GL_CONTEXT_MAJOR_VERSION, 2 );
  SDL_GL_SetAttribute( SDL_GL_CONTEXT_MINOR_VERSION, 1 );

  window =
    SDL_CreateWindow( "Atlas",
		      SDL_WINDOWPOS_CENTERED,
		      SDL_WINDOWPOS_CENTERED,
		      windowWidth, windowHeight,
		      SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN );

  if( !window )
    error( "SDL_CreateWindow Error: %s\n", SDL_GetError() );

  glContext = SDL_GL_CreateContext( window );
  if( !glContext )
    error( "SDL_GL_CreateContext Error: %s\n", SDL_GetError() );

  // Initialize GLEW
#ifndef __EMSCRIPTEN__
  glewExperimental = GL_TRUE; // Enable modern OpenGL techniques
  GLenum glewError = glewInit();
  if( glewError != GLEW_OK ) {
    printf("glewInit Error: %s\n", glewGetErrorString( glewError ));
    return 1;
  }
#endif

  // Initialize OpenGL
  glViewport( 0, 0, windowWidth, windowHeight );
  glClearColor( 0.0f, 0.0f, 0.0f, 1.0f );

  shaderProgram = createProgram( vertexSource, fragmentSource );

  // Set up vertex data
  GLfloat vertices[] = {
    -1.0f, -1.0f, // Bottom-left
    1.0f, -1.0f, // Bottom-right
    -1.0f,  1.0f, // Top-left
    1.0f,  1.0f, // Top-right
  };

  // Generate VBO
  glGenBuffers( 1, &vbo );
  glBindBuffer( GL_ARRAY_BUFFER, vbo );
  glBufferData( GL_ARRAY_BUFFER, sizeof( vertices ), vertices,
		GL_STATIC_DRAW );

  // Compile program.
  prog = newProgramFromString( testProg );
  ts = newStack();
  test();
#ifdef __EMSCRIPTEN__
  // Start the main loop
  emscripten_set_main_loop( mainLoop, 0, 1 );
  // Main loop
#else
  while( running ) {
    mainLoop();
  }
#endif
    
  deleteProgram( prog );
  deleteStack( ts );

  // Clean up
  glDeleteProgram( shaderProgram );
  glDeleteBuffers( 1, &vbo );
  SDL_GL_DeleteContext( glContext );
  SDL_DestroyWindow( window );
  dbg( "mem count %llu", memc );
  SDL_Quit();
  return 0;
}


void test( void ){
    // Create the root of the trie
    trieNode* root = newTrieNode(NULL, 0);

    // Insert some keys and values
    trieInsert(root, "apple", 100);
    trieInsert(root, "app", 50);
    trieInsert(root, "banana", 150);
    trieInsert(root, "band", 75);
    trieInsert(root, "bandana", 200);

    // Search for keys
    const char* keysToSearch[] = {"apple", "app", "banana", "band", "bandana", "bandit", "apricot", "ban"};
    size_t numKeys = sizeof(keysToSearch) / sizeof(keysToSearch[0]);

    for (size_t i = 0; i < numKeys; ++i) {
        u32 value;
        bool found = trieSearch(root, keysToSearch[i], &value);
        if (found) {
            printf("Key '%s' found with value: %u\n", keysToSearch[i], value);
        } else {
            printf("Key '%s' not found.\n", keysToSearch[i]);
        }
    }

    // Clean up
    deleteTrieNode(root);

}
