////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4-o1.      //
////////////////////////////////////////////////////////////////////////////////
#include "Atlas.h" 
// Main must define this.
u64 memc = 0;

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
  "attribute vec2 position;\n"
  "varying vec2 fragCoord;\n"
  "void main() {\n"
  "  fragCoord = position;\n"
  "  gl_Position = vec4( position, 0.0, 1.0 );\n"
  "}\n";

// Fragment Shader Source
const GLchar* fragmentSource = 
  "precision mediump float;\n"
  "varying vec2 fragCoord;\n"
  "uniform float zoom;\n"
  "uniform vec2 offset;\n"
  "uniform vec2 resolution;\n"
  "void main() {\n"
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
  "  gl_FragColor = vec4( 0.0, color, 0.0, 1.0 );\n"
  "}\n";

// Function to compile shaders
GLuint compileShader( GLenum type, const GLchar* source ) {
    GLuint shader = glCreateShader( type );
    glShaderSource( shader, 1, &source, NULL );
    glCompileShader( shader );

    // Check for compilation errors
    GLint status;
    glGetShaderiv( shader, GL_COMPILE_STATUS, &status );
    if( status != GL_TRUE ) {
        char buffer[512];
        glGetShaderInfoLog( shader, 512, NULL, buffer );
        printf("Shader compilation failed: %s\n", buffer );
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
        char buffer[512];
        glGetProgramInfoLog( program, 512, NULL, buffer );
        printf("Program linking failed: %s\n", buffer );
    }

    glDeleteShader( vertexShader );
    glDeleteShader( fragmentShader );

    return program;
}

// Main loop function
void mainLoop() {
    // Handle events
    SDL_Event event;
    while( SDL_PollEvent(&event )) {
        if( event.type == SDL_QUIT ) {
            running = false;
        } else if( event.type == SDL_MOUSEWHEEL ) {
            if( event.wheel.y > 0 ) {
                zoom *= 0.9f; // Zoom in
            } else if( event.wheel.y < 0 ) {
                zoom *= 1.1f; // Zoom out
            }
        } else if( event.type == SDL_MOUSEMOTION ) {
            if( event.motion.state & SDL_BUTTON_LMASK ) {
                float deltaX =
		  (float)event.motion.xrel / windowWidth * zoom * 2.0f;
                float deltaY =
		  (float)event.motion.yrel / windowHeight * zoom * 2.0f;
                offsetX -= deltaX;
                offsetY += deltaY;
            }
        }
    }

    // Render
    glClear( GL_COLOR_BUFFER_BIT );

    glUseProgram( shaderProgram );

    // Set uniforms
    GLint zoomLoc = glGetUniformLocation( shaderProgram, "zoom");
    glUniform1f( zoomLoc, zoom );

    GLint offsetLoc = glGetUniformLocation( shaderProgram, "offset");
    glUniform2f( offsetLoc, offsetX, offsetY );

    GLint resolutionLoc = glGetUniformLocation( shaderProgram, "resolution");
    glUniform2f( resolutionLoc, (float)windowWidth, (float)windowHeight );

    // Bind VBO and set vertex attributes
    glBindBuffer( GL_ARRAY_BUFFER, vbo );
    GLint posAttrib = glGetAttribLocation( shaderProgram, "position");
    glEnableVertexAttribArray( posAttrib );
    glVertexAttribPointer( posAttrib, 2, GL_FLOAT, GL_FALSE, 0, 0 );

    // Draw
    glDrawArrays( GL_TRIANGLE_STRIP, 0, 4 );

    // Cleanup
    glDisableVertexAttribArray( posAttrib );

    SDL_GL_SwapWindow( window );
}


void test();
int main( int argc, char* argv[] ) {
    if( SDL_Init( SDL_INIT_VIDEO ) != 0 ) {
        printf("SDL_Init Error: %s\n", SDL_GetError());
        return 1;
    }

    // Request OpenGL 2.1 context
    SDL_GL_SetAttribute( SDL_GL_CONTEXT_MAJOR_VERSION, 2 );
    SDL_GL_SetAttribute( SDL_GL_CONTEXT_MINOR_VERSION, 1 );

    window =
      SDL_CreateWindow( "Fractal Zoomer",
			SDL_WINDOWPOS_CENTERED,
			SDL_WINDOWPOS_CENTERED,
                        windowWidth, windowHeight,
			SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN );

    if( !window ) {
        printf("SDL_CreateWindow Error: %s\n", SDL_GetError());
        return 1;
    }

    glContext = SDL_GL_CreateContext( window );
    if( !glContext ) {
        printf("SDL_GL_CreateContext Error: %s\n", SDL_GetError());
        return 1;
    }

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
    
    // Clean up
    glDeleteProgram( shaderProgram );
    glDeleteBuffers( 1, &vbo );
    SDL_GL_DeleteContext( glContext );
    SDL_DestroyWindow( window );
    SDL_Quit();
    return 0;
}


void test(){
  f32 data[] = { 106.5, 2, 3, 4, 7, 6, 7.7, 8, 9, 10, 11, 12 };
  u32 shape1[] = { 2, 2, 3 };
  u32 shape2[] = { 3, 2, 2 };
  u32 shape3[] = { 2, 1, 2, 3 };
  u32 shape4[] = { 2, 6 };
  u32 shape5[] = { 3, 4 };
  u32 shape6[] = { 1, 12 };
  u32 shape7[] = { 12, 1 };
  tensorStack* ts = newStack();

  push( ts, 3, shape1, data );
  push( ts, 3, shape2, data );
  push( ts, 4, shape3, data );
  push( ts, 2, shape4, data );
  push( ts, 2, shape5, data );
  push( ts, 2, shape6, data );
  push( ts, 2, shape7, data );
  push( ts, 0, NULL, data );

  //  printStack( ts );

  pop( ts ); pop( ts ); pop( ts ); pop( ts ); pop( ts ); pop( ts ); pop( ts );
  printf( "\n\n\n\n\n\n\n\n" );
  
  f32 d2[] = { 222, 250, 1, 2, 3, 4 }; 
  u32 shape8[] = { 3, 2, 1 }; 
  push( ts, 3, shape8, d2 ); 
  //  printStack( ts ); fflush( stdout );
  //  tensorIndex( ts, ts->top - 1, ts->top - 2 );
  u32 shape9[] = { 3, 2, 1 };
  tensorReshape( ts, ts->top - 1, 3, shape9 );
  printStack( ts );

  /* u8 d3[] = { 0 }; */
  /* u32 shape9[] = { 1 }; */
  /* push( ts, 1, shape9, d3 ); */
  /* printStack( ts ); */
  /* tensorIndex( ts ); */
  /* printStack( ts ); */

  
  /* u8 d4[] = { 2, 0, 0, 0, 0, 0 }; */
  /* u32 shape10[] = { 6 }; */
  /* push( ts, 1, shape10, d4 ); */
  /* printStack( ts ); */
  /* tensorIndex( ts ); */
  /* printStack( ts ); */

  deleteStack( ts );

  printf( "mem count %llu", memc );
}
