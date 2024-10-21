////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4-o1.      //
////////////////////////////////////////////////////////////////////////////////
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengles2.h>
#include <emscripten/emscripten.h>
#include <stdio.h>

// Global variables
SDL_Window* window;
SDL_GLContext glContext;
GLuint shaderProgram;
GLuint vbo;
float zoom = 1.0f;
float offsetX = -0.5f, offsetY = 0.0f; // Centered on Mandelbrot set
int windowWidth = 800, windowHeight = 600;

// Vertex Shader Source
const GLchar* vertexSource = R"glsl(
    attribute vec2 position;
    varying vec2 fragCoord;
    void main() {
        fragCoord = position;
        gl_Position = vec4(position, 0.0, 1.0);
    }
)glsl";

// Fragment Shader Source
const GLchar* fragmentSource = R"glsl(
    precision mediump float;
    varying vec2 fragCoord;
    uniform float zoom;
    uniform vec2 offset;
    uniform vec2 resolution;
    void main() {
        vec2 uv = fragCoord;
        uv.x *= resolution.x / resolution.y; // Adjust for aspect ratio
        vec2 c = uv * zoom + offset;
        vec2 z = c;
        int iterations = 0;
        const int maxIterations = 1024;
        for (int i = 0; i < maxIterations; i++) {
            float x = (z.x * z.x - z.y * z.y) + c.x;
            float y = (2.0 * z.x * z.y) + c.y;
            if ((x * x + y * y) > 4.0) break;
            z.x = x;
            z.y = y;
            iterations++;
        }
        float color = float(iterations) / float(maxIterations);
        gl_FragColor = vec4(vec3(color), 1.0);
    }
)glsl";

// Function to compile shaders
GLuint compileShader(GLenum type, const GLchar* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    // Check for compilation errors
    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE) {
        char buffer[512];
        glGetShaderInfoLog(shader, 512, NULL, buffer);
        printf("Shader compilation failed: %s\n", buffer);
    }
    return shader;
}

// Function to create shader program
GLuint createProgram(const GLchar* vertexSource, const GLchar* fragmentSource) {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSource);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    // Bind the attribute location before linking
    glBindAttribLocation(program, 0, "position");
    glLinkProgram(program);

    // Check for linking errors
    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status != GL_TRUE) {
        char buffer[512];
        glGetProgramInfoLog(program, 512, NULL, buffer);
        printf("Program linking failed: %s\n", buffer);
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}

// Main loop function
void mainLoop() {
    // Handle events
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) {
            emscripten_cancel_main_loop();
        } else if (event.type == SDL_MOUSEWHEEL) {
            if (event.wheel.y > 0) {
                zoom *= 0.9f; // Zoom in
            } else if (event.wheel.y < 0) {
                zoom *= 1.1f; // Zoom out
            }
        } else if (event.type == SDL_MOUSEMOTION) {
            if (event.motion.state & SDL_BUTTON_LMASK) {
                float deltaX = (float)event.motion.xrel / windowWidth * zoom * 2.0f;
                float deltaY = (float)event.motion.yrel / windowHeight * zoom * 2.0f;
                offsetX -= deltaX;
                offsetY += deltaY;
            }
        }
    }

    // Render
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(shaderProgram);

    // Set uniforms
    GLint zoomLoc = glGetUniformLocation(shaderProgram, "zoom");
    glUniform1f(zoomLoc, zoom);

    GLint offsetLoc = glGetUniformLocation(shaderProgram, "offset");
    glUniform2f(offsetLoc, offsetX, offsetY);

    GLint resolutionLoc = glGetUniformLocation(shaderProgram, "resolution");
    glUniform2f(resolutionLoc, (float)windowWidth, (float)windowHeight);

    // Bind VBO and set vertex attributes
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    GLint posAttrib = glGetAttribLocation(shaderProgram, "position");
    glEnableVertexAttribArray(posAttrib);
    glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, 0, 0);

    // Draw
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    // Cleanup
    glDisableVertexAttribArray(posAttrib);
    // No need to unbind the buffer here

    SDL_GL_SwapWindow(window);
}

int main() {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        printf("SDL_Init Error: %s\n", SDL_GetError());
        return 1;
    }

    // Request OpenGL ES 2.0 context
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);

    window = SDL_CreateWindow("Fractal Zoomer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                              windowWidth, windowHeight, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);

    if (!window) {
        printf("SDL_CreateWindow Error: %s\n", SDL_GetError());
        return 1;
    }

    glContext = SDL_GL_CreateContext(window);
    if (!glContext) {
        printf("SDL_GL_CreateContext Error: %s\n", SDL_GetError());
        return 1;
    }

    // Initialize OpenGL
    glViewport(0, 0, windowWidth, windowHeight);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    shaderProgram = createProgram(vertexSource, fragmentSource);

    // Set up vertex data
    GLfloat vertices[] = {
        -1.0f, -1.0f, // Bottom-left
         1.0f, -1.0f, // Bottom-right
        -1.0f,  1.0f, // Top-left
         1.0f,  1.0f, // Top-right
    };

    // Generate VBO
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    // No need to unbind here

    // Start the main loop
    emscripten_set_main_loop(mainLoop, 0, 1);

    // Clean up (unreachable in Emscripten)
    SDL_GL_DeleteContext(glContext);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
