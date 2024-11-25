////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4 et al.   //
////////////////////////////////////////////////////////////////////////////////


#define initSize 256

// To add a command: add it in runProgram in program.c, here, and in the addStep parser in program.c.
typedef struct{
  enum{
    INIT,
    TRANSPOSE,
    REVERSE,
    PRINT,
    TENSOR,
    QUIT
  } type;
  union{
    tensor* tensor;
    struct{
      u32 initializer;
      u32 rank;
      u32 shape[ 4 ];
    } init;
    struct{
      u32 axis1;
      u32 axis2;
    } transpose;
    struct{
      u32 axis;
    } reverse;
  };
} step;

typedef struct{
  initializer** initializers;
  u32 numInitializers;
  u32 initializerStackSize;
  step* steps;
  u32 numSteps;
  u32 stepStackSize;
} program;

program* newProgramFromString( const char* prog );
program* newProgramFromFile( const char* file );
// Return false to exit program.
bool runProgram( tensorStack* ts, program* p );
void deleteProgram( program* p );
