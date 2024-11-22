////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4 et al.   //
////////////////////////////////////////////////////////////////////////////////


#define initSize 256

typedef struct {
  initializer** initializers;
  u32 numInitializers;
  u32 initializerStackSize;
  void* (*steps)( void* );
  u32 numSteps;
  u32 stepStackSize;
} program;

program* newProgram( char* prog );
program* newProgramFromFile( const char* file );
void deleteProgram( program* p );
