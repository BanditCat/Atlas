////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4 et al.   //
////////////////////////////////////////////////////////////////////////////////


#define initSize 256
#include "trie.h"


// TODO: recursive programs.  uniforms. elementwise basic ops
// To add a command: add it in runProgram in program.c, here, and in the addStep parser in program.c.
typedef struct{
  enum{
    COMPUTE,
    IF,
    TRANSPOSE,
    REVERSE,
    CAT,
    PRINT,
    TENSOR,
    TOP,
    DUP,
    QUIT,
    CALL,
    POP,
    RETURN,
    WINDOWSIZE
  } type;
  union{
    tensor* tensor;
    u32 compute;
    u32 branch;
    char* branchName;
  };
} step;

typedef struct{
  compute** computes;
  u32 numComputes;
  u32 computeStackSize;
  step* steps;
  trieNode* labels;
  u32 numSteps;
  u32 stepStackSize;
  u32* returns;
  u32 numReturns;
  u32 returnStackSize;
} program;

program* newProgramFromString( const char* prog );
program* newProgramFromFile( const char* file );
// Return false to exit program.
bool runProgram( tensorStack* ts, program* p );
void deleteProgram( program* p );
