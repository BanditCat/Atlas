////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4 et al.   //
////////////////////////////////////////////////////////////////////////////////


#ifndef PROGRAM_H_INCLUDED
#define PROGRAM_H_INCLUDED

#define initSize 256
#include "trie.h"


// To add a command: add it in runProgram in program.c, here, and in the addStep parser in program.c.
typedef struct{
  enum{
    COMPUTE,
    IF,
    IFN,
    TRANSPOSE,
    SLICE,
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
    GETINPUT,  // Three axis and three buttons
    SET,
    GET,
    WINDOWSIZE
  } type;
  union{
    tensor* tensor;
    struct{
      u32 argCount;
      char* glsl;
    } toCompute;
    struct{
      union{
	char* name;
	u32 index;
      };
      u32 size;
    } var;
    u32 compute;
    u32 branch;
    char* branchName;
    char* varName;
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
  trieNode* vars;
  u32 numVars;
  char** varNames;
  u32* varOffsets;
  u32* varSizes;
  f32* varBlock;  
} program;

program* newProgramFromString( const char* prog, u32 strlen );
program* newProgramFromFile( const char* file );
// Return false to exit program.
bool runProgram( tensorStack* ts, program* p );
void deleteProgram( program* p );

#endif // PROGRAM_H_INCLUDED
