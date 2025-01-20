////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4 et al.   //
////////////////////////////////////////////////////////////////////////////////


#ifndef PROGRAM_H_INCLUDED
#define PROGRAM_H_INCLUDED

#define initSize 256
#include "trie.h"


// To add a command: add it in runProgram in program.c, here, and in the addStep parser in program.c.
// Also add it to the documentation, docs/index.html.
typedef struct{
  enum{
    COMPUTE, // documented
    ADD, // documented
    SUB, // documented
    MUL, // documented
    DIV, // documented
    POW, // documented
    IF, // documented
    IFN, // documented
    TRANSPOSE, // documented
    SLICE, // documented
    LOAD, // documented
    MULTM, // documented
    REVERSE, // documented
    CAT, // documented
    FIRST, // documented
    LAST, // documented
    ENCLOSE, // documented
    EXTRUDE, // documented
    UNEXTRUDE, // documented
    KEYS, // documented
    PRINT, // documented
    TENSOR, // documented
    TOP, // documented
    DUP, // documented
    ROT, // documented
    TRANS, // documented
    PROJ, // documented
    LENGTH, // documented
    TIME, // documented
    REPEAT, // documented
    SHAPE, // documented
    QUIT, // documented
    CALL, // documented
    POP, // documented
    RETURN, // documented
    GETINPUT, // documented, three axes and three buttons
    SET, // documented
    GET, // documented
    TEXTURE, // documented
    WINDOWSIZE // documented
  } type;
  union{
    tensor* tensor;
    struct{
      u32 retCount; 
      u32 argCount;
      char* vglsl;
      char* glsl;
      char* vglslpre;
      char* glslpre;
      u32 channels;
      u32 vertCount;
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
    char* progName;
  };
  const char* filename;
  u32 linenum;
  u32 commandnum;
} step;

#define NUM_FILENAMES 256
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
  trieNode* bigvars;
  u32 numBigvars;
  char** bigvarNames;
  tensor** bigvarts;
  char** filenames;
  u32 numFilenames;
} program;

program* newProgramFromFile( const char* filename );
// Return false to exit program.
bool runProgram( tensorStack* ts, program** progp );
void deleteProgram( program* p );

#endif // PROGRAM_H_INCLUDED
