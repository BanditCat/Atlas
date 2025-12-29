////////////////////////////////////////////////////////////////////////////////
// Copyright © 2025 Jon DuBois. Written with the assistance of GPT-4 et al.   //
////////////////////////////////////////////////////////////////////////////////


#ifndef PROGRAM_H_INCLUDED
#define PROGRAM_H_INCLUDED

#define initSize 256
#include "trie.h"


// To add a command: add it in runProgram in program.c, here, and in the addStep parser in program.c.
// Also add it to the documentation, docs/index.html.
typedef struct{
  enum{
    INDEX, // documented
    SORT, // documented
    CLS, // documented
    COMPUTE, // documented
    CONTINUE, // documented
    ADD, // documented
    SUB, // documented
    MUL, // documented
    DIV, // documented
    MOD, // documented
    POW, // documented
    LOG, // documented
    SIN, // documented
    COS, // documented
    FLOOR, // documented
    CEIL, // documented
    MAX, // documented
    MIN, // documented
    ATAN, // documented
    GREATERTHAN, // documented
    EQUALS, // documented
    MINMAX, // documented
    GLTF, // documented
    TOSTRING, // documented
    BURY, // documented
    RAISE, // documented
    BACKFACE, // documented
    DEPTH, // documented
    ADDITIVE, // documented
    IF, // documented
    IFN, // documented
    TRANSPOSE, // documented
    SLICE, // documented
    LOAD, // documented
    EVAL, // documented
    FULLSCREEN, // documented
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
    PRINTLINE, // documented
    PRINTSTRING, // documented
    KETTLE, // documented
    UNKETTLE, // documented
    TENSOR, // documented
    TEXTBUFFERVIEW, // documented
    TOP, // documented
    DUP, // documented
    ROT, // documented
    TRANS, // documented
    PROJ, // documented
    ORTHO, // documented
    LENGTH, // documented
    TIME, // documented
    TIMEDELTA, // documented
    REPEAT, // documented
    SHAPE, // documented
    RESHAPE, // documented
    QUIT, // documented
    CALL, // documented
    POP, // documented
    RETURN, // documented
    GAMEPAD, // documented, 6 axis, 15 buttons. [throttleLeft throttleRight leftStickX leftStickY rightStickX rightStickY leftShoulder rightShoulder home up right down left select start a b x y leftStick rightStick]
    GAMEPADRUMBLE, // documented
    GETINPUT, // documented, three axes and three buttons
    TEXTINPUT, // documented 
    SET, // documented
    GET, // documented
    TEXTURE, // documented
    TEXTUREARRAY, // documented
    WINDOWSIZE, // documented
    WORKSPACE, // documented
    TRANSFERSTART, // documented
    TRANSFEREND // documented
  } type;
  union{
    tensor* tensor;
    struct{
      tensor* verts;
      tensor* indices;
      tensor* bones;
      tensor* material;
    } gltf;
    struct{
      u32 retCount; 
      u32 argCount;
      char* vglsl;
      char* glsl;
      char* vglslpre;
      char* glslpre;
      u32 channels;
      u32 vertCount;
      bool reuse;
    } toCompute;
    struct{
      struct{
        char* name;
        char* baseName;
      };
      u32 index;
      u32 size;
    } var;
    u32 compute;
    struct{
      u32 branch;
      char* branchName;
      char* branchBaseName;
    };
    char* varName;
    char* progName;
  };
  const char* filename;
  u32 linenum;
  u32 commandnum;
} step;

#define NUM_FILENAMES 65536
typedef struct{
  char* mainFilename;
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

char* newProgramFromFile( const char* filename, program** ret );
// mutates but does not deallocate the string eval.
char* copyProgramWithEval( program* p, const char* eval, u32* startStep, program** ret );
// Return false to exit program.
char* runProgram( tensorStack* ts, program** progp, u32 startstep, bool* ret );
void deleteProgram( program* p );

#endif // PROGRAM_H_INCLUDED
