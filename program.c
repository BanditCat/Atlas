////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4 et al.   //
////////////////////////////////////////////////////////////////////////////////

#include "Atlas.h"

/* typedef struct { */
/*   initializer** initializers; */
/*   u32 numInitializers; */
/*   u32 initializerStackSize; */
/*   void* (*steps)( void* ); */
/*   u32 numSteps; */
/*   u32 stepStackSize; */
/* } program; */
char* getNextLine(char** str) {
    if (*str == NULL || **str == '\0') return NULL;

    char* start = *str;
    char* end = strchr(start, '\n');

    if (end != NULL) {
        *end = '\0'; // Replace '\n' with '\0'
        *str = end + 1; // Move to the next line
    } else {
        *str = NULL; // No more lines
    }

    return start;
}
void trimWhitespace( char** str ){
  char* start = *str;
  char* end;
  
  while( isspace( *start ) )
    start++;
  
  if( *start == '\0' ){
    *str = start;
    return;
  }
  
  end = start + strlen( start ) - 1;
  
  while( end > start && isspace( *end ) ){
    *end = '\0';
    --end;
  }
  
  *str = start;
}
void addStep( u32 linenum, u32 commandnum, char* command ){
  trimWhitespace( &command );
  dbg( "Linenum %u commandnum %u: %s\n", linenum, commandnum, command );
}
// Modifies prog.
program* newProgram( char* prog ){
  program* ret = mem( 1, program );
  ret->initializers = mem( initSize, initializer* );
  ret->numInitializers = 0;
  ret->initializerStackSize = initSize;
  ret->steps = mem( initSize, void* (*)( void* ) );
  ret->numSteps = 0;
  ret->stepStackSize = initSize;
  
  char* ptr = prog;
  u32 linenum = 1;
  char* line;
  while( (line = getNextLine( &ptr )) ){
    char* comment = strstr( line, "//" );
    if( comment ){
      *comment = '\0';
      comment += 2;
    }
    u32 commandnum = 1;
    char* token = strtok( line, ";" );
    addStep( linenum, commandnum, token );
    while( (token = strtok( NULL, ":" )) ){
      ++commandnum;
      addStep( linenum, commandnum, token );
    }
    ++linenum;
  }
  
  return ret;
}
program* newProgramFromFile( const char* filename ){
  FILE* file = fopen( filename, "rb" );
  if( !file )
    error( "%s", "Failed to open file." );
  // Seek to the end of the file to determine its size
  if( fseek(file, 0, SEEK_END ) ){
    fclose( file );
    error( "%s", "Failed to seek file." );
  }
  long fileSize = ftell( file );
  if( fileSize == -1 ){
    fclose( file );
    error( "%s", "Failed to get file size." );
  }
  fseek( file, 0, SEEK_SET );
  char* buffer = mem( fileSize + 10, char );
  // Read the file contents into the buffer
  size_t bytesRead = fread( buffer, 1, fileSize, file );
  if( bytesRead != fileSize ){
    unmem( buffer );
    fclose( file );
    error( "%s", "Failed to read file." );
  }

  buffer[ fileSize ] = '\0';
  fclose( file );

  program* ret = newProgram( buffer );
  unmem( buffer );
  return ret;
}
void deleteProgram( program* p ){
  for( u32 i = 0; i < p->numInitializers; ++i )
    deleteInitializer( p->initializers[ i ] );
  unmem( p->initializers );
  unmem( p->steps );
  unmem( p );
}

