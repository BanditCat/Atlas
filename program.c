////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4 et al.   //
////////////////////////////////////////////////////////////////////////////////

#include "Atlas.h"

// This adds an initializer to p and returns its index.
u32 addInitializer( program* p, const char* glsl ){
  if( p->numInitializers >= p->initializerStackSize ){
    p->initializerStackSize *= 2;
    initializer** tp = mem( p->initializerStackSize, initializer* );
    memcpy( tp, p->initializers, sizeof( initializer* ) * p->numInitializers );
    unmem( p->initializers ); p->initializers = tp;
  }
  p->initializers[ p->numInitializers ] = makeInitializer( glsl );
  return p->numInitializers++;
}
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
void addStep( program* p, u32 linenum, u32 commandnum, char* command ){
  if( !command )
    return;
  trimWhitespace( &command );
  if( !*command )
    return;
  if( p->numSteps >= p->stepStackSize ){
    p->stepStackSize *= 2;
    step* tp = mem( p->stepStackSize, step );
    memcpy( tp, p->steps, sizeof( step ) * p->numSteps );
    unmem( p->steps ); p->steps = tp;
  }
  step* curStep = &( p->steps[ p->numSteps ] );
  ++p->numSteps;

  
  if( !strncmp( command, "i'", 2 ) ){ // Initializer
    char* starti = command + 2;
    char* endi = starti;
    while( *endi && *endi != '\'' )
      endi++;
    if( endi == starti )
      error( "Line %u, command %u: %s", linenum, commandnum, "Empty initializer." );
    if( *endi != '\'' )
      error( "Line %u, command %u: %s", linenum, commandnum, "Unmatched quote in initializer." );
    char* init = mem( 1 + endi - starti, char );
    memcpy( init, starti, endi - starti );
    init[ endi - starti ] = '\0';
    curStep->type = INIT;
    curStep->init.initializer = addInitializer( p, init );
    char* sizep = endi + 1;
    curStep->init.rank = 0;
    while( *sizep ){
      u32 charsread = 0;
      int sret = sscanf( sizep, "%u%n", curStep->init.shape + curStep->init.rank, &charsread );
      if( sret == EOF )
	error( "Line %u, command %u: %s", linenum, commandnum,
	       "Failed to parse tensor size in initializer." );
      sizep += charsread;
      ++curStep->init.rank;
      if( !sret )
	break;
    }
    if( *sizep )
      error( "Line %u, command %u: %s", linenum, commandnum,
	     "Extra characters after tensor size in initializer." );
    for( u32 i = curStep->init.rank; i < 4; ++i )
      curStep->init.shape[ i ] = 1;
    dbg( "Linenum %u commandnum %u: %s       %u\n", linenum, commandnum, init, curStep->init.rank );
    unmem( init );

    
  } else if( !strncmp( command, "r ", 2 ) ){ // Reverse
    char* sizep = command + 2;
    u32 charsread = 0;
    u32 axis = 0;
    int sret = sscanf( sizep, "%u%n", &axis, &charsread );
    if( sret != 1 || !charsread )
      error( "Line %u, command %u: %s", linenum, commandnum,
	     "Failed to parse axis in reverse command." );
    if( sizep[ charsread ] )
      error( "Line %u, command %u: %s", linenum, commandnum,
	     "Extra characters after reverse command." );
    curStep->type = REVERSE;
    curStep->reverse.axis = axis;
    dbg( "Linenum %u commandnum %u: reverse %u\n", linenum, commandnum, curStep->reverse.axis );
     
  } else if( !strncmp( command, "t ", 2 ) ){ // Transpose
    char* sizep = command + 2;
    u32 charsread = 0;
    u32 axis1 = 0;
    u32 axis2 = 0;
    int sret = sscanf( sizep, "%u%u%n", &axis1, &axis2, &charsread );
    if( sret != 2 || !charsread )
      error( "Line %u, command %u: %s", linenum, commandnum,
	     "Failed to parse axes in transpose command." );
    if( sizep[ charsread ] )
      error( "Line %u, command %u: %s", linenum, commandnum,
	     "Extra characters after transpose command." );
    curStep->type = TRANSPOSE;
    curStep->transpose.axis1 = axis1;   
    curStep->transpose.axis2 = axis2;
    dbg( "Linenum %u commandnum %u: transpose %u %u\n", linenum, commandnum, curStep->transpose.axis1,
	 curStep->transpose.axis2 );
    
  } else if( !strcmp( command, "print" ) ){ // Print
    curStep->type = PRINT;
    dbg( "Linenum %u commandnum %u: print\n", linenum, commandnum );
  } else if( !strcmp( command, "quit" ) ){ // Print
    curStep->type = QUIT;
    dbg( "Linenum %u commandnum %u: quit\n", linenum, commandnum );
  } else{
    error( "Unknown command %s.", command );
  }
}
// Modifies prog.
program* newProgram( char* prog ){
  program* ret = mem( 1, program );
  ret->initializers = mem( initSize, initializer* );
  ret->numInitializers = 0;
  ret->initializerStackSize = initSize;
  ret->steps = mem( initSize, step );
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
    while( token ){
      addStep( ret, linenum, commandnum, token );
      ++commandnum;
      token = strtok( NULL, ";" );
    }
    ++linenum;
  }
  
  return ret;
}
// Doesn't modify prog.
program* newProgramFromString( const char* prog ){
  char* cp = mem( strlen( prog ) + 1, char );
  strcpy( cp, prog );
  program* ret = newProgram( cp );
  unmem( cp );
  return ret;
}
program* newProgramFromFile( const char* filename ){
  FILE* file = fopen( filename, "rb" );

  if( !file )
    error( "%s %s.", "Failed to open file", filename );
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
void runProgram( tensorStack* ts, program* p ){
  for( u32 i = 0; i < p->numSteps; ++i ){
    step* s = p->steps + i;
    switch( s->type ){
    case PRINT:
      printStack( ts );
      //dbg( "%s", "print" );
      break;
    case INIT:
      push( ts, newTensorInitialized( s->init.rank, s->init.shape,
				      p->initializers[ s->init.initializer ] ) );
      //dbg( "%s", "init" );
      break;
    case REVERSE:
      tensorReverse( ts, ts->top - 1, s->reverse.axis );
      //dbg( "%s", "reverse" );
      break;
    case TRANSPOSE:
      tensorTranspose( ts, ts->top - 1, s->transpose.axis1, s->transpose.axis2 );
      //dbg( "%s", "transpose" );
      break;
    case QUIT:
      exit( 0 );
      //dbg( "%s", "exit" );
      break;
    default:
      error( "%s", "Logic error in Atlas!" ); 
    }
  }
}
