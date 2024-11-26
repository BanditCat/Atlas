////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4 et al.   //
////////////////////////////////////////////////////////////////////////////////

#include "Atlas.h"


void skipWhitespace( const char** str ){
  while( isspace(**str) )
    (*str)++;
}
void parseTensorRecursive( const char** str, u32 currentDim, u32* shape, float* data, u32* dataIndex ){
  skipWhitespace( str );
  if( **str != '[' )
    error( "%s", "Expected '[' to start tensor definition." );
  (*str)++; 
  skipWhitespace( str );
  
  u32 dim_size = 0;
  while( **str != ']' && **str != '\0' ){
    if( **str == '[' ){
      if( currentDim + 1 >= 4 )
	error( "%s", "Tensor exceeds maximum supported dimensions (4D)." );
      parseTensorRecursive(str, currentDim + 1, shape, data, dataIndex);
      dim_size++;
    } else{
      float num;
      int charsread;
      if( sscanf( *str, "%f%n", &num, &charsread ) != 1 )
	error( "%s", "Failed to parse number in tensor." );
      *str += charsread;
      data[*dataIndex] = num;
      (*dataIndex)++;
      dim_size++;
    }
    
    skipWhitespace( str );
  }
  
  if( **str != ']' )
    error( "%s", "Expected ']' to close tensor definition." );
  
  (*str)++; 
  skipWhitespace( str );
  
  // Update shape
  if( !shape[ currentDim ] )
    shape[ currentDim ] = dim_size;
  else if (shape[ currentDim ] != dim_size )
    error( "%s", "Inconsistent tensor shape detected." );
}
// Function to determine shape
void determineShape( const char** s, u32 currentDim, u32* tempShape ){
  skipWhitespace( s );
  if( **s != '[' )
    error( "%s", "Expected '[' to start tensor definition." );
  (*s)++;
  skipWhitespace( s );
  
  u32 dim_size = 0;
  
  while( **s != ']' && **s != '\0' ){
    if( **s == '[' ){
      // Nested tensor
      determineShape( s, currentDim + 1, tempShape );
      dim_size++;
    } else {
      // Parse a number
      float num;
      int charsread;
      if( sscanf(*s, "%f%n", &num, &charsread ) != 1 )
	error( "%s", "Failed to parse number in tensor." );
      *s += charsread;
      dim_size++;
    }
    
    skipWhitespace( s );
  }
  
  if( **s != ']' )
    error( "%s", "Expected ']' to close tensor definition." );
  (*s)++; // Skip ']'
  skipWhitespace( s );
  
  // Update shape
  if( !tempShape[ currentDim ] )
    tempShape[ currentDim ] = dim_size;
  else if( tempShape[ currentDim ] != dim_size )
    error( "%s", "Inconsistent tensor shape detected." );
}

static tensor* parseTensor( const char* command ){
  u32 shape[ 4 ] = { 0, 0, 0, 0 }; // Initialize shape to zero
  float* tempData = NULL;
  u32 dataCount = 0;

  // To determine the number of elements, we'll need to parse the tensor
  // First, we make a pass to determine the shape and total elements
  
  // Clone the string to parse
  char* clone = mem( strlen( command ) + 1, char );
  strcpy( clone, command );
  const char* parsePtr = clone;
  
  u32 tempShape[ 4 ] = { 0, 0, 0, 0 };
  
  
  determineShape( &parsePtr, 0, tempShape );
  
  // Now, determine the rank by finding the deepest non-zero dimension
  u32 rank = 0;
  for( u32 i = 0; i < 4; ++i ){
      if( tempShape[ i ] > 0 )
	rank = i + 1;
  }
  
    // Validate that all deeper dimensions are set
  for( u32 i = 0; i < rank; ++i ){
    if( tempShape[i] == 0 )
      error( "%s", "Incomplete tensor shape definition." );
  }
  
  u32 totalElements = 1;
  for( u32 i = 0; i < rank; ++i )
    totalElements *= tempShape[i];
  
  tempData = mem( totalElements, f32 );
  
  parsePtr = clone;
  parseTensorRecursive( &parsePtr, 0, tempShape, tempData, &dataCount );
    
  if( dataCount != totalElements )
    error( "%s", "Mismatch in expected and actual number of tensor elements." );
  
  memcpy( shape, tempShape, sizeof( shape ) );
  
  unmem( clone );

  tensor* t = mem( 1, tensor );
  t->rank = rank;
  t->size = 1;
  for( u32 i = 0; i < 4; ++i )
    t->strides[ i ] = t->shape[ i ] = 1;
  for( int i = rank - 1; i >= 0; --i ){
    t->strides[ i ] = t->size;
    t->shape[ i ] = shape[ i ];
    t->size *= shape[ i ];
  }
  t->offset = 0;
  t->gpu = false;
  t->ownsData = true;
  t->data = tempData;

  return t;
}
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
    curStep->initializer = addInitializer( p, init );
    char* sizep = endi + 1;
    if( *sizep )
      error( "Line %u, command %u: %s", linenum, commandnum,
	     "Extra characters after initializer." );
    dbg( "Linenum %u commandnum %u: %s\n", linenum, commandnum, init );
    unmem( init );

    
  } else if( !strcmp( command, "r" ) ){ // Reverse
    curStep->type = REVERSE;
    dbg( "Linenum %u commandnum %u: reverse\n", linenum, commandnum );

    
  } else if( !strncmp( command, "t ", 2 ) ){ // Transpose
    char* sizep = command + 2;
    int charsread = 0;
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

    
  } else if( *command == '[' || *command == '.' || isdigit( *command ) ){ // A tensor
    curStep->type = TENSOR;
    char* tp = command;
    f32 scalar;
    int charsread;
    if( sscanf( tp, "%f%n", &scalar, &charsread ) == 1 && !tp[ charsread ] ){
      curStep->tensor = mem( 1, tensor );
      curStep->tensor->size = 1;
      for( u32 i = 0; i < 4; ++i )
	curStep->tensor->shape[ i ] = curStep->tensor->strides[ i ] = 1;
      curStep->tensor->data = mem( 1, f32 );
      *curStep->tensor->data = scalar;
      curStep->tensor->ownsData = true;
    }else
      curStep->tensor = parseTensor( command );
    dbg( "Linenum %u commandnum %u: tensor\n", linenum, commandnum );

    
  } else if( !strcmp( command, "print" ) ){ // Print
    curStep->type = PRINT;
    dbg( "Linenum %u commandnum %u: print\n", linenum, commandnum );

    
  } else if( !strcmp( command, "quit" ) ){ // Quit
    curStep->type = QUIT;
    dbg( "Linenum %u commandnum %u: quit\n", linenum, commandnum );


  } else{ // Error
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
  for( u32 i = 0; i < p->numSteps; ++i ){
    if( p->steps[ i ].type == TENSOR ){
      deleteTensor( p->steps[ i ].tensor );
      p->steps[ i ].tensor = NULL;
    }
    
  }
  unmem( p->initializers );
  unmem( p->steps );
  unmem( p );
}
bool runProgram( tensorStack* ts, program* p ){
  for( u32 i = 0; i < p->numSteps; ++i ){
    step* s = p->steps + i;
    switch( s->type ){
    case TENSOR:
      push( ts, copyTensor( s->tensor ) );
      break;
    case PRINT:
      printStack( ts );
      
      //dbg( "%s", "print" );
      break;
    case INIT:
      if( !ts->top )
	error( "%s", "Attempt to run an initializer with no shape parameter on the stack." );
      if( ts->stack[ ts->top - 1 ]->rank != 1 )
	error( "%s", "The shape for an initilizer was not a rank 1 tensor." );
      if( ts->stack[ ts->top - 1 ]->size > 4 )
	error( "%s", "The shape for an initilizer was more than 4 component." );
      tensorToHostMemory( ts->stack[ ts->top - 1 ] );
      u32 shape[ 4 ];
      u32 size = ts->stack[ ts->top - 1 ]->size;
      for( u32 i = 0; i < ts->stack[ ts->top - 1 ]->size; ++i )
	shape[ i ] = ts->stack[ ts->top - 1 ]->data[ i ];
      pop( ts );
      push( ts, newTensorInitialized( size, shape,
				      p->initializers[ s->initializer ] ) );
      //dbg( "%s", "init" );
      break;
    case REVERSE:
      if( !ts->top )
	error( "%s", "Attempt to reverse with no axis parameter on the stack." );
      if( ts->stack[ ts->top - 1 ]->rank )
	error( "%s", "Attempt to reverse a nonscalar axis parameter." );
      
      tensorToHostMemory( ts->stack[ ts->top - 1 ] );
      u32 axis = *( ts->stack[ ts->top - 1 ]->data + ts->stack[ ts->top - 1 ]->offset );
      pop( ts );
      tensorReverse( ts, ts->top - 1, axis );
      //dbg( "%s %u", "reverse", axis );
      break;
    case TRANSPOSE:
      tensorTranspose( ts, ts->top - 1, s->transpose.axis1, s->transpose.axis2 );
      //dbg( "%s", "transpose" );
      break;
    case QUIT:
      //dbg( "%s", "exit" );
      return false;
    default:
      error( "%s", "Logic error in Atlas!" ); 
    }
  }
  return true;
}
