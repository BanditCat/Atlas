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
// This adds an compute statement to p and returns its index.
u32 addCompute( program* p, const char* glsl, u32 argCount ){
  if( p->numComputes >= p->computeStackSize ){
    p->computeStackSize *= 2;
    compute** tp = mem( p->computeStackSize, compute* );
    memcpy( tp, p->computes, sizeof( compute* ) * p->numComputes );
    unmem( p->computes ); p->computes = tp;
  }
  p->computes[ p->numComputes ] = makeCompute( glsl, argCount );
  return p->numComputes++;
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

  
  if( !strncmp( command, "l'", 2 ) ){ // Label
    char* starti = command + 2;
    char* endi = starti;
    while( *endi && *endi != '\'' )
      endi++;
    if( endi == starti )
      error( "Line %u, command %u: %s", linenum, commandnum, "Empty label." );
    if( *endi != '\'' )
      error( "Line %u, command %u: %s", linenum, commandnum, "Unmatched quote in label." );
    char* label = mem( 1 + endi - starti, char );
    memcpy( label, starti, endi - starti );
    label[ endi - starti ] = '\0';
    --p->numSteps;
    if( trieSearch( p->labels, label, NULL ) )
      error( "Line %u, command %u: duplicate label '%s'", linenum, commandnum, label );
    trieInsert( p->labels, label, p->numSteps );
    unmem( label );
    dbg( "Linenum %u commandnum %u: label: %s\n", linenum, commandnum, label );
    
    
  } else if( !strncmp( command, "if'", 3 ) ){ // If
    char* starti = command + 3;
    char* endi = starti;
    while( *endi && *endi != '\'' )
      endi++;
    if( endi == starti )
      error( "Line %u, command %u: %s", linenum, commandnum, "Empty if statement." );
    if( *endi != '\'' )
      error( "Line %u, command %u: %s", linenum, commandnum, "Unmatched quote in if statement." );
    char* branchName = mem( 1 + endi - starti, char );
    memcpy( branchName, starti, endi - starti );
    branchName[ endi - starti ] = '\0';
    curStep->type = IF;
    curStep->branchName = branchName;
    char* sizep = endi + 1;
    if( *sizep )
      error( "Line %u, command %u: %s", linenum, commandnum,
	     "Extra characters after if statement." );
    dbg( "Linenum %u commandnum %u: if to %s\n", linenum, commandnum, branchName );


  } else if( !strncmp( command, "c'", 2 ) ){ // Compute
    char* starti = command + 2;
    char* endi = starti;
    while( *endi && *endi != '\'' )
      endi++;
    if( endi == starti )
      error( "Line %u, command %u: %s", linenum, commandnum, "Empty compute statement." );
    if( *endi != '\'' )
      error( "Line %u, command %u: %s", linenum, commandnum, "Unmatched quote in compute statement." );
    char* comp = mem( 1 + endi - starti, char );
    memcpy( comp, starti, endi - starti );
    comp[ endi - starti ] = '\0';
    char* sizep = endi + 1;
    u32 argCount;
    int charsread;
    if( sscanf( sizep, "%u%n", &argCount, &charsread ) == 1 && !sizep[ charsread ] ){
      curStep->type = COMPUTE;
      curStep->compute = addCompute( p, comp, argCount );
      if( argCount > 4 )
	error( "%s", "Compute created with more than 4 arguments. The maximum is 4." );
      dbg( "Linenum %u commandnum %u: compute '%s' on %u arguments.\n", linenum, commandnum, comp, argCount );
      unmem( comp );
    } else
      error( "Line %u, command %u: %s", linenum, commandnum,
	     "Malformed compute statement." );

    
  } else if( !strcmp( command, "r" ) ){ // Reverse
    curStep->type = REVERSE;
    dbg( "Linenum %u commandnum %u: reverse\n", linenum, commandnum );

    
  } else if( !strcmp( command, "dup" ) ){ // Dup
    curStep->type = DUP;
    dbg( "Linenum %u commandnum %u: dup\n", linenum, commandnum );

    
  } else if( !strcmp( command, "size" ) ){ // Top
    curStep->type = TOP;
    dbg( "Linenum %u commandnum %u: top\n", linenum, commandnum );

    
  } else if( !strcmp( command, "t" ) ){ // Transpose
    curStep->type = TRANSPOSE;
    dbg( "Linenum %u commandnum %u: transpose\n", linenum, commandnum );

    
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
  ret->computes = mem( initSize, compute* );
  ret->numComputes = 0;
  ret->computeStackSize = initSize;
  ret->steps = mem( initSize, step );
  ret->numSteps = 0;
  ret->stepStackSize = initSize;
  ret->labels = newTrieNode( NULL, 0 );
    
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

  // After adding all steps now we can replace if statement branchNames with the label locations.
  for( u32 i = 0; i < ret->numSteps; ++i )
    if( ret->steps[ i ].type == IF ){
      u32 jumpTo;
      if( !trieSearch( ret->labels, ret->steps[ i ].branchName, &jumpTo ) )
	error( "If statement with unknown label %s.", ret->steps[ i ].branchName );
      unmem( ret->steps[ i ].branchName );
      ret->steps[ i ].branch = jumpTo;
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
  for( u32 i = 0; i < p->numComputes; ++i )
    deleteCompute( p->computes[ i ] );
  for( u32 i = 0; i < p->numSteps; ++i ){
    if( p->steps[ i ].type == TENSOR ){
      deleteTensor( p->steps[ i ].tensor );
      p->steps[ i ].tensor = NULL;
    }
    
  }
  deleteTrieNode( p->labels );
  unmem( p->computes );
  unmem( p->steps );
  unmem( p );
}
bool runProgram( tensorStack* ts, program* p ){
  for( u32 i = 0; i < p->numSteps; ++i ){
    //dbg( "Step %u", i );
    step* s = p->steps + i;
    switch( s->type ){
    case TENSOR:
      push( ts, copyTensor( s->tensor ) );
      break;
    case PRINT:
      printStack( ts );
      //dbg( "%s", "print" );
      break;
    case COMPUTE:
      if( !ts->size )
	error( "%s", "Attempt to run a compute statement with no shape parameter on the stack." );
      if( ts->stack[ ts->size - 1 ]->rank != 1 )
	error( "%s", "The shape for a compute was not a rank 1 tensor." );
      if( ts->stack[ ts->size - 1 ]->size > 4 )
	error( "%s", "The shape for an initilizer was more than 4 component." );
      tensorToHostMemory( ts->stack[ ts->size - 1 ] );
      u32 shape[ 4 ];
      u32 size = ts->stack[ ts->size - 1 ]->size;
      for( u32 i = 0; i < ts->stack[ ts->size - 1 ]->size; ++i )
	shape[ i ] = ts->stack[ ts->size - 1 ]->data[ i ];
      pop( ts );
      push( ts, newTensorInitialized( ts, size, shape,
				      p->computes[ s->compute ] ) );
      //dbg( "%s", "compute" );
      break;
    case REVERSE:
      if( !ts->size )
	error( "%s", "Attempt to reverse with no axis parameter on the stack." );
      if( ts->stack[ ts->size - 1 ]->rank )
	error( "%s", "Attempt to reverse a nonscalar axis parameter." );
      
      tensorToHostMemory( ts->stack[ ts->size - 1 ] );
      u32 axis = *( ts->stack[ ts->size - 1 ]->data + ts->stack[ ts->size - 1 ]->offset );
      pop( ts );
      tensorReverse( ts, ts->size - 1, axis );
      //dbg( "%s %u", "reverse", axis );
      break;
    case DUP:
      if( !ts->size )
	error( "%s", "Attempt to duplicate with no parameter on the stack." );
      if( ts->stack[ ts->size - 1 ]->rank )
	error( "%s", "Attempt to duplicate with a nonscalar parameter." );
      
      tensorToHostMemory( ts->stack[ ts->size - 1 ] );
      u32 dup = *( ts->stack[ ts->size - 1 ]->data + ts->stack[ ts->size - 1 ]->offset );
      pop( ts );
      if( dup > ts->size - 1 )
	error( "%s", "Attempt to duplicate past the end of the stack." );
      push( ts, copyTensor( ts->stack[ ( ts->size - 1 ) - dup ] ) );
      //dbg( "%s %u", "reverse", axis );
      break;
    case IF:
      if( !ts->size )
	error( "%s", "Attempt to if with no parameter on the stack." );
      if( ts->stack[ ts->size - 1 ]->rank )
	error( "%s", "Attempt to if with a non-scalar parameter." );
      
      tensorToHostMemory( ts->stack[ ts->size - 1 ] );
      f32 cond = *( ts->stack[ ts->size - 1 ]->data + ts->stack[ ts->size - 1 ]->offset );
      pop( ts );
      if( cond != 0.0 )
	i = s->branch - 1;
      //dbg( "%s %u", "if", axis );
      break;
    case TRANSPOSE:
      if( !ts->size )
	error( "%s", "Attempt to transpose with no axes parameter on the stack." );
      if( ts->stack[ ts->size - 1 ]->rank != 1 )
	error( "%s", "Attempt to transpose with a axes parameter not of rank 1." );

      u32 axis1 = *( ts->stack[ ts->size - 1 ]->data + ts->stack[ ts->size - 1 ]->offset );
      u32 axis2 = *( ts->stack[ ts->size - 1 ]->data + ts->stack[ ts->size - 1 ]->offset
		    + ts->stack[ ts->size - 1 ]->strides[ 0 ] );
      pop( ts );
      tensorTranspose( ts, ts->size - 1, axis1, axis2 );
      //dbg( "%s %u %u", "transpose", axis1, axis2 );
      break;
    case TOP:
      {
	f32* ssize = mem( 1, f32 );
	*ssize = ts->size;
	push( ts, newTensor( 0, NULL, ssize ) );
	//dbg( "%s %u %u", "transpose", axis1, axis2 );
	break;
      }
    case QUIT:
      //dbg( "%s", "exit" );
      return false;
    default:
      error( "%s", "Logic error in Atlas!" ); 
    }
  }
  return true;
}
