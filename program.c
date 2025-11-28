////////////////////////////////////////////////////////////////////////////////
// Copyright © 2025 Jon DuBois. Written with the assistance of GPT-4 et al.   //
////////////////////////////////////////////////////////////////////////////////

#include "Atlas.h"

void preprocessComputeCommands( char* prog ){
  char* ptr = prog;
  while( *ptr != '\0' ){
    if( ptr[ 0 ] == 'c' && ptr[ 1 ] == '\'' &&
        ( ptr == prog || ptr[ -1 ] == ';' || isspace( ptr[ -1 ] ) ) ){
      ptr += 2;
      
      // Process exactly 4 quoted sections
      for( int section = 0; section < 4; ++section ){
        // Replace all ; with \ in this section (until we hit the closing ')
        while( *ptr != '\0' && *ptr != '\'' ){
          if( *ptr == '\\' )
            error( "%s", "Backslash in shader! This is almost certainly an error!" );
          if( *ptr == ';' )
            *ptr = '\\';
          ptr++;
        }
        
        if( *ptr == '\'' )
          ptr++;
        else 
          break;
      }
    } else {
      ptr++;
    }
  }
}
void skipWhitespace( const char** str ){
  while( isspace( **str ) )
    ( *str )++;
}
void parseTensorRecursive( const char** str, u32 currentDim, u32* shape, float* data, u32* dataIndex ){
  skipWhitespace( str );
  if( **str != '[' )
    error( "%s", "Expected '[' to start tensor definition." );
  ( *str )++;
  skipWhitespace( str );

  u32 dim_size = 0;
  while( **str != ']' && **str != '\0' ){
    if( **str == '[' ){
      if( currentDim + 1 >= 4 )
        error( "%s", "Tensor exceeds maximum supported dimensions (4D)." );
      parseTensorRecursive( str, currentDim + 1, shape, data, dataIndex );
      dim_size++;
    } else {
      float num;
      int charsread;
      if( sscanf( *str, "%f%n", &num, &charsread ) != 1 )
        error( "%s", "Failed to parse number in tensor." );
      *str += charsread;
      data[ *dataIndex ] = num;
      ( *dataIndex )++;
      dim_size++;
    }

    skipWhitespace( str );
  }

  if( **str != ']' )
    error( "%s", "Expected ']' to close tensor definition." );

  ( *str )++;
  skipWhitespace( str );

  // Update shape
  if( !shape[ currentDim ] )
    shape[ currentDim ] = dim_size;
  else if( shape[ currentDim ] != dim_size )
    error( "%s", "Inconsistent tensor shape detected." );
}
// Function to determine shape
void determineShape( const char** s, u32 currentDim, u32* tempShape ){
  skipWhitespace( s );
  if( **s != '[' )
    error( "%s", "Expected '[' to start tensor definition." );
  ( *s )++;
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
      if( sscanf( *s, "%f%n", &num, &charsread ) != 1 )
        error( "%s", "Failed to parse number in tensor." );
      *s += charsread;
      dim_size++;
    }

    skipWhitespace( s );
  }

  if( **s != ']' )
    error( "%s", "Expected ']' to close tensor definition." );
  ( *s )++;  // Skip ']'
  skipWhitespace( s );

  // Update shape
  if( !tempShape[ currentDim ] )
    tempShape[ currentDim ] = dim_size;
  else if( tempShape[ currentDim ] != dim_size )
    error( "%s", "Inconsistent tensor shape detected." );
}

static tensor* parseTensor( const char* command ){
  u32 shape[ 4 ] = { 0, 0, 0, 0 };  // Initialize shape to zero
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
    if( tempShape[ i ] == 0 )
      error( "%s", "Incomplete tensor shape definition." );
  }

  u32 totalElements = 1;
  for( u32 i = 0; i < rank; ++i )
    totalElements *= tempShape[ i ];

  tempData = mem( totalElements, f32 );

  parsePtr = clone;
  parseTensorRecursive( &parsePtr, 0, tempShape, tempData, &dataCount );

  skipWhitespace( &parsePtr );
  if( *parsePtr != '\0' )
    error( "%s", "Unexpected characters after tensor definition." );

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
// Function to remove all '//' comments from the program string
void removeComments( char* prog ){
  char* src = prog;
  char* dst = prog;

  while( *src != '\0' ){
    if( src[ 0 ] == '/' && src[ 1 ] == '/' ){
      // Skip characters until the end of the line
      src += 2;
      while( *src != '\n' && *src != '\0' ){
        src++;
      }
    } else {
      *dst++ = *src++;
    }
  }
  *dst = '\0';  // Null-terminate the modified string
}
// This adds a compute statement to p and returns its index.
u32 addCompute( const char* filename,
                u32 linenum,
                u32 commandnum,
                program* p,
                const char* uniforms,
                const char* vglslpre,
                const char* glslpre,
                const char* vglsl,
                const char* glsl,
                u32 argCount,
                u32 retCount,
                u32 channels,
                bool reuse ){
  if( p->numComputes >= p->computeStackSize ){
    p->computeStackSize *= 2;
    compute** tp = mem( p->computeStackSize, compute* );
    memcpy( tp, p->computes, sizeof( compute* ) * p->numComputes );
    unmem( p->computes );
    p->computes = tp;
  }
  p->computes[ p->numComputes ] =
    makeCompute( filename,
                 linenum,
                 commandnum,
                 p,
                 uniforms, vglslpre, glslpre, vglsl, glsl, argCount, retCount, channels, reuse );
  return p->numComputes++;
}
char* getNextLine( char** str ){
  if( *str == NULL || **str == '\0' )
    return NULL;

  char* start = *str;
  char* end = strchr( start, '\n' );

  if( end != NULL ){
    *end = '\0';     // Replace '\n' with '\0'
    *str = end + 1;  // Move to the next line
  } else {
    *str = NULL;  // No more lines
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
void addStep( program* p, const char* filename, u32 linenum, u32 commandnum, char* command ){
  if( !command )
    return;
  trimWhitespace( &command );
  if( !*command )
    return;
  if( p->numSteps >= p->stepStackSize ){
    p->stepStackSize *= 2;
    step* tp = mem( p->stepStackSize, step );
    memcpy( tp, p->steps, sizeof( step ) * p->numSteps );
    unmem( p->steps );
    p->steps = tp;
  }
  step* curStep = &( p->steps[ p->numSteps ] );
  curStep->filename = filename;
  curStep->linenum = linenum;
  curStep->commandnum = commandnum;
  ++p->numSteps;

  if( !strncmp( command, "workspace'", 10 ) ){  // Workspace
    char* starti = command + 10;
    char* endi = starti;
    while( *endi && *endi != '\'' )
      endi++;
    if( *endi != '\'' )
      error( "%s:%u command %u: %s", filename,
             linenum,
             commandnum,
             "Unmatched quote in workspace command." );
    char* work = mem( 2 + endi - starti, char );
    memcpy( work, starti, endi - starti );
    work[ endi - starti ] = '\0';
    if( *( endi + 1 ) )
      error( "%s:%u command %u: %s", filename,
             linenum,
             commandnum,
             "Extra characters after workspace command." );
    unmem( workspace );
    workspace = work;
    --p->numSteps;
  } else if( !strncmp( command, "l'", 2 ) ){  // Label
    char* starti = command + 2;
    char* endi = starti;
    while( *endi && *endi != '\'' )
      endi++;
    if( endi == starti )
      error( "%s:%u command %u: %s", filename, linenum, commandnum, "Empty label." );
    if( *endi != '\'' )
      error( "%s:%u command %u: %s", filename,
             linenum,
             commandnum,
             "Unmatched quote in label." );
    u32 worklen = strlen( workspace );
    char* label = mem( worklen + 2 + endi - starti, char );
    if( worklen ){
      memcpy( label, workspace, worklen );
      label[ worklen ] = '.';
      ++worklen;
    }
    memcpy( label + worklen, starti, endi - starti );
    label[ worklen + endi - starti ] = '\0';
    --p->numSteps;
    if( *( endi + 1 ) )
      error( "%s:%u command %u: %s", filename,
             linenum,
             commandnum,
             "Extra characters after label." );
    if( trieSearch( p->labels, label, NULL ) )
      error( "%s:%u command %u: duplicate label '%s'", filename,
             linenum,
             commandnum,
             label );
    trieInsert( p->labels, label, p->numSteps );
    // dbg( "Linenum %u commandnum %u: label: %s\n", linenum, commandnum, label
    // );
    unmem( label );

  } else if( !strncmp( command, "set'", 4 ) ){  // Set
    char* starti = command + 4;
    char* endi = starti;
    while( *endi && *endi != '\'' )
      endi++;
    if( endi == starti )
      error( "%s:%u command %u: %s", filename,
             linenum,
             commandnum,
             "Empty name in set statement." );
    if( *endi != '\'' )
      error( "%s:%u command %u: %s", filename,
             linenum,
             commandnum,
             "Unmatched quote in set statement." );
    u32 worklen = strlen( workspace );
    char* varName = mem( worklen + 2 + endi - starti, char );
    if( worklen ){
      memcpy( varName, workspace, worklen );
      varName[ worklen ] = '.';
      ++worklen;
    }
    memcpy( varName + worklen, starti, endi - starti );
    varName[ worklen + endi - starti ] = '\0';
    char* sizep = endi + 1;
    u32 varSize;
    int charsread;
    curStep->type = SET;
    curStep->var.name = varName;
    if( !*sizep ){
      curStep->var.size = 0;
    }else if( sscanf( sizep, "%u%n", &varSize, &charsread ) == 1 &&
              !sizep[ charsread ] ){
      curStep->var.size = varSize;
      if( !varSize || ( varSize > 4 && varSize != 16 ) )
        error( "%s", "Invalid var size in set statement." );
      // dbg( "Linenum %u commandnum %u: set '%s' of size %u.\n",
      // linenum, commandnum, varName, varSize );
    } else
      error( "%s:%u command %u: %s", filename,
             linenum,
             commandnum,
             "Malformed set statement." );
    // dbg( "Linenum %u commandnum %u: set var %s\n", linenum, commandnum,
    // varName );
    if( worklen )
      curStep->var.baseName = varName + worklen;

  } else if( !strncmp( command, "get'", 4 ) ){  // Get
    char* starti = command + 4;
    char* endi = starti;
    while( *endi && *endi != '\'' )
      endi++;
    if( endi == starti )
      error( "%s:%u command %u: %s", filename,
             linenum,
             commandnum,
             "Empty name in get statement." );
    if( *endi != '\'' )
      error( "%s:%u command %u: %s", filename,
             linenum,
             commandnum,
             "Unmatched quote in get statement." );
    u32 worklen = strlen( workspace );
    char* varName = mem( worklen + 2 + endi - starti, char );
    if( worklen ){
      memcpy( varName, workspace, worklen );
      varName[ worklen ] = '.';
      ++worklen;
    }
    memcpy( varName + worklen, starti, endi - starti );
    varName[ worklen + endi - starti ] = '\0';
    if( *( endi + 1 ) )
      error( "%s:%u command %u: %s", filename,
             linenum,
             commandnum,
             "Extra characters after get statement." );
    curStep->type = GET;
    curStep->var.name = varName;
    if( worklen )
      curStep->var.baseName = varName + worklen;
    // dbg( "Linenum %u commandnum %u: get var %s\n", linenum, commandnum,
    // varName );

  } else if( !strncmp( command, "if'", 3 ) ){  // If
    char* starti = command + 3;
    char* endi = starti;
    while( *endi && *endi != '\'' )
      endi++;
    if( endi == starti )
      error(
            "%s:%u command %u: %s", filename, linenum, commandnum, "Empty if statement." );
    if( *endi != '\'' )
      error( "%s:%u command %u: %s", filename,
             linenum,
             commandnum,
             "Unmatched quote in if statement." );
    u32 worklen = strlen( workspace );
    char* branchName = mem( worklen + 2 + endi - starti, char );
    if( worklen ){
      memcpy( branchName, workspace, worklen );
      branchName[ worklen ] = '.';
      ++worklen;
    }
    memcpy( branchName + worklen, starti, endi - starti );
    branchName[ worklen + endi - starti ] = '\0';
    curStep->type = IF;
    curStep->branchName = branchName;
    if( *( endi + 1 ) )
      error( "%s:%u command %u: %s", filename,
             linenum,
             commandnum,
             "Extra characters after if statement." );
    // dbg( "Linenum %u commandnum %u: if to %s\n", linenum, commandnum,
    // branchName );
    if( worklen )
      curStep->branchBaseName = branchName + worklen;

  } else if( !strncmp( command, "ifn'", 4 ) ){  // Ifn
    char* starti = command + 4;
    char* endi = starti;
    while( *endi && *endi != '\'' )
      endi++;
    if( endi == starti )
      error( "%s:%u command %u: %s", filename,
             linenum,
             commandnum,
             "Empty ifn statement." );
    if( *endi != '\'' )
      error( "%s:%u command %u: %s", filename,
             linenum,
             commandnum,
             "Unmatched quote in ifn statement." );
    u32 worklen = strlen( workspace );
    char* branchName = mem( worklen + 2 + endi - starti, char );
    if( worklen ){
      memcpy( branchName, workspace, worklen );
      branchName[ worklen ] = '.';
      ++worklen;
    }
    memcpy( branchName + worklen, starti, endi - starti );
    branchName[ worklen + endi - starti ] = '\0';
    curStep->type = IFN;
    curStep->branchName = branchName;
    if( *( endi + 1 ) )
      error( "%s:%u command %u: %s", filename,
             linenum,
             commandnum,
             "Extra characters after ifn statement." );
    // dbg( "Linenum %u commandnum %u: ifn to %s\n", linenum, commandnum,
    // branchName );
    if( worklen )
      curStep->branchBaseName = branchName + worklen;

  } else if( !strncmp( command, "img'", 4 ) ){  // img
    char* starti = command + 4;
    char* endi = starti;
    while( *endi && *endi != '\'' )
      endi++;
    if( endi == starti )
      error( "%s:%u command %u: %s", filename,
             linenum,
             commandnum,
             "Empty img statement." );
    if( *endi != '\'' )
      error( "%s:%u command %u: %s", filename,
             linenum,
             commandnum,
             "Unmatched quote in img statement." );
    char* imgName = mem( 1 + endi - starti, char );
    memcpy( imgName, starti, endi - starti );
    imgName[ endi - starti ] = '\0';
    curStep->type = TENSOR;
    curStep->tensor = tensorFromImageFile( imgName );
    unmem( imgName );
    if( *( endi + 1 ) )
      error( "%s:%u command %u: %s", filename,
             linenum,
             commandnum,
             "Extra characters after img statement." );
    // dbg( "Linenum %u commandnum %u: img %s\n", linenum, commandnum,
    // imgName );

  }else if( !strcmp( command, "load" ) ){
    curStep->type = LOAD;
    curStep->progName = NULL;    
  }else if( !strncmp( command, "load'", 5 ) ){  // load
    char* starti = command + 5;
    char* endi = starti;
    while( *endi && *endi != '\'' )
      endi++;
    if( endi == starti )
      error( "%s:%u command %u: %s", filename,
             linenum,
             commandnum,
             "Empty load statement." );
    if( *endi != '\'' )
      error( "%s:%u command %u: %s", filename,
             linenum,
             commandnum,
             "Unmatched quote in load statement." );
    char* progName = mem( 1 + endi - starti, char );
    memcpy( progName, starti, endi - starti );
    progName[ endi - starti ] = '\0';
    curStep->type = LOAD;
    curStep->progName = progName;
    if( *( endi + 1 ) )
      error( "%s:%u command %u: %s", filename,
             linenum,
             commandnum,
             "Extra characters after load statement." );
    // dbg( "Linenum %u commandnum %u: load %s\n", linenum, commandnum,
    // progName );

  } else if( !strncmp( command, "c'", 2 ) ){  // Compute
    char* starti = command + 2;
    char* endi = starti;
    while( *endi && *endi != '\'' )
      endi++;
    if( *endi != '\'' )
      error( "%s:%u command %u: %s", filename,
             linenum,
             commandnum,
             "Unmatched quote in compute statement vertex pre block." );
    char* vpre = mem( 1 + endi - starti, char );
    memcpy( vpre, starti, endi - starti );
    // Replace \ with ;
    for( u32 i = 0; i < endi - starti; ++i )
      if( vpre[ i ] == '\\' )
        vpre[ i ] = ';';
    vpre[ endi - starti ] = '\0';
    starti = endi + 1;
    ++endi; // now get compute statements
    while( *endi && *endi != '\'' )
      endi++;
    if( *endi != '\'' )
      error( "%s:%u command %u: %s", filename,
             linenum,
             commandnum,
             "Unmatched quote in compute statement vertex block." );
    char* vcomp = mem( 1 + endi - starti, char );
    memcpy( vcomp, starti, endi - starti );
    // Replace \ with ;
    for( u32 i = 0; i < endi - starti; ++i )
      if( vcomp[ i ] == '\\' )
        vcomp[ i ] = ';';
    vcomp[ endi - starti ] = '\0';
    starti = endi + 1;
    ++endi; // now get compute statements
    while( *endi && *endi != '\'' )
      endi++;
    if( *endi != '\'' )
      error( "%s:%u command %u: %s", filename,
             linenum,
             commandnum,
             "Unmatched quote in compute statement pre block." );
    char* pre = mem( 1 + endi - starti, char );
    memcpy( pre, starti, endi - starti );
    // Replace \ with ;
    for( u32 i = 0; i < endi - starti; ++i )
      if( pre[ i ] == '\\' )
        pre[ i ] = ';';
    pre[ endi - starti ] = '\0';
    starti = endi + 1;
    ++endi; // now get compute statements
    while( *endi && *endi != '\'' )
      endi++;
    if( *endi != '\'' )
      error( "%s:%u command %u: %s", filename,
             linenum,
             commandnum,
             "Unmatched quote in compute statement." );
    char* comp = mem( 1 + endi - starti, char );
    memcpy( comp, starti, endi - starti );
    // Replace \ with ;
    for( u32 i = 0; i < endi - starti; ++i )
      if( comp[ i ] == '\\' )
        comp[ i ] = ';';
    comp[ endi - starti ] = '\0';
    
    char* sizep = endi + 1;
    u32 argCount, retCount, channels, reuse;
    int charsread;
    if( sscanf( sizep, "%u%u%u%u%n", &argCount, &retCount, &channels, &reuse, &charsread ) == 4 &&
        !sizep[ charsread ] ){
      curStep->type = COMPUTE;
      curStep->toCompute.glslpre = pre;
      curStep->toCompute.glsl = comp;
      curStep->toCompute.vglslpre = vpre;
      curStep->toCompute.vglsl = vcomp;
      curStep->toCompute.retCount = retCount;
      curStep->toCompute.argCount = argCount;
      curStep->toCompute.channels = channels;
      curStep->toCompute.reuse = reuse;
      if( channels && channels != 4 && channels != 1 && channels != 10 && channels != 40 )
        error( "%s", "Compute created with channels not equal 0, 1, 4, 10, or 40." );
      if( argCount > 4 )
        error( "%s", "Compute created with more than 4 arguments. The maximum is 4." );
      if( retCount > 4 || !retCount )
        error( "%s", "Compute created with a bad return count, must be 1-4." );
      // dbg( "Linenum %u commandnum %u: compute '%s' on %u arguments.\n",
      // linenum, commandnum, comp, argCount );
    } else
      error( "%s:%u command %u: %s", filename,
             linenum,
             commandnum,
             "Malformed compute statement." );

  } else if( !strcmp( command, "first" ) ){
    curStep->type = FIRST;
    // dbg( "Linenum %u commandnum %u: first\n", linenum, commandnum );

  } else if( !strcmp( command, "last" ) ){
    curStep->type = LAST;
    // dbg( "Linenum %u commandnum %u: last\n", linenum, commandnum );

  } else if( !strcmp( command, "bury" ) ){
    curStep->type = BURY;
    // dbg( "Linenum %u commandnum %u: bury\n", linenum, commandnum );

  } else if( !strcmp( command, "raise" ) ){
    curStep->type = RAISE;
    // dbg( "Linenum %u commandnum %u: bury\n", linenum, commandnum );

  } else if( !strcmp( command, "backface" ) ){
    curStep->type = BACKFACE;
    // dbg( "Linenum %u commandnum %u: backface\n", linenum, commandnum );

  } else if( !strcmp( command, "gamepad" ) ){
    curStep->type = GAMEPAD;
    // dbg( "Linenum %u commandnum %u: last\n", linenum, commandnum );

  } else if( !strcmp( command, "toString" ) ){
    curStep->type = TOSTRING;
    // dbg( "Linenum %u commandnum %u: toString\n", linenum, commandnum );

  } else if( !strcmp( command, "additive" ) ){
    curStep->type = ADDITIVE;
    // dbg( "Linenum %u commandnum %u: additive\n", linenum, commandnum );

  } else if( !strcmp( command, "depth" ) ){
    curStep->type = DEPTH;
    // dbg( "Linenum %u commandnum %u: depth\n", linenum, commandnum );

  } else if( !strcmp( command, "unext" ) ){
    curStep->type = UNEXTRUDE;
    // dbg( "Linenum %u commandnum %u: unext\n", linenum, commandnum );

  } else if( !strcmp( command, "l" ) ){
    curStep->type = LENGTH;
    // dbg( "Linenum %u commandnum %u: length\n", linenum, commandnum );

  } else if( !strcmp( command, "proj" ) ){
    curStep->type = PROJ;
    // dbg( "Linenum %u commandnum %u: proj\n", linenum, commandnum );

  } else if( !strcmp( command, "ortho" ) ){
    curStep->type = ORTHO;
    // dbg( "Linenum %u commandnum %u: ortho\n", linenum, commandnum );

  } else if( !strcmp( command, "translate" ) ){
    curStep->type = TRANS;
    // dbg( "Linenum %u commandnum %u: translate\n", linenum, commandnum );

  } else if( !strcmp( command, "texture" ) ){
    curStep->type = TEXTURE;
    // dbg( "Linenum %u commandnum %u: first\n", linenum, commandnum );

  } else if( !strcmp( command, "m" ) ){
    curStep->type = MULTM;
    // dbg( "Linenum %u commandnum %u: multm\n", linenum, commandnum );

  } else if( !strcmp( command, "keys" ) ){
    curStep->type = KEYS;
    // dbg( "Linenum %u commandnum %u: last\n", linenum, commandnum );

  } else if( !strcmp( command, "+" ) ){
    curStep->type = ADD;
    // dbg( "Linenum %u commandnum %u: add\n", linenum, commandnum );

  } else if( !strcmp( command, "-" ) ){
    curStep->type = SUB;
    // dbg( "Linenum %u commandnum %u: sub\n", linenum, commandnum );

  } else if( !strcmp( command, "*" ) ){
    curStep->type = MUL;
    // dbg( "Linenum %u commandnum %u: mul\n", linenum, commandnum );

  } else if( !strcmp( command, "/" ) ){
    curStep->type = DIV;
    // dbg( "Linenum %u commandnum %u: div\n", linenum, commandnum );

  } else if( !strcmp( command, "^" ) ){
    curStep->type = POW;
    // dbg( "Linenum %u commandnum %u: exp\n", linenum, commandnum );

  } else if( !strcmp( command, "sin" ) ){
    curStep->type = SIN;
    // dbg( "Linenum %u commandnum %u: sin\n", linenum, commandnum );

  } else if( !strcmp( command, "cos" ) ){
    curStep->type = COS;
    // dbg( "Linenum %u commandnum %u: cos\n", linenum, commandnum );

  } else if( !strcmp( command, "floor" ) ){
    curStep->type = FLOOR;
    // dbg( "Linenum %u commandnum %u: floor\n", linenum, commandnum );

  } else if( !strcmp( command, "ceil" ) ){
    curStep->type = CEIL;
    // dbg( "Linenum %u commandnum %u: ceil\n", linenum, commandnum );

  } else if( !strcmp( command, "minmax" ) ){
    curStep->type = MINMAX;
    // dbg( "Linenum %u commandnum %u: minmax\n", linenum, commandnum );

  } else if( !strcmp( command, "r" ) ){
    curStep->type = REVERSE;
    // dbg( "Linenum %u commandnum %u: reverse\n", linenum, commandnum );

  } else if( !strcmp( command, "timeDelta" ) ){
    curStep->type = TIME;
    // dbg( "Linenum %u commandnum %u: reverse\n", linenum, commandnum );

  } else if( !strcmp( command, "e" ) ){
    curStep->type = ENCLOSE;
    // dbg( "Linenum %u commandnum %u: enclose\n", linenum, commandnum );

  } else if( !strcmp( command, "ext" ) ){
    curStep->type = EXTRUDE;
    // dbg( "Linenum %u commandnum %u: extrude\n", linenum, commandnum );

  } else if( !strcmp( command, "cat" ) ){
    curStep->type = CAT;
    // dbg( "Linenum %u commandnum %u: cat\n", linenum, commandnum );

  } else if( !strcmp( command, "pop" ) ){
    curStep->type = POP;
    // dbg( "Linenum %u commandnum %u: pop\n", linenum, commandnum );

  } else if( !strcmp( command, "rep" ) ){
    curStep->type = REPEAT;
    // dbg( "Linenum %u commandnum %u: repeat\n", linenum, commandnum );

  } else if( !strcmp( command, "shape" ) ){
    curStep->type = SHAPE;
    // dbg( "Linenum %u commandnum %u: shape\n", linenum, commandnum );

  } else if( !strcmp( command, "dup" ) ){
    curStep->type = DUP;
    // dbg( "Linenum %u commandnum %u: dup\n", linenum, commandnum );

  } else if( !strcmp( command, "s" ) ){
    curStep->type = SLICE;
    // dbg( "Linenum %u commandnum %u: slice\n", linenum, commandnum );

  } else if( !strcmp( command, "size" ) ){
    curStep->type = TOP;
    // dbg( "Linenum %u commandnum %u: top\n", linenum, commandnum );

  } else if( !strcmp( command, "return" ) ){
    curStep->type = RETURN;
    // dbg( "Linenum %u commandnum %u: return\n", linenum, commandnum );

  } else if( !strcmp( command, "rot" ) ){
    curStep->type = ROT;
    // dbg( "Linenum %u commandnum %u: rotate\n", linenum, commandnum );

  } else if( !strcmp( command, "input" ) ){
    curStep->type = GETINPUT;
    // dbg( "Linenum %u commandnum %u: input\n", linenum, commandnum );

  } else if( !strcmp( command, "windowSize" ) ){
    curStep->type = WINDOWSIZE;
    // dbg( "Linenum %u commandnum %u: window size\n", linenum, commandnum );

  } else if( !strcmp( command, "t" ) ){
    curStep->type = TRANSPOSE;
    // dbg( "Linenum %u commandnum %u: transpose\n", linenum, commandnum );

  } else if( *command == '[' || isfloat( command ) || *command == '\'' ){
    curStep->type = TENSOR;
    if( *command == '\'' ){
      char* starti = command + 1;
      char* endi = starti;
      while (*endi) {
        if (*endi == '\'') {
          // Count backslashes immediately before this quote
          int bslashes = 0;
          char *p = endi;
          
          while( p > starti && p[-1] == '\\' ){
            ++bslashes;
            --p;
          }

          if( ( bslashes & 1 ) == 0 )
            break;
        }
        ++endi;
      }    
      
      if( endi == starti )
        error( "%s:%u command %u: %s", filename,
               linenum,
               commandnum,
               "Empty string statement." );
      if( *endi != '\'' )
        error( "%s:%u command %u: %s", filename,
               linenum,
               commandnum,
               "Unmatched quote in string statement." );
      char* str = mem( 1 + endi - starti, char );
      memcpy( str, starti, endi - starti );
      u32 back = endi - starti;
      
      // remove backslashes
      for( u32 i = 0; i < back; ++i )
        if( str[ i ] == '\\' ){
          --back;
          for( u32 j = i; j < back; ++j )
            str[ j ] = str[ j + 1 ];
        }
      str[ back ] = '\0';
      
      
      curStep->tensor = tensorFromString( str );
      unmem( str );
      if( *( endi + 1 ) )
        error( "%s:%u command %u: %s", filename,
               linenum,
               commandnum,
               "Extra characters after string statement." );
      // dbg( "Linenum %u commandnum %u: string %s\n", linenum, commandnum,
      // string );
    } else{
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
      } else
        curStep->tensor = parseTensor( command );
    }
    // dbg( "Linenum %u commandnum %u: tensor\n", linenum, commandnum );

  } else if( !strcmp( command, "print" ) ){
    curStep->type = PRINT;
    // dbg( "Linenum %u commandnum %u: print\n", linenum, commandnum );

  } else if( !strcmp( command, "quit" ) ){
    curStep->type = QUIT;
    // dbg( "Linenum %u commandnum %u: quit\n", linenum, commandnum );

  } else {  // Call, get or set.
    char* starti = command;
    char* endi = starti;
    while( *endi )
      endi++;
    if( endi == starti )
      error( "%s:%u command %u: %s", filename,
             linenum,
             commandnum,
             "Empty call statement." );
    u32 worklen = strlen( workspace );
    char* branchName = mem( worklen + 2 + endi - starti, char );
    if( worklen ){
      memcpy( branchName, workspace, worklen );
      branchName[ worklen ] = '.';
      ++worklen;
    }
    memcpy( branchName + worklen, starti, endi - starti );
    branchName[ worklen + endi - starti ] = '\0';
    curStep->type = CALL;
    curStep->branchName = branchName;
    // dbg( "Linenum %u commandnum %u: call to %s\n", linenum, commandnum,
    // branchName );
    if( worklen )
      curStep->branchBaseName = branchName + worklen;
  }
}
program* newProgram( void ){
  program* ret = mem( 1, program );
  ret->computes = mem( initSize, compute* );
  ret->numComputes = 0;
  ret->computeStackSize = initSize;
  ret->steps = mem( initSize, step );
  ret->numSteps = 0;
  ret->stepStackSize = initSize;
  ret->labels = newTrieNode( NULL, 0 );
  ret->vars = newTrieNode( NULL, 0 );
  ret->bigvars = newTrieNode( NULL, 0 );
  ret->numReturns = 0;
  ret->returns = mem( initSize, step );
  ret->returnStackSize = initSize;
  ret->filenames = mem( NUM_FILENAMES, char* );
  return ret;
}
void addProgramFromFile( const char* filename, program* program );
// Modifies prog, adds all steps in prog to program.
void addProgram( const char* filename, char* prog, program* program ){
  removeComments( prog );
  preprocessComputeCommands( prog );
  
  
  char* ptr = prog;
  u32 linenum = 1;
  u32 commandnum = 0;

  while( *ptr != '\0' ){
    // Remember the start of this chunk.
    char* oldPtr = ptr;

    // Find the next semicolon (or end-of-string if none).
    char* semicolon = strchr( ptr, ';' );
    if( !semicolon ){
      semicolon = ptr + strlen( ptr );  // If no semicolon, go to end of input
    }

    // Calculate length of this “command” chunk
    size_t cmd_length = semicolon - ptr;

    // If the chunk is empty, check if we are on a newline, then skip
    if( cmd_length == 0 ){
      // We still want to count any newlines in the region from oldPtr to semicolon
      for( char* c = oldPtr; c < semicolon; c++ ){
        if( *c == '\n' ){
          linenum++;
          commandnum = 0;
        }
      }
      // Advance ptr beyond the semicolon if it exists
      if( *semicolon == ';' ){
        ptr = semicolon + 1;
        commandnum++;
      } else {
        ptr = semicolon;
      }
      continue;
    }

    // Extract the command into a temporary buffer
    char* buf = mem( cmd_length + 1, char );
    strncpy( buf, ptr, cmd_length );
    buf[ cmd_length ] = '\0';

    // Count newlines in [oldPtr .. semicolon) BEFORE we move ptr.
    for( char* c = oldPtr; c < semicolon; c++ ){
      if( *c == '\n' ){
        linenum++;
        commandnum = 0;
      }
    }

    // Now we can move ptr to the next chunk
    if( *semicolon == ';' ){
      ptr = semicolon + 1;  // Skip past the semicolon
      commandnum++;
    } else {
      ptr = semicolon;      // Or end-of-string
    }

    // Trim and parse the command, e.g. for “include”, “addStep”, etc.
    char* command = buf;
    trimWhitespace( &command );
    if( !strncmp( command, "include'", 8 ) ){
      char* starti = command + 8;
      char* endi = starti;
      while( *endi && *endi != '\'' )
        endi++;
      if( endi == starti )
        error( "%s:%u command %u: %s", filename,
               linenum,
               commandnum,
               "Empty include statement." );
      if( *endi != '\'' )
        error( "%s:%u command %u: %s", filename,
               linenum,
               commandnum,
               "Unmatched quote in include statement." );
      char* inc = mem( 1 + endi - starti, char );
      memcpy( inc, starti, endi - starti );
      inc[ endi - starti ] = '\0';
      addProgramFromFile( inc, program );
      // reset workspace.
      unmem( workspace );
      workspace = mem( 1, char );
      workspace[ 0 ] = 0;
      program->filenames[ program->numFilenames++ ] = inc;
      if( program->numFilenames == NUM_FILENAMES )
        error( "%s", "NUM_FILENAMES exceeded." );
      // ... handle 'include' ...
      // addProgramFromFile(...);
      unmem( buf );
    } else {
      // ... handle everything else ...
      addStep( program, filename, linenum, commandnum, command );
      unmem( buf );
    }
  }
}


void finalize( program* program ){
  // Collect variables and craft the uniform block and the program vars.
  char* glslUniformBlock;
  {
    program->numVars = 0;
    u32 nameslen = 0;
    u32 baselen = strlen( "uniform float %s;" ) + 30;
    // Check here for calls that are sets
    for( u32 i = 0; i < program->numSteps; ++i ){
      if( program->steps[ i ].type == CALL ){
        u32 len = strlen( program->steps[ i ].branchName );
        s64 back = len - 1;
        bool arg = false;
        u32 argloc = 0;
        u32 sz = 0;
        while( back >= 0 && isdigit( program->steps[ i ].branchName[ back ] ) ){
          arg = true;
          --back;
        }
        argloc = back + 1;
        while( back >= 0 && isspace( program->steps[ i ].branchName[ back ] ) )
          --back;
        if( back >= 0 && program->steps[ i ].branchName[ back ] == '=' ){
          if( arg ){
            int charsread;
            if( sscanf( program->steps[ i ].branchName + argloc, "%u%n",
                        &sz, &charsread ) != 1 )
              error( "%s", "sscanf failed!" );
          }

          if( sz > 4 && sz != 16 )
            error( "%s", "Invalid var size in short form set statement." );

          char* ns = mem( back + 2, char );
          memcpy( ns, program->steps[ i ].branchName, back );
          ns[ back ] = '\0';
          unmem( program->steps[ i ].branchName );
          program->steps[ i ].type = SET;
          program->steps[ i ].var.name = ns;
          program->steps[ i ].var.size = sz;
        }
      }
    }    
    for( u32 i = 0; i < program->numSteps; ++i )
      if( program->steps[ i ].type == SET ){
        if( program->steps[ i ].var.size ){
          nameslen += strlen( program->steps[ i ].var.name ) + 2;
          program->numVars++;
        }else
          program->numBigvars++;
      }
    program->varOffsets = mem( program->numVars, u32 );
    program->varSizes = mem( program->numVars, u32 );
    u32 bufsize = baselen * program->numVars + nameslen + 200;
    glslUniformBlock = mem( bufsize, u8 );
    char* p = glslUniformBlock;
    //    p += snprintf( p, bufsize - ( p - glslUniformBlock ), "layout(std140)
    //    uniform vars{\n" );
    program->varNames = mem( program->numVars, char* );
    program->bigvarNames = mem( program->numBigvars, char* );
    program->numVars = 0;
    program->numBigvars = 0;
    u32 offset = 0;
    for( u32 i = 0; i < program->numSteps; ++i ){
      if( program->steps[ i ].type == SET ){
        if( !program->steps[ i ].var.size ){
          u32 val;
          if( trieSearch( program->bigvars, program->steps[ i ].var.name, &val ) ){
            unmem( program->steps[ i ].var.name );
            program->steps[ i ].var.index = val;
          }else{
            trieInsert( program->bigvars, program->steps[ i ].var.name, program->numBigvars );
            program->bigvarNames[ program->numBigvars ] = program->steps[ i ].var.name;
            program->steps[ i ].var.index = program->numBigvars;
            ++program->numBigvars;
          }
        }else{
          u32 val;
          if( trieSearch( program->vars, program->steps[ i ].var.name, &val ) ){
            if( program->steps[ i ].var.size != program->varSizes[ val ] )
              error( "%s:%u command %u: %s", program->steps[ i ].filename,
                     program->steps[ i ].linenum,
                     program->steps[ i ].commandnum,
                     "Incorrect size setting already set value. Size is static." );
            unmem( program->steps[ i ].var.name );
            program->steps[ i ].var.index = val;
          } else {
            u32 varlen = strlen( program->steps[ i ].var.name );
            char* safeName = mem( varlen + 1, char );
            memcpy( safeName, program->steps[ i ].var.name, varlen + 1 );
            for( u32 i = 0; i < varlen; ++i )
              if( safeName[ i ] == '.' )
                safeName[ i ] = '_';
            trieInsert( program->vars, program->steps[ i ].var.name, program->numVars );
            switch( program->steps[ i ].var.size ){
            case 1:
              p += snprintf( p,
                             bufsize - ( p - glslUniformBlock ),
                             "uniform float %s;\n",
                             safeName );
              break;
            case 2:
              p += snprintf( p,
                             bufsize - ( p - glslUniformBlock ),
                             "uniform vec2 %s;\n",
                             safeName );
              break;
            case 3:
              p += snprintf( p,
                             bufsize - ( p - glslUniformBlock ),
                             "uniform vec3 %s;\n",
                             safeName );
              break;
            case 4:
              p += snprintf( p,
                             bufsize - ( p - glslUniformBlock ),
                             "uniform vec4 %s;\n",
                             safeName );
              break;
            case 16:
              p += snprintf( p,
                             bufsize - ( p - glslUniformBlock ),
                             "uniform mat4 %s;\n",
                             safeName );
              break;
            default:
              error( "%s", "Logic error in Atlas!" );
            }
            unmem( safeName );
            program->varNames[ program->numVars ] = program->steps[ i ].var.name;
            program->varOffsets[ program->numVars ] = offset;
            program->varSizes[ program->numVars ] = program->steps[ i ].var.size;
            if( program->steps[ i ].var.size == 1 || program->steps[ i ].var.size == 2 )
              offset += 2;
            else if( program->steps[ i ].var.size == 3 ||
                     program->steps[ i ].var.size == 4 )
              offset += 4;
            else
              offset += 16;
            program->steps[ i ].var.index = program->numVars;
            ++program->numVars;
          }
        }
      }
    }

    // p += snprintf( p, bufsize - ( p - glslUniformBlock ), "};\n" );
    program->varBlock = mem( offset, f32 );
    program->bigvarts = mem( program->numBigvars, tensor* );
    // populate bigvarts with scalar 0s.
    for( u32 i = 0; i < program->numBigvars; ++i ){
      f32* nt = mem( 1, f32 );
      program->bigvarts[ i ] = newTensor( 0, NULL, nt );;
    }
      
    // glGenBuffers( 1, &program->ubo );
    // glBindBuffer( GL_UNIFORM_BUFFER, program->ubo );
    // glBufferData( GL_UNIFORM_BUFFER, sizeof( f32 ) * offset, program->varBlock,
    // GL_DYNAMIC_DRAW ); glBindBufferBase( GL_UNIFORM_BUFFER, 0, program->ubo );
    // glBindBuffer( GL_UNIFORM_BUFFER, 0 );
    // glBindBufferBase( GL_UNIFORM_BUFFER, 0, 0 );
    // dbg( "Block %s, totsize %u", glslUniformBlock, offset );
  }

  // Second pass for ifs, ifns, calls, computes, and gets.
  for( u32 i = 0; i < program->numSteps; ++i )
    if( program->steps[ i ].type == IF || program->steps[ i ].type == IFN ||
        program->steps[ i ].type == CALL ){
      u32 jumpTo;

      if( !trieSearch( program->labels, program->steps[ i ].branchName, &jumpTo )
          && !trieSearch( program->labels, program->steps[ i ].branchBaseName, &jumpTo ) ){
        if( program->steps[ i ].type == CALL ){
          u32 vi;
          char* tp = program->steps[ i ].branchName;
          if( !trieSearch( program->vars, program->steps[ i ].branchName, &vi ) ){
            if( !trieSearch( program->bigvars, program->steps[ i ].branchName, &vi ) ){
              if( !trieSearch( program->vars, program->steps[ i ].branchBaseName, &vi ) ){
                if( !trieSearch( program->bigvars, program->steps[ i ].branchBaseName, &vi ) )
                  error( "%s:%u command %u: Statement with unknown label or variable %s",
                         program->steps[ i ].filename, program->steps[ i ].linenum,
                         program->steps[ i ].commandnum,
                         program->steps[ i ].branchBaseName?
                         program->steps[ i ].branchBaseName:
                         program->steps[ i ].branchName );
                program->steps[ i ].type = GET;
                program->steps[ i ].var.index = vi;
                program->steps[ i ].var.size = 0;
                unmem( tp );
              } else{
                program->steps[ i ].type = GET;
                program->steps[ i ].var.index = vi;
                program->steps[ i ].var.size = program->varSizes[ vi ];
                unmem( tp );
              }
            } else {
              program->steps[ i ].type = GET;
              program->steps[ i ].var.index = vi;
              program->steps[ i ].var.size = 0;
              unmem( tp );
            }
          } else{
            program->steps[ i ].type = GET;
            program->steps[ i ].var.index = vi;
            program->steps[ i ].var.size = program->varSizes[ vi ];
            unmem( tp );
          }
        } else
          error( "%s:%u command %u: Statement with unknown label %s",
                 program->steps[ i ].filename, program->steps[ i ].linenum,
                 program->steps[ i ].commandnum,
                 program->steps[ i ].branchBaseName?
                 program->steps[ i ].branchBaseName:
                 program->steps[ i ].branchName );
                
      } else {
        unmem( program->steps[ i ].branchName );
        program->steps[ i ].branch = jumpTo;
      }
    } else if( program->steps[ i ].type == GET ){
      u32 vi;
      if( !trieSearch( program->vars, program->steps[ i ].var.name, &vi ) ){
        if( !trieSearch( program->bigvars, program->steps[ i ].var.name, &vi ) ){
          if( !trieSearch( program->vars, program->steps[ i ].var.baseName, &vi ) ){
            if( !trieSearch( program->bigvars, program->steps[ i ].var.baseName, &vi ) )
              error( "%s:%u command %u: Attempt to get an an unknown variable %s",
                     program->steps[ i ].filename,
                     program->steps[ i ].linenum,
                     program->steps[ i ].commandnum,
                     program->steps[ i ].var.name );
            char* varName = program->steps[ i ].var.name;
            program->steps[ i ].var.index = vi;
            unmem( varName );
            program->steps[ i ].var.size = 0;
          } else {
            char* varName = program->steps[ i ].var.name;
            program->steps[ i ].var.index = vi;
            unmem( varName );
            program->steps[ i ].var.size = program->varSizes[ vi ];
          }
        } else {
          char* varName = program->steps[ i ].var.name;
          program->steps[ i ].var.index = vi;
          unmem( varName );
          program->steps[ i ].var.size = 0;
        }
      } else {
        char* varName = program->steps[ i ].var.name;
        program->steps[ i ].var.index = vi;
        unmem( varName );
        program->steps[ i ].var.size = program->varSizes[ vi ];
      }
    } else if( program->steps[ i ].type == COMPUTE ){
      char* vglslpre = program->steps[ i ].toCompute.vglslpre;
      char* glslpre = program->steps[ i ].toCompute.glslpre;
      char* glsl = program->steps[ i ].toCompute.glsl;
      char* vglsl = program->steps[ i ].toCompute.vglsl;
      program->steps[ i ].compute =
        addCompute( program->steps[ i ].filename,
                    program->steps[ i ].linenum,
                    program->steps[ i ].commandnum,
                    program,
                    glslUniformBlock,
                    vglslpre, glslpre, vglsl, glsl,
                    program->steps[ i ].toCompute.argCount,
                    program->steps[ i ].toCompute.retCount,
                    program->steps[ i ].toCompute.channels,
                    program->steps[ i ].toCompute.reuse );
      unmem( glsl );
      unmem( glslpre );
      unmem( vglsl );
      unmem( vglslpre );
    }
  unmem( glslUniformBlock );
}
bool fileExists( const char *filename ){
  FILE *file = fopen( filename, "rb" );
  if( file ){
    fclose( file );
    return 1;
  }
  return 0;
}
void addProgramFromFile( const char* filename, program* program ){
  FILE* file = fopen( filename, "rb" );

  if( !file )
    error( "%s %s.", "Failed to open file", filename );
  // Seek to the end of the file to determine its size
  if( fseek( file, 0, SEEK_END ) ){
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
  addProgram( filename, buffer, program );
  unmem( buffer );
}
program* newProgramFromFile( const char* filename ){
  // Always reset to workplace'' for a new program.
  unmem( workspace );
  workspace = mem( 1, char );
  workspace[ 0 ] = 0;

  program* prog = newProgram();
  addProgramFromFile( filename, prog );
  finalize( prog );


  // Reset afterwards too.
  unmem( workspace );
  workspace = mem( 1, char );
  workspace[ 0 ] = 0;
  return prog;
}

void deleteProgram( program* p ){
  for( u32 i = 0; i < p->numComputes; ++i )
    deleteCompute( p->computes[ i ] );
  for( u32 i = 0; i < p->numSteps; ++i ){
    if( p->steps[ i ].type == LOAD && p->steps[ i ].progName )
      unmem( p->steps[ i ].progName );
    else if( p->steps[ i ].type == TENSOR ){
      deleteTensor( p->steps[ i ].tensor );
      p->steps[ i ].tensor = NULL;
    }
  }
  deleteTrieNode( p->labels );
  for( u32 i = 0; i < p->numFilenames; ++i )
    unmem( p->filenames[ i ] );
  unmem( p->filenames );
  for( u32 i = 0; i < p->numVars; ++i )
    unmem( p->varNames[ i ] );
  for( u32 i = 0; i < p->numBigvars; ++i )
    unmem( p->bigvarNames[ i ] );
  for( u32 i = 0; i < p->numBigvars; ++i )
    if( p->bigvarts[ i ] )
      deleteTensor( p->bigvarts[ i ] );
  unmem( p->bigvarts );
  unmem( p->varNames );
  unmem( p->bigvarNames );
  deleteTrieNode( p->vars );
  deleteTrieNode( p->bigvars );
  unmem( p->varBlock );
  unmem( p->varSizes );
  unmem( p->varOffsets );
  unmem( p->returns );
  unmem( p->computes );
  unmem( p->steps );
  unmem( p );
}
// A pointer pointer because program might change during e.g. a load.
bool runProgram( tensorStack* ts, program** progp ){
  program* p = *progp;
  CHECK_GL_ERROR();
  for( u32 i = 0; i < p->numSteps; ++i ){
    // dbg( "Step %u", i );
    step* s = p->steps + i;
    switch( s->type ){
    case WINDOWSIZE: {
      static const u32 wsshape[ 1 ] = { 2 };
      int windowWidth, windowHeight;
      SDL_GetWindowSize( window, &windowWidth, &windowHeight );
      f32* data = mem( 2, f32 );
      data[ 0 ] = windowWidth;
      data[ 1 ] = windowHeight;
      push( ts, newTensor( 1, wsshape, data ) );
      break;
    }
    case GETINPUT: {
      static const u32 wsshape[ 1 ] = { 6 };
      f32* data = mem( 6, f32 );
      //int dx, dy;
      //mainPoll();
      //SDL_GetRelativeMouseState( &dx, &dy );  // Get mouse delta
#ifndef __EMSCRIPTEN__
      //SDL_LockMutex( data_mutex );
#endif
      data[ 0 ] = dx;dx = 0;
      data[ 1 ] = dy;dy = 0;
      f32 delta = ( mouseWheel - mouseWheelPos ) / 10.0;
      data[ 2 ] = delta;
      mouseWheelPos += delta;
      //Uint32 buttons = SDL_GetMouseState( NULL, NULL );
      if( ( buttons & SDL_BUTTON( SDL_BUTTON_LEFT ) || touchClicks[ 0 ] ) && !touchClicks[ 1 ] && !touchClicks[ 2 ] )
        data[ 3 ] = 1;
      else
        data[ 3 ] = 0;
      if( buttons & SDL_BUTTON( SDL_BUTTON_RIGHT ) || touchClicks[ 1 ] )
        data[ 4 ] = 1;
      else
        data[ 4 ] = 0;
      if( buttons & SDL_BUTTON( SDL_BUTTON_MIDDLE ) || touchClicks[ 2 ] )
        data[ 5 ] = 1;
      else
        data[ 5 ] = 0;

      if( pinchZoom != 0.0 ){
        data[ 2 ] += pinchZoom;
        pinchZoom = 0.0;
      }
      
      if( doubleClicks[ 0 ] ){
        doubleClicks[ 0 ] = 0;
        data[ 3 ] = 2;
      } 
      if( doubleClicks[ 1 ] ){
        doubleClicks[ 1 ] = 0;
        data[ 4 ] = 2;
      } 
      if( doubleClicks[ 2 ] ){
        doubleClicks[ 2 ] = 0;
        data[ 5 ] = 2;
      } 
#ifndef __EMSCRIPTEN__
      //SDL_UnlockMutex( data_mutex );
#endif
      push( ts, newTensor( 1, wsshape, data ) );
      break;
    }
    case GAMEPAD: {
#ifndef __EMSCRIPTEN__
      //SDL_LockMutex( data_mutex );
#endif
      u32 gpshape[ 4 ] = { 0, 21, 1, 1 };
      f32* data;
      
      for( u32 i = 0; i < MAX_CONTROLLERS; ++i )
        if( controllers[ i ] ){
          ++gpshape[ 0 ];
        }
      data = mem( gpshape[ 0 ] * 21, f32 );
      u32 c = 0;
      for( u32 i = 0; i < MAX_CONTROLLERS; ++i )
        if( controllers[ i ] ){
          memcpy( data + c++ * 21, &joysticks[ i * 21 ], sizeof( f32 ) * 21 );
        }

      push( ts, newTensor( 2, gpshape, data ) );
#ifndef __EMSCRIPTEN__
      //SDL_UnlockMutex( data_mutex );
#endif
      break;
    }
    case POW: {
      if( ts->size < 2 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to pow without enough arguments on the stack." );

      tensorToHostMemory( ts->stack[ ts->size - 1 ] );
      tensorToHostMemory( ts->stack[ ts->size - 2 ] );
      takeOwnership( ts->stack[ ts->size - 2 ] );
      tensor* t1 = ts->stack[ ts->size - 1 ];
      tensor* t2 = ts->stack[ ts->size - 2 ];
      if( t1->rank != t2->rank )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to pow tensors with incompatible ranks." );
      for( u32 i = 0; i < t1->rank; ++i )
        if( t1->shape[ i ] != t2->shape[ i ] )
          error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to pow tensors with incompatible shapes." );
      for( s32 i0 = 0; i0 < t1->shape[ 0 ]; ++i0 )
        for( s32 i1 = 0; i1 < t1->shape[ 1 ]; ++i1 )
          for( s32 i2 = 0; i2 < t1->shape[ 2 ]; ++i2 )
            for( s32 i3 = 0; i3 < t1->shape[ 3 ]; ++i3 ){
              f32* offset1 = t1->data + t1->offset + i0 * t1->strides[ 0 ] +
                i1 * t1->strides[ 1 ] + i2 * t1->strides[ 2 ] +
                i3 * t1->strides[ 3 ];
              f32* offset2 = t2->data + t2->offset + i0 * t2->strides[ 0 ] +
                i1 * t2->strides[ 1 ] + i2 * t2->strides[ 2 ] +
                i3 * t2->strides[ 3 ];
              *offset2 = powf( *offset2, *offset1 );
            }

      pop( ts );
      // dbg( "%s", "pow" );
      break;
    }
    case ADD: {
      if( ts->size < 2 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to add without enough arguments on the stack." );

      tensorToHostMemory( ts->stack[ ts->size - 1 ] );
      tensorToHostMemory( ts->stack[ ts->size - 2 ] );
      takeOwnership( ts->stack[ ts->size - 2 ] );
      tensor* t1 = ts->stack[ ts->size - 1 ];
      tensor* t2 = ts->stack[ ts->size - 2 ];
      if( t1->rank != t2->rank )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to add tensors with incompatible ranks." );
      for( u32 i = 0; i < t1->rank; ++i )
        if( t1->shape[ i ] != t2->shape[ i ] )
          error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to add tensors with incompatible shapes." );
      for( s32 i0 = 0; i0 < t1->shape[ 0 ]; ++i0 )
        for( s32 i1 = 0; i1 < t1->shape[ 1 ]; ++i1 )
          for( s32 i2 = 0; i2 < t1->shape[ 2 ]; ++i2 )
            for( s32 i3 = 0; i3 < t1->shape[ 3 ]; ++i3 ){
              f32* offset1 = t1->data + t1->offset + i0 * t1->strides[ 0 ] +
                i1 * t1->strides[ 1 ] + i2 * t1->strides[ 2 ] +
                i3 * t1->strides[ 3 ];
              f32* offset2 = t2->data + t2->offset + i0 * t2->strides[ 0 ] +
                i1 * t2->strides[ 1 ] + i2 * t2->strides[ 2 ] +
                i3 * t2->strides[ 3 ];
              *offset2 = *offset2 + *offset1;
            }

      pop( ts );
      // dbg( "%s", "add" );
      break;
    }
    case SUB: {
      if( ts->size < 2 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to sub without enough arguments on the stack." );

      tensorToHostMemory( ts->stack[ ts->size - 1 ] );
      tensorToHostMemory( ts->stack[ ts->size - 2 ] );
      takeOwnership( ts->stack[ ts->size - 2 ] );
      tensor* t1 = ts->stack[ ts->size - 1 ];
      tensor* t2 = ts->stack[ ts->size - 2 ];
      if( t1->rank != t2->rank )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to sub tensors with incompatible ranks." );
      for( u32 i = 0; i < t1->rank; ++i )
        if( t1->shape[ i ] != t2->shape[ i ] )
          error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to sub tensors with incompatible shapes." );
      for( s32 i0 = 0; i0 < t1->shape[ 0 ]; ++i0 )
        for( s32 i1 = 0; i1 < t1->shape[ 1 ]; ++i1 )
          for( s32 i2 = 0; i2 < t1->shape[ 2 ]; ++i2 )
            for( s32 i3 = 0; i3 < t1->shape[ 3 ]; ++i3 ){
              f32* offset1 = t1->data + t1->offset + i0 * t1->strides[ 0 ] +
                i1 * t1->strides[ 1 ] + i2 * t1->strides[ 2 ] +
                i3 * t1->strides[ 3 ];
              f32* offset2 = t2->data + t2->offset + i0 * t2->strides[ 0 ] +
                i1 * t2->strides[ 1 ] + i2 * t2->strides[ 2 ] +
                i3 * t2->strides[ 3 ];
              *offset2 = *offset2 - *offset1;
            }
      pop( ts );
      // dbg( "%s", "sub" );
      break;
    }
    case MUL: {
      if( ts->size < 2 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to mul without enough arguments on the stack." );

      tensorToHostMemory( ts->stack[ ts->size - 1 ] );
      tensorToHostMemory( ts->stack[ ts->size - 2 ] );
      takeOwnership( ts->stack[ ts->size - 2 ] );
      tensor* t1 = ts->stack[ ts->size - 1 ];
      tensor* t2 = ts->stack[ ts->size - 2 ];
      if( t1->rank != t2->rank )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to mul tensors with incompatible ranks." );
      for( u32 i = 0; i < t1->rank; ++i )
        if( t1->shape[ i ] != t2->shape[ i ] )
          error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to mul tensors with incompatible shapes." );
      for( s32 i0 = 0; i0 < t1->shape[ 0 ]; ++i0 )
        for( s32 i1 = 0; i1 < t1->shape[ 1 ]; ++i1 )
          for( s32 i2 = 0; i2 < t1->shape[ 2 ]; ++i2 )
            for( s32 i3 = 0; i3 < t1->shape[ 3 ]; ++i3 ){
              f32* offset1 = t1->data + t1->offset + i0 * t1->strides[ 0 ] +
                i1 * t1->strides[ 1 ] + i2 * t1->strides[ 2 ] +
                i3 * t1->strides[ 3 ];
              f32* offset2 = t2->data + t2->offset + i0 * t2->strides[ 0 ] +
                i1 * t2->strides[ 1 ] + i2 * t2->strides[ 2 ] +
                i3 * t2->strides[ 3 ];
              *offset2 = *offset2 * *offset1;
            }

      pop( ts );
      // dbg( "%s", "mul" );
      break;
    }
    case DIV: {
      if( ts->size < 2 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to div without enough arguments on the stack." );

      tensorToHostMemory( ts->stack[ ts->size - 1 ] );
      tensorToHostMemory( ts->stack[ ts->size - 2 ] );
      takeOwnership( ts->stack[ ts->size - 2 ] );
      tensor* t1 = ts->stack[ ts->size - 1 ];
      tensor* t2 = ts->stack[ ts->size - 2 ];
      if( t1->rank != t2->rank )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to div tensors with incompatible ranks." );
      for( u32 i = 0; i < t1->rank; ++i )
        if( t1->shape[ i ] != t2->shape[ i ] )
          error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to div tensors with incompatible shapes." );
      for( s32 i0 = 0; i0 < t1->shape[ 0 ]; ++i0 )
        for( s32 i1 = 0; i1 < t1->shape[ 1 ]; ++i1 )
          for( s32 i2 = 0; i2 < t1->shape[ 2 ]; ++i2 )
            for( s32 i3 = 0; i3 < t1->shape[ 3 ]; ++i3 ){
              f32* offset1 = t1->data + t1->offset + i0 * t1->strides[ 0 ] +
                i1 * t1->strides[ 1 ] + i2 * t1->strides[ 2 ] +
                i3 * t1->strides[ 3 ];
              f32* offset2 = t2->data + t2->offset + i0 * t2->strides[ 0 ] +
                i1 * t2->strides[ 1 ] + i2 * t2->strides[ 2 ] +
                i3 * t2->strides[ 3 ];
              *offset2 = *offset2 / *offset1;
            }

      pop( ts );
      // dbg( "%s", "div" );
      break;
    }
    case SIN: {
      if( ts->size < 1 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Attempt to call sin without an argument." );

      tensorToHostMemory( ts->stack[ ts->size - 1 ] );
      tensor* t1 = ts->stack[ ts->size - 1 ];
      for( s32 i0 = 0; i0 < t1->shape[ 0 ]; ++i0 )
        for( s32 i1 = 0; i1 < t1->shape[ 1 ]; ++i1 )
          for( s32 i2 = 0; i2 < t1->shape[ 2 ]; ++i2 )
            for( s32 i3 = 0; i3 < t1->shape[ 3 ]; ++i3 ){
              f32* offset1 = t1->data + t1->offset + i0 * t1->strides[ 0 ] +
                i1 * t1->strides[ 1 ] + i2 * t1->strides[ 2 ] +
                i3 * t1->strides[ 3 ];
              *offset1 = sinf( *offset1 );
            }

      // dbg( "%s", "sin" );
      break;
    }
    case COS: {
      if( ts->size < 1 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Attempt to call cos without an argument." );

      tensorToHostMemory( ts->stack[ ts->size - 1 ] );
      tensor* t1 = ts->stack[ ts->size - 1 ];
      for( s32 i0 = 0; i0 < t1->shape[ 0 ]; ++i0 )
        for( s32 i1 = 0; i1 < t1->shape[ 1 ]; ++i1 )
          for( s32 i2 = 0; i2 < t1->shape[ 2 ]; ++i2 )
            for( s32 i3 = 0; i3 < t1->shape[ 3 ]; ++i3 ){
              f32* offset1 = t1->data + t1->offset + i0 * t1->strides[ 0 ] +
                i1 * t1->strides[ 1 ] + i2 * t1->strides[ 2 ] +
                i3 * t1->strides[ 3 ];
              *offset1 = cosf( *offset1 );
            }

      // dbg( "%s", "cos" );
      break;
    }
    case FLOOR: {
      if( ts->size < 1 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Attempt to call floor without an argument." );

      tensorToHostMemory( ts->stack[ ts->size - 1 ] );
      tensor* t1 = ts->stack[ ts->size - 1 ];
      for( s32 i0 = 0; i0 < t1->shape[ 0 ]; ++i0 )
        for( s32 i1 = 0; i1 < t1->shape[ 1 ]; ++i1 )
          for( s32 i2 = 0; i2 < t1->shape[ 2 ]; ++i2 )
            for( s32 i3 = 0; i3 < t1->shape[ 3 ]; ++i3 ){
              f32* offset1 = t1->data + t1->offset + i0 * t1->strides[ 0 ] +
                i1 * t1->strides[ 1 ] + i2 * t1->strides[ 2 ] +
                i3 * t1->strides[ 3 ];
              *offset1 = floorf( *offset1 );
            }

      // dbg( "%s", "floor" );
      break;
    }
    case CEIL: {
      if( ts->size < 1 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Attempt to call ceil without an argument." );

      tensorToHostMemory( ts->stack[ ts->size - 1 ] );
      tensor* t1 = ts->stack[ ts->size - 1 ];
      for( s32 i0 = 0; i0 < t1->shape[ 0 ]; ++i0 )
        for( s32 i1 = 0; i1 < t1->shape[ 1 ]; ++i1 )
          for( s32 i2 = 0; i2 < t1->shape[ 2 ]; ++i2 )
            for( s32 i3 = 0; i3 < t1->shape[ 3 ]; ++i3 ){
              f32* offset1 = t1->data + t1->offset + i0 * t1->strides[ 0 ] +
                i1 * t1->strides[ 1 ] + i2 * t1->strides[ 2 ] +
                i3 * t1->strides[ 3 ];
              *offset1 = ceilf( *offset1 );
            }

      // dbg( "%s", "ceil" );
      break;
    }
    case MINMAX: {
      if( ts->size < 1 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Attempt to call minmax without an argument." );
      if( !ts->stack[ ts->size - 1 ]->size )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Attempt to call minmax on an empty tensor." );
        
      tensorToHostMemory( ts->stack[ ts->size - 1 ] );
      float min = FLT_MAX;
      float max = FLT_MIN;
      tensor* t1 = ts->stack[ ts->size - 1 ];
      for( s32 i0 = 0; i0 < t1->shape[ 0 ]; ++i0 )
        for( s32 i1 = 0; i1 < t1->shape[ 1 ]; ++i1 )
          for( s32 i2 = 0; i2 < t1->shape[ 2 ]; ++i2 )
            for( s32 i3 = 0; i3 < t1->shape[ 3 ]; ++i3 ){
              f32* offset1 = t1->data + t1->offset + i0 * t1->strides[ 0 ] +
                i1 * t1->strides[ 1 ] + i2 * t1->strides[ 2 ] +
                i3 * t1->strides[ 3 ];
              if( *offset1 < min ) min = *offset1;
              if( *offset1 > max ) max = *offset1;
            }
      
      pop( ts );
      float* nt = mem( 2, float );
      nt[ 0 ] = min; nt[ 1 ] = max;
      u32 ntshape[ 1 ] = { 2 };
      push( ts, newTensor( 1, ntshape, nt ) );
      // dbg( "%s", "minmax" );

      break;
    }

    case TENSOR:
      push( ts, copyTensor( s->tensor ) );
      break;
    case PRINT:
      printStack( ts );
      // dbg( "%s", "print" );
      break;
    case POP:
      pop( ts );
      // dbg( "%s %u", "pop", ts->size );
      break;
    case CALL:
      if( p->numReturns >= p->returnStackSize ){
        p->returnStackSize *= 2;
        u32* t = mem( p->returnStackSize, u32 );
        memcpy( t, p->returns, p->numReturns * sizeof( u32 ) );
        unmem( p->returns );
        p->returns = t;
      }
      p->returns[ p->numReturns++ ] = i;
      i = s->branch - 1;
      // dbg( "%s", "call" );
      break;
    case RETURN:
      if( !p->numReturns )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to return with an empty return stack." );
      i = p->returns[ --p->numReturns ];
      // dbg( "%s", "return" );
      break;
    case COMPUTE:{
      if( ts->size < 2 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Attempt to run a compute statement without both a shape parameter and a vertex count on "
               "the stack." );
      if( ts->stack[ ts->size - 1 ]->rank != 1 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "The shape for a compute was not a rank 1 tensor." );
      if( ts->stack[ ts->size - 1 ]->size > 4 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "The shape for an initilizer was more than 4 component." );
      if( ts->stack[ ts->size - 2 ]->rank != 0 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "A compute was run with a non-scalar vertex count." );
      tensorToHostMemory( ts->stack[ ts->size - 1 ] );
      tensorToHostMemory( ts->stack[ ts->size - 2 ] );
      f32 vertCount = ts->stack[ ts->size - 2 ]->data[ ts->stack[ ts->size - 2 ]->offset ];
      u32 shape[ 4 ];
      u32 rank = ts->stack[ ts->size - 1 ]->size;
      for( u32 i = 0; i < rank; ++i )
        shape[ i ] = ts->stack[ ts->size - 1 ]->data[ i ];
      for( u32 i = rank; i < 4; ++ i )
        shape[ i ] = 1;
      // dbg( "%u rc", p->computes[ s->compute ]->retCount );
      u32 channels = p->computes[ s->compute ]->channels;
      if( channels >= 10 )
        channels /= 10;
      if( channels && rank != 3 )
        error( "%s:%u command %u: %s %u.", s->filename, s->linenum, s->commandnum,
               "Attempt to run a compute statement into texture not of rank 3 but of rank", rank );
      if( channels && ( rank != 3 || shape[ 2 ] != channels ) )
        error( "%s:%u command %u: %s %u.", s->filename, s->linenum, s->commandnum,
               "Attempt to run a compute statement into a texture with a bad number of components ", shape[ 2 ] );
      
      pop( ts );
      pop( ts );
      tensor** rets =
        newTensorsInitialized( p, ts, rank, shape, p->computes[ s->compute ], vertCount );
      if( rets ){
        for( u32 i = 0; i < p->computes[ s->compute ]->retCount; ++i )
          push( ts, rets[ p->computes[ s->compute ]->retCount - i - 1 ] );
        unmem( rets );
      }
      // dbg( "%s", "compute" );
      break;
    }
    case CAT: {
      if( ts->size < 3 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Attempt to concatenate without enough arguments on the stack." );
      if( ts->stack[ ts->size - 1 ]->rank )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Attempt to concatenate with a nonscalar axis parameter." );

      tensorToHostMemory( ts->stack[ ts->size - 1 ] );
      u32 axis = *( ts->stack[ ts->size - 1 ]->data +
                    ts->stack[ ts->size - 1 ]->offset );
      pop( ts );
      if( ts->stack[ ts->size - 2 ]->rank != ts->stack[ ts->size - 1 ]->rank )
        error( "%s:%u command %u: %s: %u vs %u.", s->filename, s->linenum, s->commandnum,
               "Attempt to concatenate tensors of different rank",
               ts->stack[ ts->size - 1 ]->rank,
               ts->stack[ ts->size - 2 ]->rank );
      
      tensorCat( ts, ts->size - 2, ts->size - 1, axis );
      pop( ts );
      // dbg( "%s %u", "cat", axis );
      break;
    }
    case MULTM: {
      if( ts->size < 2 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Attempt to multiply matrices with not enough parameters on the stack." );
      tensor* t1 = ts->stack[ ts->size - 1 ];
      tensor* t2 = ts->stack[ ts->size - 2 ];
      if( !t1 || t1->rank > 2 || !t2 || t2->rank > 2 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Bad tensor or tensor rank in matrix multiplication." );
      while( t1->rank < 2 )
        tensorExtrude( t1 );
      while( t2->rank < 2 )
        tensorEnclose( t2 );
      if( t1->shape[ 0 ] != t2->shape[ 1 ] )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Incompatible shapes in matrix multiplication." );
      tensorMultiply( ts );
      break;
    }
    case TOSTRING: {
      if( ts->size < 1 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Attempt to create a string without a parameter on the stack." );
      tensor* t1 = ts->stack[ ts->size - 1 ];
      if( t1->rank != 0 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Expected a scalar for toString." );
      char* fd = formatTensorData( t1 );
      pop( ts );
      tensor* nt = tensorFromString( fd );
      unmem( fd );
      push( ts, nt );
      break;
    }
    case ROT: {
      if( ts->size < 2 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Attempt to create a rotation matrix without enough parameters on the stack." );
      tensor* t1 = ts->stack[ ts->size - 1 ];
      tensor* t2 = ts->stack[ ts->size - 2 ];
      if( t1->rank != 1 || t1->shape[ 0 ] != 3 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Expected a rank 1 length 3 vector for rotation." );
	
      if( t2->rank )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Expected a scalar angle for rotation." );
      tensorRotate( ts, ts->size - 1, ts->size - 2 );
      break;
    }
    case PROJ: {
      if( !ts->size )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Attempt to create a projection matrix without a parameter on the stack." );
      tensor* t1 = ts->stack[ ts->size - 1 ];
      if( t1->rank != 1 || t1->shape[ 0 ] != 5 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Expected a rank 1 length 5 vector (fov, width, height, near, far) for projection." );
      tensorProject( ts, ts->size - 1 );
      break;
    }
    case ORTHO: {
      if( !ts->size )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Attempt to create a orthographic projection matrix without a parameter on the stack." );
      tensor* t1 = ts->stack[ ts->size - 1 ];
      if( t1->rank != 1 || t1->shape[ 0 ] != 6 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Expected a rank 1 length 6 vector (left, right, bottom, top, near, far) for orthographic projection." );
      tensorOrtho( ts, ts->size - 1 );
      break;
    }
    case TRANS: {
      if( !ts->size )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Attempt to create a translation matrix without a parameter on the stack." );
      tensor* t1 = ts->stack[ ts->size - 1 ];
      if( t1->rank != 1 || t1->shape[ 0 ] != 3 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Expected a rank 1 length 3 vector for translation." );
      tensorTranslate( ts, ts->size - 1 );
      break;
    }
    case TEXTURE: {
      tensor* cur = ts->stack[ ts->size - 1 ];
      if( !cur->gpu || cur->tex.channels == 0 )
        error( "%s", "Attempt to use an inapropriate tensor as a texture. Must be channeled." );
      glBindTexture( GL_TEXTURE_2D, cur->tex.texture );
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR );
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR  );
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT );
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT );
      if( getMaxAnisotropy() > 1.0 ){
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, getMaxAnisotropy() );
      }
      glGenerateMipmap( GL_TEXTURE_2D );
      glBindTexture( GL_TEXTURE_2D, 0 );
      break;
     
    }
    case REVERSE: {
      if( !ts->size )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Attempt to reverse with no axis parameter on the stack." );
      if( ts->stack[ ts->size - 1 ]->rank )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to reverse a nonscalar axis parameter." );

      tensorToHostMemory( ts->stack[ ts->size - 1 ] );
      u32 axis = *( ts->stack[ ts->size - 1 ]->data +
                    ts->stack[ ts->size - 1 ]->offset );
      pop( ts );
      tensorReverse( ts, ts->size - 1, axis );
      // dbg( "%s %u", "reverse", axis );
      break;
    }
    case SHAPE: {
      if( !ts->size )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Attempt to get the shape of a tensor with nothing on the stack." );
      tensor* cur = ts->stack[ ts->size - 1 ];
      f32* newData = mem( cur->rank, f32 );
      for( u32 i = 0; i < cur->rank; ++i )
        newData[ i ] = cur->shape[ i ];
      u32 newShape[ 1 ] = { cur->rank };
      pop( ts );
      push( ts, newTensor( 1, newShape, newData ) );
      // dbg( "%s %u", "shape", axis );
      break;
    }
    case LENGTH: {
      if( !ts->size )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Attempt to get the length of a tensor with nothing on the stack." );
      const tensor* top = ts->stack[ ts->size - 1 ];
      if( top->rank != 1 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Attempt to get the length of a tensor with rank not equal 1." );
      f32 sumsquares = 0;
      for( u32 i = 0; i < top->shape[ 0 ]; ++i )
        sumsquares += top->data[ top->offset + top->strides[ 0 ] * i ] *
          top->data[ top->offset + top->strides[ 0 ] * i ];
      f32* newData = mem( 1, f32 );
      *newData = sqrtf( sumsquares );
      pop( ts );
      push( ts, newTensor( 0, NULL, newData ) );
      // dbg( "%s %f", "length", *newData );
      break;
    }
    case TIME: {
      f32* time = mem( 1, f32 );
      *time = timeDelta;
      push( ts, newTensor( 0, NULL, time ) );
      // dbg( "%s", "time" );
      break;
    }
    case LOAD: {
      if( !s->progName ){
        if( !ts->size )
          error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
                 "Attempt to load a string filename with no string on the stack." );
        tensor* cur = ts->stack[ ts->size - 1 ];
        if( cur->rank != 1 )
          error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
                 "Attempt to load a string filename with a nonvector." );
        char* fn = tensorToString( ts->stack[ ts->size - 1 ] );
        p = newProgramFromFile( fn );
        p->filenames[ p->numFilenames++ ] = fn;
      }else{
        u32 len = strlen( s->progName );
        char* nn = mem( len + 2, char );
        strncpy( nn, s->progName, len + 2 );
        p = newProgramFromFile( nn );
        p->filenames[ p->numFilenames++ ] = nn;
      }
      deleteProgram( *progp );
      *progp = p;
      while( ts->size )
        pop( ts );
      // dbg( "%s'%s'", "load ", s=>progName );
      return true;
    }
    case FIRST: {
      if( !ts->size )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Attempt to take the first element with no parameter on the stack." );

      tensorTakeFirst( ts, ts->size - 1 );
      // dbg( "%s", "first" );
      break;
    }
    case LAST: {
      if( !ts->size )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Attempt to take the first element with no parameter on the stack." );

      tensorTakeLast( ts, ts->size - 1 );
      // dbg( "%s", "first" );
      break;
    }
    case DUP:
      if( !ts->size )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to duplicate with no parameter on the stack." );
      if( ts->stack[ ts->size - 1 ]->rank )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to duplicate with a nonscalar parameter." );

      tensorToHostMemory( ts->stack[ ts->size - 1 ] );
      u32 dup = *( ts->stack[ ts->size - 1 ]->data +
                   ts->stack[ ts->size - 1 ]->offset );
      pop( ts );
      if( dup + 1 > ts->size )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to duplicate past the end of the stack." );
      push( ts, copyTensor( ts->stack[ ( ts->size - 1 ) - dup ] ) );
      break;
    case REPEAT: {
      if( ts->size < 2 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to repeate without enough parameters on the stack." );
      if( ts->stack[ ts->size - 1 ]->rank )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to repeat with a nonscalar count." );

      tensorToHostMemory( ts->stack[ ts->size - 1 ] );
      u32 count = *( ts->stack[ ts->size - 1 ]->data +
                     ts->stack[ ts->size - 1 ]->offset );
      pop( ts );
      tensorRepeat( ts, ts->size - 1, count );
      // dbg( "%s %u", "rep", count );
      break;
    }
    case BURY: {
      if( !ts->size )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to bury with no parameter on the stack." );
      if( ts->stack[ ts->size - 1 ]->rank )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to bury with a nonscalar parameter." );
      
      tensorToHostMemory( ts->stack[ ts->size - 1 ] );
      u32 bury = *( ts->stack[ ts->size - 1 ]->data +
                    ts->stack[ ts->size - 1 ]->offset );
      pop( ts );
      if( bury >= ts->size )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to bury past the end of the stack." );
      tensor* tb = ts->stack[ ts->size - 1 ];
      for( u32 i = ts->size - 1; i > ( ts->size - 1 ) - bury; --i ){
        if( !tb->ownsData && ts->stack[ i - 1 ]->ownsData &&
            tb->gpu == ts->stack[ i - 1 ]->gpu &&
            ( tb->data == ts->stack[ i - 1 ]->data || tb->tex.texture == ts->stack[ i - 1 ]->tex.texture ) )
          takeOwnership( tb );
        ts->stack[ i ] = ts->stack[ i - 1 ];
      }
      ts->stack[ ( ts->size - 1 ) - bury ] = tb;
      break;
    }
    case RAISE: {
      if( !ts->size )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to raise with no parameter on the stack." );
      if( ts->stack[ ts->size - 1 ]->rank )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to raise with a nonscalar parameter." );
      
      tensorToHostMemory( ts->stack[ ts->size - 1 ] );
      u32 raise = *( ts->stack[ ts->size - 1 ]->data +
                     ts->stack[ ts->size - 1 ]->offset );
      pop( ts );
      if( raise >= ts->size )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to bury past the end of the stack." );
      tensor* tr = ts->stack[ ( ts->size - 1 ) - raise ];
      for( u32 i = ( ts->size - 1 ) - raise; i < ts->size - 1; ++i ){
        if( tr->ownsData && !ts->stack[ i + 1 ]->ownsData &&
            tr->gpu == ts->stack[ i + 1 ]->gpu &&
            ( tr->data == ts->stack[ i + 1 ]->data || tr->tex.texture == ts->stack[ i + 1 ]->tex.texture ) )
          takeOwnership( ts->stack[ i + 1 ] );
        ts->stack[ i ] = ts->stack[ i + 1 ];
      }
      ts->stack[ ts->size - 1 ] = tr;
      break;
    }
    case BACKFACE: {
      if( glIsEnabled( GL_CULL_FACE ) == GL_TRUE )
        glDisable( GL_CULL_FACE );
      else
        glEnable( GL_CULL_FACE );
      break;
    }
    case IF: {
      if( !ts->size )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to if with no parameter on the stack." );
      if( ts->stack[ ts->size - 1 ]->rank )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to if with a non-scalar parameter." );

      tensorToHostMemory( ts->stack[ ts->size - 1 ] );
      f32 cond = *( ts->stack[ ts->size - 1 ]->data +
                    ts->stack[ ts->size - 1 ]->offset );
      pop( ts );
      if( cond > 0.0 )
        i = s->branch - 1;
      // dbg( "%s %f", "if", cond );
      break;
    }
    case IFN: {
      if( !ts->size )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to ifn with no parameter on the stack." );
      if( ts->stack[ ts->size - 1 ]->rank )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to ifn with a non-scalar parameter." );

      tensorToHostMemory( ts->stack[ ts->size - 1 ] );
      f32 cond = *( ts->stack[ ts->size - 1 ]->data +
                    ts->stack[ ts->size - 1 ]->offset );
      pop( ts );
      if( cond <= 0.0 )
        i = s->branch - 1;
      // dbg( "%s %f", "ifn", cond );
      break;
    }
    case TRANSPOSE:
      if( !ts->size )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Attempt to transpose with no axes parameter on the stack." );
      if( ts->stack[ ts->size - 1 ]->rank != 1 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum,
               "Attempt to transpose with a axes parameter not of rank 1." );

      u32 axis1 = *( ts->stack[ ts->size - 1 ]->data +
                     ts->stack[ ts->size - 1 ]->offset );
      u32 axis2 =
        *( ts->stack[ ts->size - 1 ]->data + ts->stack[ ts->size - 1 ]->offset +
           ts->stack[ ts->size - 1 ]->strides[ 0 ] );
      pop( ts );
      tensorTranspose( ts, ts->size - 1, axis1, axis2 );
      // dbg( "%s %u %u", "transpose", axis1, axis2 );
      break;
    case DEPTH: {
      depthTest = !depthTest;
      break;
    }
    case ENCLOSE: {
      tensor* t = ts->stack[ ts->size - 1 ];
      if( !ts->size )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to enclose with an empty stack." );
      if( t->rank == 4 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to enclose a parameter of rank 4 (rank too high)." );
      tensorEnclose( t );
      break;
    }
    case EXTRUDE: {
      tensor* t = ts->stack[ ts->size - 1 ];
      if( !ts->size )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to extrude with an empty stack." );
      if( t->rank == 4 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to extrude a parameter of rank 4 (rank too high)." );
      tensorExtrude( t );
      break;
    }
    case UNEXTRUDE: {
      tensor* t = ts->stack[ ts->size - 1 ];
      if( !ts->size )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to unextrude with an empty stack." );
      if( t->rank == 0 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to unextrude a scalar." );
      if( t->shape[ t->rank - 1 ] != 1 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to unextrude a tensor with the last dimension not equal 1." );
      tensorUnextrude( t );
      break;
    }
    case SLICE:
      if( ts->size < 2 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to slice without enough elements on the stack." );
      if( ts->stack[ ts->size - 1 ]->rank != 1 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to slice with a parameter not of rank 1." );
      if( ts->stack[ ts->size - 1 ]->size != 3 )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Attempt to slice with a parameter not a vector of length 3." );

      u32 start = *( ts->stack[ ts->size - 1 ]->data +
                     ts->stack[ ts->size - 1 ]->offset );
      u32 end =
        *( ts->stack[ ts->size - 1 ]->data + ts->stack[ ts->size - 1 ]->offset +
           ts->stack[ ts->size - 1 ]->strides[ 0 ] );
      u32 axis =
        *( ts->stack[ ts->size - 1 ]->data + ts->stack[ ts->size - 1 ]->offset +
           ts->stack[ ts->size - 1 ]->strides[ 0 ] * 2 );
      pop( ts );
      tensorSlice( ts, ts->size - 1, axis, start, end );
      // dbg( "%s %u %u", "transpose", axis1, axis2 );
      break;
    case TOP: {
      f32* ssize = mem( 1, f32 );
      *ssize = ts->size;
      push( ts, newTensor( 0, NULL, ssize ) );
      // dbg( "%s %u %u", "size", axis1, axis2 );
      break;
    }
    case ADDITIVE: {
      additive = !additive;
      break;
    }
    case KEYS: {
      f32* data = mem( SDL_NUM_SCANCODES, f32 );
      //mainPoll();
      const u8* ks = keys;
      u32 size = SDL_NUM_SCANCODES;
      for( u32 i = 0; i < SDL_NUM_SCANCODES; ++i )
        data[ i ] = ks[ i ];
      push( ts, newTensor( 1, &size, data ) );
      break;
    }
    case SET:
#ifdef DBG
      check_memory_leaks();
#endif
      if( !ts->size )
        error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Empty stack during set statement." );
      if( !s->var.size ){
        if( p->bigvarts[ s->var.index ] )
          deleteTensor( p->bigvarts[ s->var.index ] );
        p->bigvarts[ s->var.index ] = ts->stack[ ts->size - 1 ];
        ts->stack[ --ts->size ] = NULL;
        takeOwnership( p->bigvarts[ s->var.index ] );

      }else{
        if( ( s->var.size <= 4 && ts->stack[ ts->size - 1 ]->rank != 1 ) ||
            ( s->var.size == 16 && ts->stack[ ts->size - 1 ]->rank != 2 ) )
          error( "%s:%u command %u: %s", s->filename, s->linenum, s->commandnum, "Incorrect rank during set statement." );
        if( s->var.size != ts->stack[ ts->size - 1 ]->size ){
          // dbg( "%u %u", s->var.size, ts->stack[ ts->size - 1 ]->size );
          error( "Incorrect size %u during set statement. Expecting %u.",
                 s->var.size, ts->stack[ ts->size - 1 ]->size );
        }
	
        tensorToHostMemory( ts->stack[ ts->size - 1 ] );
        f32* uniform = p->varBlock + p->varOffsets[ s->var.index ];
        if( s->var.size <= 4 )
          for( s32 i = 0; i < s->var.size; ++i )
            uniform[ i ] = *( ts->stack[ ts->size - 1 ]->data +
                              ts->stack[ ts->size - 1 ]->offset +
                              ts->stack[ ts->size - 1 ]->strides[ 0 ] * i );
        else
          for( u32 i = 0; i < 4; ++i )
            for( u32 j = 0; j < 4; ++j )
              uniform[ i * 4 + j ] =
                *( ts->stack[ ts->size - 1 ]->data +
                   ts->stack[ ts->size - 1 ]->offset +
                   ts->stack[ ts->size - 1 ]->strides[ 0 ] * i +
                   ts->stack[ ts->size - 1 ]->strides[ 1 ] * j );
        for( u32 i = 0; i < p->numComputes; ++i ){
          glUseProgram( p->computes[ i ]->program );
          switch( p->varSizes[ s->var.index ] ){
          case 1:
            glUniform1fv( p->computes[ i ]->uniformLocs[ s->var.index ],
                          1,
                          p->varBlock + p->varOffsets[ s->var.index ] );
            break;
          case 2:
            glUniform2fv( p->computes[ i ]->uniformLocs[ s->var.index ],
                          1,
                          p->varBlock + p->varOffsets[ s->var.index ] );
            break;
          case 3:
            glUniform3fv( p->computes[ i ]->uniformLocs[ s->var.index ],
                          1,
                          p->varBlock + p->varOffsets[ s->var.index ] );
            break;
          case 4:
            glUniform4fv( p->computes[ i ]->uniformLocs[ s->var.index ],
                          1,
                          p->varBlock + p->varOffsets[ s->var.index ] );
            break;
          case 16:
            glUniformMatrix4fv( p->computes[ i ]->uniformLocs[ s->var.index ],
                                1,
                                GL_TRUE,
                                p->varBlock + p->varOffsets[ s->var.index ] );
            break;
          default:
            error( "%s", "Logic error in Atlas! Bad variable size." );
          }
        }
	
        pop( ts );
        // dbg( "%s", "set" );
      }
      break;
    case GET: {
      if( !s->var.size ){
        tensor* t = copyTensor( p->bigvarts[ s->var.index ] );
        takeOwnership( t );
        push( ts, t );
      }else{
        static const u32 shape1[ 4 ] = { 1 };
        static const u32 shape2[ 4 ] = { 2 };
        static const u32 shape3[ 4 ] = { 3 };
        static const u32 shape4[ 4 ] = { 4 };
        static const u32 shape16[ 2 ] = { 4, 4 };
        const u32* shape;
        u32 rank = 1;
        switch( p->varSizes[ s->var.index ] ){
        case 1:
          shape = shape1;
          break;
        case 2:
          shape = shape2;
          break;
        case 3:
          shape = shape3;
          break;
        case 4:
          shape = shape4;
          break;
        case 16:
          shape = shape16;
          rank = 2;
          break;
        default:
          error( "%s %u.",
                 "Logic error in atlas! Bad p->varSizes[ s->var.index ]",
                 p->varSizes[ s->var.index ] );
        }
        tensor* t =
          newTensor( rank, shape, p->varBlock + p->varOffsets[ s->var.index ] );
        t->ownsData = false;  // Ensure the tensor does not own the data
        push( ts, t );
      }
      // dbg( "%s", "get" );
      break;
    }
    case QUIT:
      // dbg( "%s", "exit" );
      return false;
    default:
      error( "%s", "Logic error in Atlas!" );
    }
  }
  return true;
}
