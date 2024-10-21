////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4-o1.      //
////////////////////////////////////////////////////////////////////////////////


#include "Atlas.h" 


// Function to compute the length of a number when converted to string
int numLength( f32 num ){
  char buffer[ 50 ];
  int length = snprintf( buffer, sizeof( buffer ), "%g", num );
  return length;
}

// Function to compute the product of elements in an array slice
int product( int* arr, int start, int end ){
  int p = 1;
  for( int i = start; i < end; i++ ){
    p *= arr[ i ];
  }
  return p;
}

// Function to repeat a character 'c' for 'count' times and return the string
char* repeatChar( const char* c, int count ){
  // Since c can be multi-byte, we need to get its length in bytes
  int charLen = strlen( c );
  int totalLen = charLen * count;
  char* str = (char*)mem( totalLen + 1, u8 );
  for( int i = 0; i < count; i++ ){
    memcpy( str + i * charLen, c, charLen );
  }
  str[ totalLen ] = '\0';
  return str;
}

// Custom strdup function
char* custom_strdup( const char* s ){
  size_t len = strlen( s );
  char* dup = (char*)mem( len + 1, u8 );
  if( dup ){
    strcpy( dup, s );
  }
  return dup;
}

// Function to split a string by '\n' and return an array of strings
char** splitString( char* str, int* lineCount ){
  int capacity = 10;
  char** lines = (char**)mem( sizeof( char* ) * capacity, u8 );
  *lineCount = 0;

  char* line = strtok( str, "\n" );
  while (line != NULL ){
    if( *lineCount >= capacity ){
      capacity *= 2;
      lines = (char**)realloc( lines, sizeof( char*) * capacity );
    }
    lines[ *lineCount ] = custom_strdup( line );
    (*lineCount)++;
    line = strtok( NULL, "\n" );
  }
  return lines;
}

// Function to calculate the display width of a string (number of characters)
int displayWidth( const char* str ){
  int width = 0;
  const unsigned char* s = (const unsigned char*)str;
  while( *s ){
    if((*s & 0x80) == 0 ){
      // 1-byte character
      s += 1;
    } else if( (*s & 0xE0) == 0xC0){
      // 2-byte character
      s += 2;
    } else if( (*s & 0xF0) == 0xE0){
      // 3-byte character
      s += 3;
    } else if( (*s & 0xF8) == 0xF0){
      // 4-byte character
      s += 4;
    } else {
      // Invalid UTF-8 character
      s += 1;
    }
    width++;
  }
  return width;
}

// The recursive helper function
char* helper( int dimIndex, int offset, int depth, int* shape, int shape_length,
	      u8* data, int data_length, int maxNumLength ){
  if( dimIndex == shape_length - 1 ){
    // Base case: last dimension 
    int num_elements = shape[ dimIndex ];
    int total_length = num_elements * maxNumLength + (num_elements - 1 );
    char* result = (char*)mem( total_length + 1, u8 );
    char* ptr = result;

    for( int i = 0; i < num_elements; i++ ){
      char numStr[ 50 ];
      snprintf( numStr, sizeof( numStr ), "%u", data[ offset + i ] );
      int numStr_len = strlen( numStr );

      // Pad the number string to maxNumLength
      int pad_len = maxNumLength - numLength( data[ offset + i ] );
      // Use numLength for correct padding
      memcpy( ptr, numStr, numStr_len );
      ptr += numStr_len;

      for( int k = 0; k < pad_len; k++ ){
	*ptr = ' ';
	ptr++;
      }
      if( i < num_elements - 1 ){
	*ptr = ' ';
	ptr++;
      }
    }
    *ptr = '\0';
    return result;
  } else {
    // Recursive case
    int isHorizontal = (depth % 2 == 0 );
    int size = product( shape, dimIndex + 1, shape_length );
    int num_blocks = shape[ dimIndex ];

    char** blocks = (char**)mem( sizeof( char*) * num_blocks, u8 );
    for( int i = 0; i < num_blocks; i++ ){
      blocks[ i ] = helper( dimIndex + 1, offset + i * size,
			    depth + 1, shape, shape_length, data,
			    data_length, maxNumLength );
    }

    if( isHorizontal ){
      // Stack horizontally with boxes
      int maxHeight = 0;
      char*** blockLinesArray =
	(char***)mem( sizeof( char**) * num_blocks, u8 );
      int* blockLineCounts = (int*)mem( sizeof( int ) * num_blocks, u8 );

      for( int i = 0; i < num_blocks; i++ ){
	// Split block into lines
	char* blockCopy = custom_strdup( blocks[ i ] );
	int lineCount = 0;
	char** lines = splitString( blockCopy, &lineCount );
	unmem( blockCopy );

	// Find max line length (in display width )
	int maxLineDisplayWidth = 0;
	for( int j = 0; j < lineCount; j++ ){
	  int len = displayWidth( lines[ j ] );
	  if( len > maxLineDisplayWidth ){
	    maxLineDisplayWidth = len;
	  }
	}
	int maxLength = (maxLineDisplayWidth > maxNumLength ) ?
	  maxLineDisplayWidth : maxNumLength;

	// Build box
	int boxDisplayWidth = maxLength + 4; // 4 for '┌┐' and padding
	// Estimate, since UTF-8 characters can be up to 3 bytes
	int boxByteWidth = boxDisplayWidth * 3; 
	char* top = (char*)mem( boxByteWidth + 1, u8 );
	char* bottom = (char*)mem( boxByteWidth + 1, u8 );
	char* lineFill = repeatChar( "─", maxLength + 2 );
	snprintf( top, boxByteWidth + 1, "┌%s┐", lineFill );
	snprintf( bottom, boxByteWidth + 1, "└%s┘", lineFill );
	unmem( lineFill );

	char** blockLines = (char**)mem( lineCount + 2, char* );
	blockLines[ 0 ] = top;

	for( int j = 0; j < lineCount; j++ ){
	  int padding = maxLength - displayWidth( lines[ j ] );
	  int middleLineByteWidth = boxByteWidth + 1;
	  char* middleLine = (char*)mem( middleLineByteWidth, u8 );
	  snprintf( middleLine, middleLineByteWidth, "│ %s%*s │",
		    lines[ j ], padding, "" );
	  blockLines[ j + 1 ] = middleLine;
	  unmem( lines[ j ] );
	}
	unmem( lines );
	blockLines[ lineCount + 1 ] = bottom;

	blockLinesArray[ i ] = blockLines;
	blockLineCounts[ i ] = lineCount + 2;

	if( blockLineCounts[ i ] > maxHeight ){
	  maxHeight = blockLineCounts[ i ];
	}
      }

      // Pad blocks to have the same height
      for( int i = 0; i < num_blocks; i++ ){
	int currentHeight = blockLineCounts[ i ];
	if( currentHeight < maxHeight ){
	  int width = strlen( blockLinesArray[ i ][ 0 ] );
	  char* emptyLine = (char*)mem( width + 1, u8 );
	  memset( emptyLine, ' ', width );
	  emptyLine[ width ] = '\0';

	  blockLinesArray[ i ] = (char**)realloc( blockLinesArray[ i ],
						  sizeof( char*) * maxHeight );
	  for( int j = currentHeight; j < maxHeight; j++ ){
	    blockLinesArray[ i ][ j ] = custom_strdup( emptyLine );
	  }
	  blockLineCounts[ i ] = maxHeight;
	  unmem( emptyLine );
	}
      }

      // Combine blocks line by line
      char** combinedLines = (char**)mem( sizeof( char*) * maxHeight, u8 );
      for( int i = 0; i < maxHeight; i++ ){
	int totalLineLength = 0;
	for( int j = 0; j < num_blocks; j++ ){
	  totalLineLength += strlen( blockLinesArray[ j ][ i ] );
	  if( j < num_blocks - 1 ){
	    totalLineLength += 1; // space between blocks
	  }
	}
	combinedLines[ i ] = (char*)mem( totalLineLength + 1, u8 );
	combinedLines[ i ][ 0 ] = '\0';
	for( int j = 0; j < num_blocks; j++ ){
	  strcat( combinedLines[ i ], blockLinesArray[ j ][ i ] );
	  if( j < num_blocks - 1 ){
	    strcat( combinedLines[ i ], " " );
	  }
	}
      }

      // Join combined lines
      int totalLength = 0;
      for( int i = 0; i < maxHeight; i++ ){
	totalLength += strlen( combinedLines[ i ] );
	if( i < maxHeight - 1 ){
	  totalLength += 1; // for '\n'
	}
      }
      char* result = (char*)mem( totalLength + 1, u8 );
      result[ 0 ] = '\0';
      for( int i = 0; i < maxHeight; i++ ){
	strcat( result, combinedLines[ i ] );
	if( i < maxHeight - 1 ){
	  strcat( result, "\n" );
	}
	unmem( combinedLines[ i ] );
      }
      unmem( combinedLines );

      // Free memory
      for( int i = 0; i < num_blocks; i++ ){
	for( int j = 0; j < blockLineCounts[ i ]; j++ ){
	  unmem( blockLinesArray[ i ][ j ] );
	}
	unmem( blockLinesArray[ i ] );
      }
      unmem( blockLinesArray );
      unmem( blockLineCounts );

      for( int i = 0; i < num_blocks; i++ ){
	unmem( blocks[ i ] );
      }
      unmem( blocks );

      return result;
    } else {
      // Stack vertically with boxes
      int maxWidth = 0;
      char*** blockLinesArray =
	(char***)mem( sizeof( char**) * num_blocks, u8 );
      int* blockLineCounts = (int*)mem( sizeof( int ) * num_blocks, u8 );

      for( int i = 0; i < num_blocks; i++ ){
	// Split block into lines
	char* blockCopy = custom_strdup( blocks[ i ] );
	int lineCount = 0;
	char** lines = splitString( blockCopy, &lineCount );
	unmem( blockCopy );

	// Find max line length
	int maxLineDisplayWidth = 0;
	for( int j = 0; j < lineCount; j++ ){
	  int len = displayWidth( lines[ j ] );
	  if( len > maxLineDisplayWidth ){
	    maxLineDisplayWidth = len;
	  }
	}
	int maxLength =
	  ( maxLineDisplayWidth > maxNumLength ) ?
	  maxLineDisplayWidth : maxNumLength;
	if( maxLength > maxWidth ){
	  maxWidth = maxLength;
	}

	// Build box
	int boxDisplayWidth = maxLength + 4; // 4 for '┌┐' and padding
	int boxByteWidth = boxDisplayWidth * 3;
	char* top = (char*)mem( boxByteWidth + 1, u8 );
	char* bottom = (char*)mem( boxByteWidth + 1, u8 );
	char* lineFill = repeatChar( "─", maxLength + 2 );
	snprintf( top, boxByteWidth + 1, "┌%s┐", lineFill );
	snprintf( bottom, boxByteWidth + 1, "└%s┘", lineFill );
	unmem( lineFill );

	char** blockLines = (char**)mem( lineCount + 2, char*);
	blockLines[ 0 ] = top;

	for( int j = 0; j < lineCount; j++ ){
	  int padding = maxLength - displayWidth( lines[ j ] );
	  int middleLineByteWidth = boxByteWidth + 1;
	  char* middleLine = (char*)mem( middleLineByteWidth, u8 );
	  snprintf( middleLine, middleLineByteWidth, "│ %s%*s │",
		    lines[ j ], padding, "" );
	  blockLines[ j + 1 ] = middleLine;
	  unmem( lines[ j ] );
	}
	unmem( lines );
	blockLines[ lineCount + 1 ] = bottom;

	blockLinesArray[ i ] = blockLines;
	blockLineCounts[ i ] = lineCount + 2;
      }

      // Pad blocks to have the same width
      int maxBoxByteWidth = (maxWidth + 4 ) * 3;
      for( int i = 0; i < num_blocks; i++ ){
	int currentWidth = strlen( blockLinesArray[ i ][ 0 ] );
	if( currentWidth < maxBoxByteWidth ){
	  int extraWidth = maxBoxByteWidth - currentWidth;
	  for( int j = 0; j < blockLineCounts[ i ]; j++ ){
	    blockLinesArray[ i ][ j ] =
	      (char*)realloc( blockLinesArray[ i ][ j ], maxBoxByteWidth + 1 );
	    memset( blockLinesArray[ i ][ j ] + currentWidth, ' ', extraWidth );
	    blockLinesArray[ i ][ j ][ maxBoxByteWidth ] = '\0';
	  }
	}
      }

      // Combine blocks vertically
      int totalLines = 0;
      for( int i = 0; i < num_blocks; i++ ){
	totalLines += blockLineCounts[ i ];
      }
      char** combinedLines = (char**)mem( sizeof( char*) * totalLines, u8 );
      int index = 0;
      for( int i = 0; i < num_blocks; i++ ){
	for( int j = 0; j < blockLineCounts[ i ]; j++ ){
	  combinedLines[ index++ ] = custom_strdup( blockLinesArray[ i ][ j ] );
	}
      }

      // Join combined lines
      int totalLength = 0;
      for( int i = 0; i < totalLines; i++ ){
	totalLength += strlen( combinedLines[ i ] );
	if( i < totalLines - 1 ){
	  totalLength += 1; // for '\n'
	}
      }
      char* result = (char*)mem( totalLength + 1, u8 );
      result[ 0 ] = '\0';
      for( int i = 0; i < totalLines; i++ ){
	strcat( result, combinedLines[ i ] );
	if( i < totalLines - 1 ){
	  strcat( result, "\n" );
	}
	unmem( combinedLines[ i ] );
      }
      unmem( combinedLines );

      // Free memory
      for( int i = 0; i < num_blocks; i++ ){
	for( int j = 0; j < blockLineCounts[ i ]; j++ ){
	  unmem( blockLinesArray[ i ][ j ] );
	}
	unmem( blockLinesArray[ i ] );
      }
      unmem( blockLinesArray );
      unmem( blockLineCounts );

      for( int i = 0; i < num_blocks; i++ ){
	unmem( blocks[ i ] );
      }
      unmem( blocks );

      return result;
    }
  }
}

tensor* newTensor( u32 rank, u64 size, u64* shape, u8* data ){
  tensor* ret = mem( 1, tensor );
  if( rank ){
    if( !shape )
      error( "A tensor with non-zero rank was given with no shape." );
    ret->rank = rank;
    ret->size = size;
    ret->shape = mem( rank, u64 );
    memcpy( ret->shape, shape, sizeof( u64 ) * rank );
    ret->data = mem( size, f32 );
    memcpy( ret->data, data, sizeof( f32 ) * size );
  } else {
    ret->rank = 0;
    ret->size = 1;
    ret->shape = NULL;
    ret->data = mem( 1, f32 );
    *(ret->data) = *data;
  }
  ret->references = 1;
  return ret;
}

void deleteTensor( tensor* t ){
  --t->references;
  if( !t->references ){
    if( t->shape )
      unmem( t->shape );
    if( t->data )
      unmem( t->data );
  }
  unmem( t );  
}

// Main function to format tensor data
char* formatTensorData( const tensor* t ){
  // Create shape = [ 1, ...shapeArg ]
  u32 shapeArg_length = t->rank;
  u64* shapeArg = t->shape;
  u32 data_length = t->size;
  u8* data = t->data;
  
  int shape_length = shapeArg_length + 1;
  int* shape = (int*)mem( shape_length, int );
  shape[ 0 ] = 1;
  for( int i = 0; i < shapeArg_length; i++ ){
    shape[ i + 1 ] = shapeArg[ i ];
  }
  // Compute maxNumLength
  int maxNumLength = 0;
  for( int i = 0; i < data_length; i++ ){
    int len = numLength( data[ i ] );
    if( len > maxNumLength ){
      maxNumLength = len;
    }
  }
  // Call helper function
  char* result = helper( 0, 0, 0, shape, shape_length, data, data_length,
			 maxNumLength );
  unmem( shape );
  return result; // Remember to free this string after use
}
