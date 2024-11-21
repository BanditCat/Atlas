////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4-o1.      //
////////////////////////////////////////////////////////////////////////////////



#include "Atlas.h"



// Function to compute the length of a number when converted to string
u32 numLength( f32 num ){
  char buffer[ 50 ];
  u32 length = snprintf( buffer, sizeof( buffer ), "%.2f", num );
  return length;
}


// Function to compute the product of elements in an array slice
u32 product( u32* arr, u32 start, u32 end ){
  u32 p = 1;
  for( u32 i = start; i < end; i++ ){
    p *= arr[ i ];
  }
  return p;
}

// Function to repeat a character 'c' for 'count' times and return the string
char* repeatChar( const char* c, u32 count ){
  // Since c can be multi-byte, we need to get its length in bytes
  u32 charLen = strlen( c );
  u32 totalLen = charLen * count;
  char* str = (char*)mem( totalLen + 1, u8 );
  for( u32 i = 0; i < count; i++ ){
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
char** splitString( char* str, u32* lineCount ){
  u32 capacity = 10;
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
u32 displayWidth( const char* str ){
  u32 width = 0;
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


// Translate function: converts linear index to properly indexed value
s32 translateIndex( u32 linearIndex, u32* shape, u32* strides, u32 rank ){
  if( rank <= 1 )
    return linearIndex;
  u32 li = linearIndex;
  u32 tensorIndex[ 5 ]; // Assumes a maximum rank of 4. Adjust as needed.
  s32 newIndex = 0;
  for( int i = rank - 1; i >= 0; --i ){
    tensorIndex[ i ] =  li % shape[ i ];
    li /= shape[ i ];
  }
  for( u32 i = 0; i < rank; ++i )
    newIndex += tensorIndex[ i ] * strides[ i ];
 
  return newIndex; 
}


// The recursive helper function
char* helper( u32 dimIndex, u32 offset, u32 depth, u32* shape, u32* strides, u32 shape_length,
	      f32* data, u32 data_length, u32 maxNumLength, const tensor* t ){

  if( dimIndex == shape_length - 1 ){
    // Base case: last dimension
    u32 num_elements = shape[ dimIndex ];
    u32 total_length = num_elements * maxNumLength + (num_elements - 1 );
    char* result = (char*)mem( total_length + 1, u8 );
    char* ptr = result;

    for( u32 i = 0; i < num_elements; i++ ){
      char numStr[ 50 ];

      u32 translatedIndex = translateIndex( offset + i, shape, strides, shape_length );
      snprintf( numStr, sizeof( numStr ), "%.2f", data[ t->offset + translatedIndex ] );
      u32 numStr_len = strlen( numStr );

      // Pad the number string to maxNumLength
      u32 pad_len = maxNumLength - numLength( data[ t->offset + translatedIndex ] );
      // Use numLength for correct padding
      memcpy( ptr, numStr, numStr_len );
      ptr += numStr_len;

      for( u32 k = 0; k < pad_len; k++ ){
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
    u32 isHorizontal = (depth % 2 == 0 );
    u32 size = product( shape, dimIndex + 1, shape_length );
    u32 num_blocks = shape[ dimIndex ];

    char** blocks = (char**)mem( sizeof( char*) * num_blocks, u8 );
    for( u32 i = 0; i < num_blocks; i++ ){
      blocks[ i ] = helper( dimIndex + 1, offset + i * size,
			    depth + 1, shape, strides, shape_length, data,
			    data_length, maxNumLength, t );
    }

    if( isHorizontal ){
      // Stack horizontally with boxes
      u32 maxHeight = 0;
      char*** blockLinesArray =
	(char***)mem( sizeof( char**) * num_blocks, u8 );
      u32* blockLineCounts = (u32*)mem( sizeof( u32 ) * num_blocks, u8 );

      for( u32 i = 0; i < num_blocks; i++ ){
	// Split block into lines
	char* blockCopy = custom_strdup( blocks[ i ] );
	u32 lineCount = 0;
	char** lines = splitString( blockCopy, &lineCount );
	unmem( blockCopy );

	// Find max line length (in display width )
	u32 maxLineDisplayWidth = 0;
	for( u32 j = 0; j < lineCount; j++ ){
	  u32 len = displayWidth( lines[ j ] );
	  if( len > maxLineDisplayWidth ){
	    maxLineDisplayWidth = len;
	  }
	}
	u32 maxLength = ( maxLineDisplayWidth > maxNumLength ) ?
	  maxLineDisplayWidth : maxNumLength;

	// Build box
	u32 boxDisplayWidth = maxLength + 4; // 4 for '┌┐' and padding
	// Estimate, since UTF-8 characters can be up to 3 bytes
	u32 boxByteWidth = boxDisplayWidth * 3;
	char* top = (char*)mem( boxByteWidth + 1, u8 );
	char* bottom = (char*)mem( boxByteWidth + 1, u8 );
	char* lineFill = repeatChar( "─", maxLength + 2 );
	snprintf( top, boxByteWidth + 1, "┌%s┐", lineFill );
	snprintf( bottom, boxByteWidth + 1, "└%s┘", lineFill );
	unmem( lineFill );

	char** blockLines = (char**)mem( lineCount + 2, char* );
	blockLines[ 0 ] = top;

	for( u32 j = 0; j < lineCount; j++ ){
	  u32 padding = maxLength - displayWidth( lines[ j ] );
	  u32 middleLineByteWidth = boxByteWidth + 1;
	  char* middleLine = (char*)mem( middleLineByteWidth, u8 );
	  snprintf( middleLine, middleLineByteWidth, "│ %s%*s │",
		    lines[ j ], (int)padding, "" );
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
      for( u32 i = 0; i < num_blocks; i++ ){
	u32 currentHeight = blockLineCounts[ i ];
	if( currentHeight < maxHeight ){
	  u32 width = strlen( blockLinesArray[ i ][ 0 ] );
	  char* emptyLine = (char*)mem( width + 1, u8 );
	  memset( emptyLine, ' ', width );
	  emptyLine[ width ] = '\0';

	  blockLinesArray[ i ] = (char**)realloc( blockLinesArray[ i ],
						  sizeof( char*) * maxHeight );
	  for( u32 j = currentHeight; j < maxHeight; j++ ){
	    blockLinesArray[ i ][ j ] = custom_strdup( emptyLine );
	  }
	  blockLineCounts[ i ] = maxHeight;
	  unmem( emptyLine );
	}
      }

      // Combine blocks line by line
      char** combinedLines = (char**)mem( sizeof( char*) * maxHeight, u8 );
      for( u32 i = 0; i < maxHeight; i++ ){
	u32 totalLineLength = 0;
	for( u32 j = 0; j < num_blocks; j++ ){
	  totalLineLength += strlen( blockLinesArray[ j ][ i ] );
	  if( j < num_blocks - 1 ){
	    totalLineLength += 1; // space between blocks
	  }
	}
	combinedLines[ i ] = (char*)mem( totalLineLength + 1, u8 );
	combinedLines[ i ][ 0 ] = '\0';
	for( u32 j = 0; j < num_blocks; j++ ){
	  strcat( combinedLines[ i ], blockLinesArray[ j ][ i ] );
	  if( j < num_blocks - 1 ){
	    strcat( combinedLines[ i ], " " );
	  }
	}
      }

      // Join combined lines
      u32 totalLength = 0;
      for( u32 i = 0; i < maxHeight; i++ ){
	totalLength += strlen( combinedLines[ i ] );
	if( i < maxHeight - 1 ){
	  totalLength += 1; // for '\n'
	}
      }
      char* result = (char*)mem( totalLength + 1, u8 );
      result[ 0 ] = '\0';
      for( u32 i = 0; i < maxHeight; i++ ){
	strcat( result, combinedLines[ i ] );
	if( i < maxHeight - 1 ){
	  strcat( result, "\n" );
	}
	unmem( combinedLines[ i ] );
      }
      unmem( combinedLines );

      // Free memory
      for( u32 i = 0; i < num_blocks; i++ ){
	for( u32 j = 0; j < blockLineCounts[ i ]; j++ ){
	  unmem( blockLinesArray[ i ][ j ] );
	}
	unmem( blockLinesArray[ i ] );
      }
      unmem( blockLinesArray );
      unmem( blockLineCounts );

      for( u32 i = 0; i < num_blocks; i++ ){
	unmem( blocks[ i ] );
      }
      unmem( blocks );

      return result;
    } else {
      // Stack vertically with boxes
      u32 maxWidth = 0;
      char*** blockLinesArray =
	(char***)mem( sizeof( char**) * num_blocks, u8 );
      u32* blockLineCounts = (u32*)mem( sizeof( u32 ) * num_blocks, u8 );

      for( u32 i = 0; i < num_blocks; i++ ){
	// Split block into lines
	char* blockCopy = custom_strdup( blocks[ i ] );
	u32 lineCount = 0;
	char** lines = splitString( blockCopy, &lineCount );
	unmem( blockCopy );

	// Find max line length
	u32 maxLineDisplayWidth = 0;
	for( u32 j = 0; j < lineCount; j++ ){
	  u32 len = displayWidth( lines[ j ] );
	  if( len > maxLineDisplayWidth ){
	    maxLineDisplayWidth = len;
	  }
	}
	u32 maxLength =
	  ( maxLineDisplayWidth > maxNumLength ) ?
	  maxLineDisplayWidth : maxNumLength;
	if( maxLength > maxWidth ){
	  maxWidth = maxLength;
	}

	// Build box
	u32 boxDisplayWidth = maxLength + 4; // 4 for '┌┐' and padding
	u32 boxByteWidth = boxDisplayWidth * 3;
	char* top = (char*)mem( boxByteWidth + 1, u8 );
	char* bottom = (char*)mem( boxByteWidth + 1, u8 );
	char* lineFill = repeatChar( "─", maxLength + 2 );
	snprintf( top, boxByteWidth + 1, "┌%s┐", lineFill );
	snprintf( bottom, boxByteWidth + 1, "└%s┘", lineFill );
	unmem( lineFill );

	char** blockLines = (char**)mem( lineCount + 2, char*);
	blockLines[ 0 ] = top;

	for( u32 j = 0; j < lineCount; j++ ){
	  u32 padding = maxLength - displayWidth( lines[ j ] );
	  u32 middleLineByteWidth = boxByteWidth + 1;
	  char* middleLine = (char*)mem( middleLineByteWidth, u8 );
	  snprintf( middleLine, middleLineByteWidth, "│ %s%*s │",
		    lines[ j ], (int)padding, "" );
	  blockLines[ j + 1 ] = middleLine;
	  unmem( lines[ j ] );
	}
	unmem( lines );
	blockLines[ lineCount + 1 ] = bottom;

	blockLinesArray[ i ] = blockLines;
	blockLineCounts[ i ] = lineCount + 2;
      }

      // Pad blocks to have the same width
      u32 maxBoxByteWidth = (maxWidth + 4 ) * 3;
      for( u32 i = 0; i < num_blocks; i++ ){
	u32 currentWidth = strlen( blockLinesArray[ i ][ 0 ] );
	if( currentWidth < maxBoxByteWidth ){
	  u32 extraWidth = maxBoxByteWidth - currentWidth;
	  for( u32 j = 0; j < blockLineCounts[ i ]; j++ ){
	    blockLinesArray[ i ][ j ] =
	      (char*)realloc( blockLinesArray[ i ][ j ], maxBoxByteWidth + 1 );
	    memset( blockLinesArray[ i ][ j ] + currentWidth, ' ', extraWidth );
	    blockLinesArray[ i ][ j ][ maxBoxByteWidth ] = '\0';
	  }
	}
      }

      // Combine blocks vertically
      u32 totalLines = 0;
      for( u32 i = 0; i < num_blocks; i++ ){
	totalLines += blockLineCounts[ i ];
      }
      char** combinedLines = (char**)mem( sizeof( char*) * totalLines, u8 );
      u32 index = 0;
      for( u32 i = 0; i < num_blocks; i++ ){
	for( u32 j = 0; j < blockLineCounts[ i ]; j++ ){
	  combinedLines[ index++ ] = custom_strdup( blockLinesArray[ i ][ j ] );
	}
      }

      // Join combined lines
      u32 totalLength = 0;
      for( u32 i = 0; i < totalLines; i++ ){
	totalLength += strlen( combinedLines[ i ] );
	if( i < totalLines - 1 ){
	  totalLength += 1; // for '\n'
	}
      }
      char* result = (char*)mem( totalLength + 1, u8 );
      result[ 0 ] = '\0';
      for( u32 i = 0; i < totalLines; i++ ){
	strcat( result, combinedLines[ i ] );
	if( i < totalLines - 1 ){
	  strcat( result, "\n" );
	}
	unmem( combinedLines[ i ] );
      }
      unmem( combinedLines );

      // Free memory
      for( u32 i = 0; i < num_blocks; i++ ){
	for( u32 j = 0; j < blockLineCounts[ i ]; j++ ){
	  unmem( blockLinesArray[ i ][ j ] );
	}
	unmem( blockLinesArray[ i ] );
      }
      unmem( blockLinesArray );
      unmem( blockLineCounts );

      for( u32 i = 0; i < num_blocks; i++ ){
	unmem( blocks[ i ] );
      }
      unmem( blocks );

      return result;
    }
  }
}

// Main function to format tensor data
char* formatTensorData( const tensor* t ){
  // Create shape = [ 1, ...shapeArg ]
  u32 shapeArg_length = t->rank;
  u32 const* shapeArg = t->shape;
  u32 data_length = t->size;
  f32* data = tensorToHostMemory( t );
  
  u32 shape_length = shapeArg_length + 1;
  u32* shape = (u32*)mem( shape_length, u32 );
  u32* strides = (u32*)mem( shape_length, u32 );
  shape[ 0 ] = 1;
  for( u32 i = 0; i < shapeArg_length; ++i ){
    shape[ i + 1 ] = shapeArg[ i ];
    strides[ i + 1 ] = t->strides[ i ];
  }
  strides[ 0 ] = strides[ 1 ] * shape[ 1 ];
  // Compute maxNumLength
  u32 maxNumLength = 0;
  for( u32 i = 0; i < data_length; i++ ){
    u32 len = numLength( data[ i ] );
    if( len > maxNumLength ){
      maxNumLength = len;
    }
  }
  // Call helper function
  char* result = helper( 0, 0, 0, shape, strides, shape_length, data, data_length,
			 maxNumLength, t );
  unmem( shape );
  unmem( strides );
  unmem( data );
  return result; // Remember to free this string after use
}
