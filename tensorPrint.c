////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4-o1.      //
////////////////////////////////////////////////////////////////////////////////



#include "Atlas.h"



// Function to compute the length of a number when converted to string
u64 numLength( u8 num ){
  char buffer[ 50 ];
  u64 length = snprintf( buffer, sizeof( buffer ), "%u", num );
  return length;
}

// Function to compute the product of elements in an array slice
u64 product( u64* arr, u64 start, u64 end ){
  u64 p = 1;
  for( u64 i = start; i < end; i++ ){
    p *= arr[ i ];
  }
  return p;
}

// Function to repeat a character 'c' for 'count' times and return the string
char* repeatChar( const char* c, u64 count ){
  // Since c can be multi-byte, we need to get its length in bytes
  u64 charLen = strlen( c );
  u64 totalLen = charLen * count;
  char* str = (char*)mem( totalLen + 1, u8 );
  for( u64 i = 0; i < count; i++ ){
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
char** splitString( char* str, u64* lineCount ){
  u64 capacity = 10;
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
u64 displayWidth( const char* str ){
  u64 width = 0;
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
char* helper( u64 dimIndex, u64 offset, u64 depth, u64* shape, u64 shape_length,
	      u8* data, u64 data_length, u64 maxNumLength ){
  if( dimIndex == shape_length - 1 ){
    // Base case: last dimension
    u64 num_elements = shape[ dimIndex ];
    u64 total_length = num_elements * maxNumLength + (num_elements - 1 );
    char* result = (char*)mem( total_length + 1, u8 );
    char* ptr = result;

    for( u64 i = 0; i < num_elements; i++ ){
      char numStr[ 50 ];
      snprintf( numStr, sizeof( numStr ), "%u", data[ offset + i ] );
      u64 numStr_len = strlen( numStr );

      // Pad the number string to maxNumLength
      u64 pad_len = maxNumLength - numLength( data[ offset + i ] );
      // Use numLength for correct padding
      memcpy( ptr, numStr, numStr_len );
      ptr += numStr_len;

      for( u64 k = 0; k < pad_len; k++ ){
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
    u64 isHorizontal = (depth % 2 == 0 );
    u64 size = product( shape, dimIndex + 1, shape_length );
    u64 num_blocks = shape[ dimIndex ];

    char** blocks = (char**)mem( sizeof( char*) * num_blocks, u8 );
    for( u64 i = 0; i < num_blocks; i++ ){
      blocks[ i ] = helper( dimIndex + 1, offset + i * size,
			    depth + 1, shape, shape_length, data,
			    data_length, maxNumLength );
    }

    if( isHorizontal ){
      // Stack horizontally with boxes
      u64 maxHeight = 0;
      char*** blockLinesArray =
	(char***)mem( sizeof( char**) * num_blocks, u8 );
      u64* blockLineCounts = (u64*)mem( sizeof( u64 ) * num_blocks, u8 );

      for( u64 i = 0; i < num_blocks; i++ ){
	// Split block into lines
	char* blockCopy = custom_strdup( blocks[ i ] );
	u64 lineCount = 0;
	char** lines = splitString( blockCopy, &lineCount );
	unmem( blockCopy );

	// Find max line length (in display width )
	u64 maxLineDisplayWidth = 0;
	for( u64 j = 0; j < lineCount; j++ ){
	  u64 len = displayWidth( lines[ j ] );
	  if( len > maxLineDisplayWidth ){
	    maxLineDisplayWidth = len;
	  }
	}
	u64 maxLength = ( maxLineDisplayWidth > maxNumLength ) ?
	  maxLineDisplayWidth : maxNumLength;

	// Build box
	u64 boxDisplayWidth = maxLength + 4; // 4 for '┌┐' and padding
	// Estimate, since UTF-8 characters can be up to 3 bytes
	u64 boxByteWidth = boxDisplayWidth * 3;
	char* top = (char*)mem( boxByteWidth + 1, u8 );
	char* bottom = (char*)mem( boxByteWidth + 1, u8 );
	char* lineFill = repeatChar( "─", maxLength + 2 );
	snprintf( top, boxByteWidth + 1, "┌%s┐", lineFill );
	snprintf( bottom, boxByteWidth + 1, "└%s┘", lineFill );
	unmem( lineFill );

	char** blockLines = (char**)mem( lineCount + 2, char* );
	blockLines[ 0 ] = top;

	for( u64 j = 0; j < lineCount; j++ ){
	  u64 padding = maxLength - displayWidth( lines[ j ] );
	  u64 middleLineByteWidth = boxByteWidth + 1;
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
      for( u64 i = 0; i < num_blocks; i++ ){
	u64 currentHeight = blockLineCounts[ i ];
	if( currentHeight < maxHeight ){
	  u64 width = strlen( blockLinesArray[ i ][ 0 ] );
	  char* emptyLine = (char*)mem( width + 1, u8 );
	  memset( emptyLine, ' ', width );
	  emptyLine[ width ] = '\0';

	  blockLinesArray[ i ] = (char**)realloc( blockLinesArray[ i ],
						  sizeof( char*) * maxHeight );
	  for( u64 j = currentHeight; j < maxHeight; j++ ){
	    blockLinesArray[ i ][ j ] = custom_strdup( emptyLine );
	  }
	  blockLineCounts[ i ] = maxHeight;
	  unmem( emptyLine );
	}
      }

      // Combine blocks line by line
      char** combinedLines = (char**)mem( sizeof( char*) * maxHeight, u8 );
      for( u64 i = 0; i < maxHeight; i++ ){
	u64 totalLineLength = 0;
	for( u64 j = 0; j < num_blocks; j++ ){
	  totalLineLength += strlen( blockLinesArray[ j ][ i ] );
	  if( j < num_blocks - 1 ){
	    totalLineLength += 1; // space between blocks
	  }
	}
	combinedLines[ i ] = (char*)mem( totalLineLength + 1, u8 );
	combinedLines[ i ][ 0 ] = '\0';
	for( u64 j = 0; j < num_blocks; j++ ){
	  strcat( combinedLines[ i ], blockLinesArray[ j ][ i ] );
	  if( j < num_blocks - 1 ){
	    strcat( combinedLines[ i ], " " );
	  }
	}
      }

      // Join combined lines
      u64 totalLength = 0;
      for( u64 i = 0; i < maxHeight; i++ ){
	totalLength += strlen( combinedLines[ i ] );
	if( i < maxHeight - 1 ){
	  totalLength += 1; // for '\n'
	}
      }
      char* result = (char*)mem( totalLength + 1, u8 );
      result[ 0 ] = '\0';
      for( u64 i = 0; i < maxHeight; i++ ){
	strcat( result, combinedLines[ i ] );
	if( i < maxHeight - 1 ){
	  strcat( result, "\n" );
	}
	unmem( combinedLines[ i ] );
      }
      unmem( combinedLines );

      // Free memory
      for( u64 i = 0; i < num_blocks; i++ ){
	for( u64 j = 0; j < blockLineCounts[ i ]; j++ ){
	  unmem( blockLinesArray[ i ][ j ] );
	}
	unmem( blockLinesArray[ i ] );
      }
      unmem( blockLinesArray );
      unmem( blockLineCounts );

      for( u64 i = 0; i < num_blocks; i++ ){
	unmem( blocks[ i ] );
      }
      unmem( blocks );

      return result;
    } else {
      // Stack vertically with boxes
      u64 maxWidth = 0;
      char*** blockLinesArray =
	(char***)mem( sizeof( char**) * num_blocks, u8 );
      u64* blockLineCounts = (u64*)mem( sizeof( u64 ) * num_blocks, u8 );

      for( u64 i = 0; i < num_blocks; i++ ){
	// Split block into lines
	char* blockCopy = custom_strdup( blocks[ i ] );
	u64 lineCount = 0;
	char** lines = splitString( blockCopy, &lineCount );
	unmem( blockCopy );

	// Find max line length
	u64 maxLineDisplayWidth = 0;
	for( u64 j = 0; j < lineCount; j++ ){
	  u64 len = displayWidth( lines[ j ] );
	  if( len > maxLineDisplayWidth ){
	    maxLineDisplayWidth = len;
	  }
	}
	u64 maxLength =
	  ( maxLineDisplayWidth > maxNumLength ) ?
	  maxLineDisplayWidth : maxNumLength;
	if( maxLength > maxWidth ){
	  maxWidth = maxLength;
	}

	// Build box
	u64 boxDisplayWidth = maxLength + 4; // 4 for '┌┐' and padding
	u64 boxByteWidth = boxDisplayWidth * 3;
	char* top = (char*)mem( boxByteWidth + 1, u8 );
	char* bottom = (char*)mem( boxByteWidth + 1, u8 );
	char* lineFill = repeatChar( "─", maxLength + 2 );
	snprintf( top, boxByteWidth + 1, "┌%s┐", lineFill );
	snprintf( bottom, boxByteWidth + 1, "└%s┘", lineFill );
	unmem( lineFill );

	char** blockLines = (char**)mem( lineCount + 2, char*);
	blockLines[ 0 ] = top;

	for( u64 j = 0; j < lineCount; j++ ){
	  u64 padding = maxLength - displayWidth( lines[ j ] );
	  u64 middleLineByteWidth = boxByteWidth + 1;
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
      u64 maxBoxByteWidth = (maxWidth + 4 ) * 3;
      for( u64 i = 0; i < num_blocks; i++ ){
	u64 currentWidth = strlen( blockLinesArray[ i ][ 0 ] );
	if( currentWidth < maxBoxByteWidth ){
	  u64 extraWidth = maxBoxByteWidth - currentWidth;
	  for( u64 j = 0; j < blockLineCounts[ i ]; j++ ){
	    blockLinesArray[ i ][ j ] =
	      (char*)realloc( blockLinesArray[ i ][ j ], maxBoxByteWidth + 1 );
	    memset( blockLinesArray[ i ][ j ] + currentWidth, ' ', extraWidth );
	    blockLinesArray[ i ][ j ][ maxBoxByteWidth ] = '\0';
	  }
	}
      }

      // Combine blocks vertically
      u64 totalLines = 0;
      for( u64 i = 0; i < num_blocks; i++ ){
	totalLines += blockLineCounts[ i ];
      }
      char** combinedLines = (char**)mem( sizeof( char*) * totalLines, u8 );
      u64 index = 0;
      for( u64 i = 0; i < num_blocks; i++ ){
	for( u64 j = 0; j < blockLineCounts[ i ]; j++ ){
	  combinedLines[ index++ ] = custom_strdup( blockLinesArray[ i ][ j ] );
	}
      }

      // Join combined lines
      u64 totalLength = 0;
      for( u64 i = 0; i < totalLines; i++ ){
	totalLength += strlen( combinedLines[ i ] );
	if( i < totalLines - 1 ){
	  totalLength += 1; // for '\n'
	}
      }
      char* result = (char*)mem( totalLength + 1, u8 );
      result[ 0 ] = '\0';
      for( u64 i = 0; i < totalLines; i++ ){
	strcat( result, combinedLines[ i ] );
	if( i < totalLines - 1 ){
	  strcat( result, "\n" );
	}
	unmem( combinedLines[ i ] );
      }
      unmem( combinedLines );

      // Free memory
      for( u64 i = 0; i < num_blocks; i++ ){
	for( u64 j = 0; j < blockLineCounts[ i ]; j++ ){
	  unmem( blockLinesArray[ i ][ j ] );
	}
	unmem( blockLinesArray[ i ] );
      }
      unmem( blockLinesArray );
      unmem( blockLineCounts );

      for( u64 i = 0; i < num_blocks; i++ ){
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
  u64* shapeArg = t->shape;
  u32 data_length = t->size;
  u8* data = t->data;
  
  u64 shape_length = shapeArg_length + 1;
  u64* shape = (u64*)mem( shape_length, u64 );
  shape[ 0 ] = 1;
  for( u64 i = 0; i < shapeArg_length; i++ ){
    shape[ i + 1 ] = shapeArg[ i ];
  }
  // Compute maxNumLength
  u64 maxNumLength = 0;
  for( u64 i = 0; i < data_length; i++ ){
    u64 len = numLength( data[ i ] );
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
