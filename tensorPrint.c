////////////////////////////////////////////////////////////////////////////////
// Copyright © 2025 Jon DuBois. Written with the assistance of GPT-4 et al.   //
////////////////////////////////////////////////////////////////////////////////

#include "Atlas.h"

// Function to compute the length of a number when converted to string
u32 numLength( f32 num ){
  char buffer[ 50 ];
  u32 length = snprintf( buffer, sizeof( buffer ), "%.2f", num );
  return length;
}

// Trim trailing spaces from a string
void rtrim( char* s ){
  int end = strlen( s ) - 1;
  while( end >= 0 && s[ end ] == ' ' ){
    s[ end-- ] = '\0';
  }
}

// Function to compute the product of elements in an array slice
u32 product( u32* arr, u32 start, u32 end ){
  u32 p = 1;
  for( u32 i = start; i < end; i++ ){
    p *= arr[ i ];
  }
  return p;
}

// Repeat a character 'c' for 'count' times and return the string
char* repeatChar( const char* c, u32 count ){
  // 'c' may be multi-byte, get its length in bytes
  u32 charLen = (u32)strlen( c );
  u32 totalLen = charLen * count;
  char* str = (char*)mem( totalLen + 1, u8 );
  for( u32 i = 0; i < count; i++ ){
    memcpy( str + i * charLen, c, charLen );
  }
  str[ totalLen ] = '\0';
  return str;
}

// Custom strdup
char* custom_strdup( const char* s ){
  size_t len = strlen( s );
  char* dup = (char*)mem( len + 1, u8 );
  if( dup ){
    strcpy( dup, s );
  }
  return dup;
}

// Split a string by '\n' and return an array of strings
char** splitString( char* str, u32* lineCount ){
  u32 capacity = 10;
  char** lines = (char**)mem( sizeof( char* ) * capacity, u8 );
  *lineCount = 0;
  char* line = strtok( str, "\n" );

  while( line != NULL ){
    if( *lineCount >= capacity ){
      capacity *= 2;
      lines = (char**)realloc( lines, sizeof( char* ) * capacity );
    }
    lines[ *lineCount ] = custom_strdup( line );
    ( *lineCount )++;
    line = strtok( NULL, "\n" );
  }
  return lines;
}

// Calculate the display width of a string (number of characters)
u32 displayWidth( const char* str ){
  u32 width = 0;
  const unsigned char* s = (const unsigned char*)str;
  while( *s ){
    if( ( *s & 0x80 ) == 0 ){
      // 1-byte character
      s += 1;
    } else if( ( *s & 0xE0 ) == 0xC0 ){
      // 2-byte character
      s += 2;
    } else if( ( *s & 0xF0 ) == 0xE0 ){
      // 3-byte character
      s += 3;
    } else if( ( *s & 0xF8 ) == 0xF0 ){
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

// Compute the maximum number length in the tensor
u32 computeMaxNumLength( const tensor* t ){
  u32 maxNumLength = 0;
  u32 indices[ 4 ] = { 0 };
  u32 rank = t->rank;
  const u32* shape = t->shape;
  const s32* strides = t->strides;
  u32 totalElements = t->size;

  for( u32 count = 0; count < totalElements; count++ ){
    s32 index = t->offset;
    for( u32 i = 0; i < rank; i++ ){
      index += indices[ i ] * strides[ i ];
    }

    f32 value = t->data[ index ];
    char buffer[ 50 ];
    int len = snprintf( buffer, sizeof( buffer ), "%.2f", value );
    if( len > (int)maxNumLength ){
      maxNumLength = len;
    }

    // Increment indices
    for( s32 i = rank - 1; i >= 0; i-- ){
      indices[ i ]++;
      if( indices[ i ] < shape[ i ] ){
        break;
      } else {
        indices[ i ] = 0;
      }
    }
  }
  return maxNumLength;
}

s32 translateIndex( u32 linearIndex, u32* shape, s32* strides, u32 rank ){
  u32 li = linearIndex;
  u32 tensorIndex[ 5 ] = { 0, 0, 0, 0, 0 };  // For up to rank 5
  s32 newIndex = 0;

  // Convert linear index to multi-dimensional indices
  for( int i = rank - 1; i >= 0; --i ){
    tensorIndex[ i ] = li % shape[ i ];
    li /= shape[ i ];
  }
  // Then compute the actual offset
  for( u32 i = 0; i < rank; ++i ){
    newIndex += tensorIndex[ i ] * strides[ i ];
  }
  return newIndex;
}

// Recursive helper
char* helper( u32 dimIndex,
              u32 offset,
              u32 depth,
              u32* shape,
              s32* strides,
              u32 shape_length,
              const f32* data,
              u32 maxNumLength,
              const tensor* t ){

  // Base case: last dimension
  if( dimIndex == shape_length - 1 ){
    u32 num_elements = shape[ dimIndex ];
    // Enough space for all numbers plus spaces in between
    u32 total_length = num_elements * maxNumLength + ( num_elements - 1 );
    char* result = (char*)mem( total_length + 1, u8 );
    char* ptr = result;

    for( u32 i = 0; i < num_elements; i++ ){
      char numStr[ 50 ];
      s32 translatedIndex =
        translateIndex( offset + i, shape + 1, strides + 1, shape_length - 1 );
      translatedIndex += t->offset;

      snprintf( numStr,
                sizeof( numStr ),
                "%-*.*f",
                (int)maxNumLength,
                2,
                data[ translatedIndex ] );

      u32 numStr_len = (u32)strlen( numStr );
      memcpy( ptr, numStr, numStr_len );
      ptr += numStr_len;

      if( i < num_elements - 1 ){
        *ptr = ' ';
        ptr++;
      }
    }
    *ptr = '\0';
    return result;
  } else {
    // Recursive case
    u32 isHorizontal = ( ( depth % 2 ) == 0 );
    u32 size = product( shape, dimIndex + 1, shape_length );
    u32 num_blocks = shape[ dimIndex ];

    // Build sub-blocks
    char** blocks = (char**)mem( sizeof( char* ) * num_blocks, u8 );
    for( u32 i = 0; i < num_blocks; i++ ){
      blocks[ i ] = helper( dimIndex + 1,
                            offset + i * size,
                            depth + 1,
                            shape,
                            strides,
                            shape_length,
                            data,
                            maxNumLength,
                            t );
    }

    if( isHorizontal ){
      // Stack horizontally with boxes
      u32 maxHeight = 0;
      char*** blockLinesArray =
        (char***)mem( sizeof( char** ) * num_blocks, u8 );
      u32* blockLineCounts = (u32*)mem( sizeof( u32 ) * num_blocks, u8 );

      // Convert each sub-block into lines, measure, boxify
      for( u32 i = 0; i < num_blocks; i++ ){
        // Split block into lines
        char* blockCopy = custom_strdup( blocks[ i ] );
        u32 lineCount = 0;
        char** lines = splitString( blockCopy, &lineCount );
        unmem( blockCopy );

        // Find max line length (in chars, not bytes)
        u32 maxLineDisplayWidth = 0;
        for( u32 j = 0; j < lineCount; j++ ){
          u32 len = displayWidth( lines[ j ] );
          if( len > maxLineDisplayWidth ){
            maxLineDisplayWidth = len;
          }
        }

        // Build ASCII box around those lines
        u32 maxLength = maxLineDisplayWidth;
        u32 boxDisplayWidth = maxLength + 4;  // +4 for +...+ with spaces
        u32 boxByteWidth = boxDisplayWidth + 1;
        char* top = (char*)mem( boxByteWidth, u8 );
        char* bottom = (char*)mem( boxByteWidth, u8 );
        char* lineFill = repeatChar( "-", maxLength + 2 );

        snprintf( top, boxByteWidth, "+%s+", lineFill );
        snprintf( bottom, boxByteWidth, "+%s+", lineFill );
        unmem( lineFill );

        // Now build the block lines with vertical bars
        char** blockLines =
          (char**)mem( sizeof( char* ) * ( lineCount + 2 ), u8 );
        blockLines[ 0 ] = top;

        for( u32 j = 0; j < lineCount; j++ ){
          u32 padding = maxLength - displayWidth( lines[ j ] );
          u32 lineBytes = boxDisplayWidth + 1;
          char* middleLine = (char*)mem( lineBytes, u8 );
          snprintf(
            middleLine, lineBytes, "| %s%*s |", lines[ j ], (int)padding, "" );
          // Trim trailing spaces
          rtrim( middleLine );
          blockLines[ j + 1 ] = middleLine;

          unmem( lines[ j ] );
        }
        unmem( lines );

        blockLines[ lineCount + 1 ] = bottom;
        // Also trim top/bottom in case we want to remove trailing spaces
        rtrim( top );
        rtrim( bottom );

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
          // Create an empty line of that width
          char* emptyLine = (char*)mem( width + 1, u8 );
          memset( emptyLine, ' ', width );
          emptyLine[ width ] = '\0';

          blockLinesArray[ i ] = (char**)realloc( blockLinesArray[ i ],
                                                  sizeof( char* ) * maxHeight );
          for( u32 j = currentHeight; j < maxHeight; j++ ){
            blockLinesArray[ i ][ j ] = custom_strdup( emptyLine );
          }
          blockLineCounts[ i ] = maxHeight;
          unmem( emptyLine );
        }
      }

      // Combine blocks line by line
      char** combinedLines = (char**)mem( sizeof( char* ) * maxHeight, u8 );

      for( u32 lineIdx = 0; lineIdx < maxHeight; lineIdx++ ){
        // Calculate total length needed
        u32 totalLineLength = 0;
        for( u32 b = 0; b < num_blocks; b++ ){
          totalLineLength += (u32)strlen( blockLinesArray[ b ][ lineIdx ] );
          if( b < num_blocks - 1 ){
            totalLineLength += 1;  // space between blocks
          }
        }
        char* combined = (char*)mem( totalLineLength + 1, u8 );
        combined[ 0 ] = '\0';

        // Stitch them together
        for( u32 b = 0; b < num_blocks; b++ ){
          strcat( combined, blockLinesArray[ b ][ lineIdx ] );
          if( b < num_blocks - 1 ){
            strcat( combined, " " );
          }
        }
        rtrim( combined );  // trim trailing spaces after combining
        combinedLines[ lineIdx ] = combined;
      }

      // Join combined lines with '\n'
      u32 totalLength = 0;
      for( u32 i = 0; i < maxHeight; i++ ){
        totalLength += (u32)strlen( combinedLines[ i ] );
        if( i < maxHeight - 1 ){
          totalLength += 1;  // newline
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

      // Clean up
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
        (char***)mem( sizeof( char** ) * num_blocks, u8 );
      u32* blockLineCounts = (u32*)mem( sizeof( u32 ) * num_blocks, u8 );

      for( u32 i = 0; i < num_blocks; i++ ){
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
        u32 maxLength = maxLineDisplayWidth;
        if( maxLength > maxWidth ){
          maxWidth = maxLength;
        }

        // Build ASCII box
        u32 boxDisplayWidth = maxLength + 4;  // +4 for +...+
        // We no longer do *3, just +1 for the null terminator
        u32 boxByteWidth = boxDisplayWidth + 1;

        char* top = (char*)mem( boxByteWidth, u8 );
        char* bottom = (char*)mem( boxByteWidth, u8 );
        char* lineFill = repeatChar( "-", maxLength + 2 );

        snprintf( top, boxByteWidth, "+%s+", lineFill );
        snprintf( bottom, boxByteWidth, "+%s+", lineFill );
        unmem( lineFill );

        char** blockLines =
          (char**)mem( sizeof( char* ) * ( lineCount + 2 ), u8 );
        blockLines[ 0 ] = top;

        for( u32 j = 0; j < lineCount; j++ ){
          u32 padding = maxLength - displayWidth( lines[ j ] );
          u32 lineSize = boxDisplayWidth + 1;
          char* middleLine = (char*)mem( lineSize, u8 );
          snprintf(
            middleLine, lineSize, "| %s%*s |", lines[ j ], (int)padding, "" );
          rtrim( middleLine );
          blockLines[ j + 1 ] = middleLine;
          unmem( lines[ j ] );
        }
        unmem( lines );
        blockLines[ lineCount + 1 ] = bottom;
        rtrim( top );
        rtrim( bottom );

        blockLinesArray[ i ] = blockLines;
        blockLineCounts[ i ] = lineCount + 2;
      }

      // Pad blocks to have the same width
      // We'll compute the largest "real" width among the boxes
      u32 actualMaxWidth = 0;
      for( u32 i = 0; i < num_blocks; i++ ){
        u32 w = strlen( blockLinesArray[ i ][ 0 ] );
        if( w > actualMaxWidth ){
          actualMaxWidth = w;
        }
      }

      for( u32 i = 0; i < num_blocks; i++ ){
        // For each line, if it's shorter, pad with spaces
        for( u32 j = 0; j < blockLineCounts[ i ]; j++ ){
          u32 currLen = strlen( blockLinesArray[ i ][ j ] );
          if( currLen < actualMaxWidth ){
            u32 diff = actualMaxWidth - currLen;
            blockLinesArray[ i ][ j ] =
              (char*)realloc( blockLinesArray[ i ][ j ], actualMaxWidth + 1 );
            memset( blockLinesArray[ i ][ j ] + currLen, ' ', diff );
            blockLinesArray[ i ][ j ][ actualMaxWidth ] = '\0';
            // Optionally rtrim if you want to remove trailing spaces again
          }
        }
      }

      // Combine blocks vertically
      u32 totalLines = 0;
      for( u32 i = 0; i < num_blocks; i++ ){
        totalLines += blockLineCounts[ i ];
      }
      char** combinedLines = (char**)mem( sizeof( char* ) * totalLines, u8 );
      u32 index = 0;
      for( u32 i = 0; i < num_blocks; i++ ){
        for( u32 j = 0; j < blockLineCounts[ i ]; j++ ){
          // copy line
          combinedLines[ index++ ] = custom_strdup( blockLinesArray[ i ][ j ] );
        }
      }

      // Join combined lines
      u32 totalLength = 0;
      for( u32 i = 0; i < totalLines; i++ ){
        totalLength += (u32)strlen( combinedLines[ i ] );
        if( i < totalLines - 1 ){
          totalLength += 1;  // for '\n'
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
char* formatTensorData( tensor* t ){
  /* if( t->gpu && t->tex.channels != 0 ){ */
  /*   char* rets = "[texture]"; */
  /*   char* ret = mem( strlen( rets ) + 1, char ); */
  /*   strcpy( ret, rets ); */
  /*   return ret; */
  /* } */
  tensorToHostMemory( t );
  u32 shape_length = t->rank + 1;
  u32* shape = mem( shape_length, u32 );
  s32* strides = mem( shape_length, s32 );

  // "Extra" dimension trick used by the original code
  shape[ 0 ] = 1;
  strides[ 0 ] = t->strides[ 0 ] * t->shape[ 0 ];
  for( u32 i = 0; i < t->rank; ++i ){
    shape[ i + 1 ] = t->shape[ i ];
    strides[ i + 1 ] = t->strides[ i ];
  }

  // Compute maxNumLength
  u32 maxNumLength = computeMaxNumLength( t );

  // Build the ASCII representation
  char* result =
    helper( 0, 0, 0, shape, strides, shape_length, t->data, maxNumLength, t );

  unmem( shape );
  unmem( strides );

  return result;  // Caller must free
}
