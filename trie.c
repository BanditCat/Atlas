////////////////////////////////////////////////////////////////////////////////
// Copyright © 2025 Jon DuBois. Written with the assistance of GPT-4 et al.   //
////////////////////////////////////////////////////////////////////////////////

#include "Atlas.h"

char* stringDup( const char* src ){
  if( src == NULL )
    return NULL;
  size_t len = strlen( src ) + 1;
  char* dest = (char*)mem( len, char );
  memcpy( dest, src, len );
  return dest;
}

// Helper function to find the common prefix length between two strings
u32 commonPrefix( const char* s1, const char* s2 ){
  u32 i = 0;
  while( s1[ i ] && s2[ i ] && s1[ i ] == s2[ i ] )
    i++;
  return i;
}

trieNode* newTrieNode( const char* part, u32 value ){
  trieNode* node = (trieNode*)mem( 1, trieNode );
  if( part )
    node->thisPart = stringDup( part );
  else
    node->thisPart = NULL;

  // No need to zero pointers because mem calls calloc.
  node->value = value;
  return node;
}

void deleteTrieNode( trieNode* node ){
  if( !node )
    return;
  for( int i = 0; i < 256; ++i )
    if( node->nextParts[ i ] )
      deleteTrieNode( node->nextParts[ i ] );

  if( node->thisPart )
    unmem( node->thisPart );
  unmem( node );
}

void trieInsert( trieNode* root, const char* key, u32 value ){
  if( !root || !key )
    error( "%s", "Root node or key is NULL." );

  trieNode* current = root;
  const char* remainingKey = key;

  while( *remainingKey ){
    unsigned char currentChar = (unsigned char)*remainingKey;
    trieNode* child = current->nextParts[ currentChar ];

    if( !child ){
      trieNode* newNode = newTrieNode( remainingKey, value );
      current->nextParts[ currentChar ] = newNode;
      return;
    }

    size_t prefixLen = commonPrefix( child->thisPart, remainingKey );

    if( prefixLen == strlen( child->thisPart ) ){
      current = child;
      remainingKey += prefixLen;
    } else {
      trieNode* splitNode =
        newTrieNode( child->thisPart, (u32)-1 );  // New intermediate node
      splitNode->thisPart[ prefixLen ] = '\0';

      // Adjust the existing child
      char* childSuffix = stringDup( child->thisPart + prefixLen );
      unmem( child->thisPart );
      child->thisPart = childSuffix;

      // Move the existing child under the split node
      unsigned char suffixFirstChar = (unsigned char)*child->thisPart;
      splitNode->nextParts[ suffixFirstChar ] = child;

      // Attach the split node to the current node
      current->nextParts[ currentChar ] = splitNode;

      // Now, add the new node for the remaining key
      const char* newSuffix = remainingKey + prefixLen;
      if( *newSuffix ){
        trieNode* newChild = newTrieNode( newSuffix, value );
        splitNode->nextParts[ (unsigned char)*newSuffix ] = newChild;
      } else
        splitNode->value = value;

      return;
    }
  }
  // The key exactly matches a node's thisPart
  current->value = value;
}

// Search for a key in the trie and retrieve its value
bool trieSearch( trieNode* root, const char* key, u32* value ){
  if( !root || !key )
    return false;

  trieNode* current = root;
  const char* remainingKey = key;

  while( *remainingKey ){
    unsigned char currentChar = (unsigned char)*remainingKey;
    trieNode* child = current->nextParts[ currentChar ];

    if( !child )
      return false;

    u32 partLen = strlen( child->thisPart );
    u32 prefixLen = commonPrefix( child->thisPart, remainingKey );

    if( prefixLen == partLen ){
      current = child;
      remainingKey += prefixLen;
    } else
      return false;
  }

  if( current->value != (u32)-1 ){
    if( value )
      *value = current->value;
    return true;
  }

  return false;
}
