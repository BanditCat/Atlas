////////////////////////////////////////////////////////////////////////////////
// Copyright © 2024 Jon DuBois. Written with the assistance of GPT-4 et al.   //
////////////////////////////////////////////////////////////////////////////////


#ifndef TRIE_H_INCLUDED
#define TRIE_H_INCLUDED

typedef struct trieNode{
  char* thisPart;
  struct trieNode* nextParts[ 256 ];
  u32 value;
} trieNode;


trieNode* newTrieNode( const char* part, u32 value );
void deleteTrieNode( trieNode* node );
void trieInsert( trieNode* root, const char* key, u32 value );
bool trieSearch( trieNode* root, const char* key, u32* value );

#endif //TRIE_H_INCLUDED


