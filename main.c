#include "Atlas.h"


// Helper function to calculate the maximum number length
int maxNumLength( int *data, int dataSize ){
    int maxLength = 0;
    for( int i = 0; i < dataSize; i++ ){
        int length = snprintf( NULL, 0, "%d", data[i] );
        if( length > maxLength ){
	  maxLength = length;
        }
    }
    return maxLength;
}

// Recursive helper function
void helper(int *data, int *shape, int dimIndex, int offset, int depth, int maxNumLength, char *result) {
  if (dimIndex == shape[0] - 1) {
        // Base case: last dimension
        for (int i = 0; i < shape[dimIndex]; i++) {
            char numStr[maxNumLength + 1];
            snprintf(numStr, sizeof(numStr), "%-*d", maxNumLength, data[offset + i]);
            strcat(result, numStr);
            if (i < shape[dimIndex] - 1) {
                strcat(result, " ");
            }
        }
    } else {
        // Recursive case
        int size = 1;
        for (int i = dimIndex + 1; i < shape[0]; i++) {
            size *= shape[i];
        }
        for (int i = 0; i < shape[dimIndex]; i++) {
            helper(data, shape, dimIndex + 1, offset + i * size, depth + 1, maxNumLength, result);
            if (i < shape[dimIndex] - 1) {
                strcat(result, "\n");
            }
        }
    }
}

// Main function to format tensor data
void formatTensorData(int *data, int dataSize, int *shape, int shapeSize, char *result) {
    int maxNumLengthValue = maxNumLength(data, dataSize);
    helper(data, shape, 1, 0, 0, maxNumLengthValue, result);
}

int main() {
    int data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int shape[] = {2, 2, 2};
    char result[1024] = {0};

    formatTensorData(data, 8, shape, 3, result);
    printf("%s\n", result);

    return 0;
}
  

  

int main( int argc, char** argv ){
  printf( " Hello %s world!\n", *argv );
}
