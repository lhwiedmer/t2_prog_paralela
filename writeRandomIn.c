#include <stdio.h>
#include <stdlib.h>
#include <time.h>


int main(int argv, char** argc) {
    const char* fileName = argc[1];
    const char* inSizeC = argc[2];
    int inSize = atoi(inSizeC);
    srand(time(NULL));
    FILE *file = fopen(fileName, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file %s\n", fileName);
        return 1;
    }
    for (size_t i = 0; i < inSize; i++) {
        char random_char = 'a' + rand() % 26;
        fputc(random_char, file);
    }
    fclose(file);
}