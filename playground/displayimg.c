#define _CRT_SECURE_NO_WARNINGS 1 // for msvc
#include <stdio.h>
#include <stdlib.h>

#define HEADER_SIZE 16
#define IMAGE_ROW (28)
#define IMAGE_COL (28)
#define IMAGE_SIZE (1 * 28 * 28)

int main(int argc, char** argv) {
    int image_index = 0;
    if(argc >= 2) {
        image_index = strtol(argv[1], NULL, 10);
        if(errno != 0) {
            return -1;
        }
    }

    FILE* file = fopen("t10k-images.idx3-ubyte", "rb");
    unsigned char header[HEADER_SIZE];
    fread(header, HEADER_SIZE, 1, file);

    unsigned char img[IMAGE_SIZE];
    fseek(file, HEADER_SIZE + image_index * IMAGE_SIZE, SEEK_SET);
    fread(img, IMAGE_SIZE, 1, file);

    for(int i = 0; i < IMAGE_ROW; ++i) {
        for(int j = 0; j < IMAGE_COL; ++j) {
            printf("%3d ", (int)img[IMAGE_ROW * i + j]);
        }
        putchar('\n');
    }

    fclose(file);
}
