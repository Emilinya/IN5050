#include <stdlib.h>

#include "utils.h"

// simple funcition to read image with error handeling
FILE *errcheck_fopen(const char *filename, const char *mode) {
    FILE *file = fopen(filename, mode);
    if (file == NULL)
    {
        char buff[50];
        snprintf(buff, 50, "can't open %s:", filename);
        perror(buff);
        exit(EXIT_FAILURE);
    }
    return file;
}

void dump_image(yuv_t *image, int w, int h, FILE *fp)
{
    fwrite(image->Y, 1, w * h, fp);
    fwrite(image->U, 1, w * h / 4, fp);
    fwrite(image->V, 1, w * h / 4, fp);
}
