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

// Read planar YUV frames with 4:2:0 chroma sub-sampling */
yuv_t *read_yuv(FILE *file, struct c63_common *cm)
{
    size_t len = 0;
    yuv_t *image = malloc(sizeof(*image));

    /* Read Y. The size of Y is the same as the size of the image. The indices
       represents the color component (0 is Y, 1 is U, and 2 is V) */
    image->Y = calloc(1, cm->padw[Y_COMPONENT] * cm->padh[Y_COMPONENT]);
    len += fread(image->Y, 1, cm->width * cm->height, file);

    /* Read U. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y
       because (height/2)*(width/2) = (height*width)/4. */
    image->U = calloc(1, cm->padw[U_COMPONENT] * cm->padh[U_COMPONENT]);
    len += fread(image->U, 1, (cm->width * cm->height) / 4, file);

    /* Read V. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y. */
    image->V = calloc(1, cm->padw[V_COMPONENT] * cm->padh[V_COMPONENT]);
    len += fread(image->V, 1, (cm->width * cm->height) / 4, file);

    if (ferror(file))
    {
        perror("ferror");
        exit(EXIT_FAILURE);
    }

    if (feof(file))
    {
        free_yuv(image);
        return NULL;
    }
    else if (len != cm->width * cm->height * 1.5)
    {
        fprintf(stderr, "Reached end of file, but incorrect bytes read.\n");
        fprintf(stderr, "Wrong input? (height: %d width: %d)\n", cm->height, cm->width);

        free_yuv(image);
        return NULL;
    }

    return image;
}

void free_yuv(yuv_t *image)
{
    free(image->Y);
    free(image->U);
    free(image->V);
    free(image);
}
