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

struct frame *create_frame(struct c63_common *cm, yuv_t *image)
{
    struct frame *f = malloc(sizeof(struct frame));

    f->orig = image;

    f->recons = malloc(sizeof(yuv_t));
    f->recons->Y = malloc(cm->ypw * cm->yph);
    f->recons->U = malloc(cm->upw * cm->uph);
    f->recons->V = malloc(cm->vpw * cm->vph);

    f->predicted = malloc(sizeof(yuv_t));
    f->predicted->Y = calloc(cm->ypw * cm->yph, sizeof(uint8_t));
    f->predicted->U = calloc(cm->upw * cm->uph, sizeof(uint8_t));
    f->predicted->V = calloc(cm->vpw * cm->vph, sizeof(uint8_t));

    f->residuals = malloc(sizeof(dct_t));
    f->residuals->Ydct = calloc(cm->ypw * cm->yph, sizeof(int16_t));
    f->residuals->Udct = calloc(cm->upw * cm->uph, sizeof(int16_t));
    f->residuals->Vdct = calloc(cm->vpw * cm->vph, sizeof(int16_t));

    f->mbs[Y_COMPONENT] =
        calloc(cm->mb_rows * cm->mb_cols, sizeof(struct macroblock));
    f->mbs[U_COMPONENT] =
        calloc(cm->mb_rows / 2 * cm->mb_cols / 2, sizeof(struct macroblock));
    f->mbs[V_COMPONENT] =
        calloc(cm->mb_rows / 2 * cm->mb_cols / 2, sizeof(struct macroblock));

    return f;
}

void destroy_frame(struct frame *f)
{
    /* First frame doesn't have a reconstructed frame to destroy */
    if (!f)
    {
        return;
    }

    free(f->recons->Y);
    free(f->recons->U);
    free(f->recons->V);
    free(f->recons);

    free(f->residuals->Ydct);
    free(f->residuals->Udct);
    free(f->residuals->Vdct);
    free(f->residuals);

    free(f->predicted->Y);
    free(f->predicted->U);
    free(f->predicted->V);
    free(f->predicted);

    free(f->mbs[Y_COMPONENT]);
    free(f->mbs[U_COMPONENT]);
    free(f->mbs[V_COMPONENT]);

    free(f);
}

void dump_image(yuv_t *image, int w, int h, FILE *fp)
{
    fwrite(image->Y, 1, w * h, fp);
    fwrite(image->U, 1, w * h / 4, fp);
    fwrite(image->V, 1, w * h / 4, fp);
}
