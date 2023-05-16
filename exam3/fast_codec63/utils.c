#include <stdlib.h>

#include "utils.h"
#include "tables.h"

struct c63_common *init_c63_enc(int width, int height)
{
    struct c63_common *cm = calloc(1, sizeof(struct c63_common));

    cm->width = width;
    cm->height = height;

    cm->padw[Y_COMPONENT] = cm->ypw = (uint32_t)(ceil(width / 16.0f) * 16);
    cm->padh[Y_COMPONENT] = cm->yph = (uint32_t)(ceil(height / 16.0f) * 16);
    cm->padw[U_COMPONENT] = cm->upw = (uint32_t)(ceil(width * UX / (YX * 8.0f)) * 8);
    cm->padh[U_COMPONENT] = cm->uph = (uint32_t)(ceil(height * UY / (YY * 8.0f)) * 8);
    cm->padw[V_COMPONENT] = cm->vpw = (uint32_t)(ceil(width * VX / (YX * 8.0f)) * 8);
    cm->padh[V_COMPONENT] = cm->vph = (uint32_t)(ceil(height * VY / (YY * 8.0f)) * 8);

    cm->mb_cols = cm->ypw / 8;
    cm->mb_rows = cm->yph / 8;

    /* Quality parameters -- Home exam deliveries should have original values,
     i.e., quantization factor should be 25, search range should be 16, and the
     keyframe interval should be 100. */
    cm->qp = 25;                 // Constant quantization factor. Range: [1..50]
    cm->me_search_range = 16;    // Pixels in every direction
    cm->keyframe_interval = 100; // Distance between keyframes

    /* Initialize quantization tables */
    for (int i = 0; i < 64; ++i)
    {
        cm->quanttbl[Y_COMPONENT][i] = yquanttbl_def[i] / (cm->qp / 10.0);
        cm->quanttbl[U_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
        cm->quanttbl[V_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
    }

    return cm;
}

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
int read_yuv(FILE *file, struct c63_common *cm, yuv_t *image)
{
    size_t len = 0;

    /* Read Y. The size of Y is the same as the size of the image. The indices
       represents the color component (0 is Y, 1 is U, and 2 is V) */
    len += fread(image->Y, 1, cm->width * cm->height, file);

    /* Read U. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y
       because (height/2)*(width/2) = (height*width)/4. */
    len += fread(image->U, 1, (cm->width * cm->height) / 4, file);

    /* Read V. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y. */
    len += fread(image->V, 1, (cm->width * cm->height) / 4, file);

    if (ferror(file))
    {
        perror("ferror");
        exit(EXIT_FAILURE);
    }

    if (feof(file))
    {
        return FALSE;
    }
    else if (len != cm->width * cm->height * 1.5)
    {
        fprintf(stderr, "Reached end of file, but incorrect bytes read.\n");
        fprintf(stderr, "Wrong input? (height: %d width: %d)\n", cm->height, cm->width);

        return FALSE;
    }

    return TRUE;
}

yuv_t *create_yuv(struct c63_common *cm)
{
    yuv_t *yuv = malloc(sizeof(yuv_t));
    yuv->Y = calloc(cm->ypw * cm->yph, sizeof(uint8_t));
    yuv->U = calloc(cm->upw * cm->uph, sizeof(uint8_t));
    yuv->V = calloc(cm->vpw * cm->vph, sizeof(uint8_t));

    return yuv;
}

void free_yuv(yuv_t *image)
{
    free(image->Y);
    free(image->U);
    free(image->V);
    free(image);
}

struct frame *create_frame(struct c63_common *cm)
{
    struct frame *f = malloc(sizeof(struct frame));

    f->recons = create_yuv(cm);
    f->predicted = create_yuv(cm);

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

    free_yuv(f->recons);
    free_yuv(f->residuals);
    free_yuv(f->predicted);

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
