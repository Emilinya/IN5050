#include <cuda.h>

#include "utils.h"
#include "tables.h"

// Read planar YUV frames with 4:2:0 chroma sub-sampling */
__host__ yuv_t *read_yuv(FILE *file, struct c63_common *cm)
{
    size_t len = 0;
    yuv_t *image;
    cudaMallocErr(image, sizeof(yuv_t));

    /* Read Y. The size of Y is the same as the size of the image. The indices
       represents the color component (0 is Y, 1 is U, and 2 is V) */
    cudaCallocErr(image->Y, 1, cm->ypw * cm->yph);
    len += fread(image->Y, 1, cm->width * cm->height, file);

    /* Read U. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y
       because (height/2)*(width/2) = (height*width)/4. */
    cudaCallocErr(image->U, 1, cm->upw * cm->uph);
    len += fread(image->U, 1, (cm->width * cm->height) / 4, file);

    /* Read V. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y. */
    cudaCallocErr(image->V, 1, cm->vpw * cm->vph);
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

__host__ yuv_t *create_yuv(struct c63_common *cm)
{
    yuv_t *yuv;
    
    cudaMallocErr(yuv, sizeof(yuv_t));
    cudaCallocErr(yuv->Y, cm->ypw * cm->yph, sizeof(uint8_t));
    cudaCallocErr(yuv->U, cm->upw * cm->uph, sizeof(uint8_t));
    cudaCallocErr(yuv->V, cm->vpw * cm->vph, sizeof(uint8_t));

    return yuv;
}

__host__ void free_yuv(yuv_t *image)
{
    cudaFree(image->Y);
    cudaFree(image->U);
    cudaFree(image->V);
    cudaFree(image);
}

__host__ struct frame *create_frame(struct c63_common *cm)
{
    struct frame *f;
    cudaMallocErr(f, sizeof(struct frame));
    
    f->recons = create_yuv(cm);
    f->predicted = create_yuv(cm);

    cudaMallocErr(f->residuals, sizeof(dct_t));
    cudaCallocErr(f->residuals->Ydct, cm->ypw * cm->yph, sizeof(int16_t));
    cudaCallocErr(f->residuals->Udct, cm->upw * cm->uph, sizeof(int16_t));
    cudaCallocErr(f->residuals->Vdct, cm->vpw * cm->vph, sizeof(int16_t));

    cudaCallocErr(
        f->mbs[Y_COMPONENT],
        cm->mb_rows * cm->mb_cols, sizeof(struct macroblock));
    cudaCallocErr(
        f->mbs[U_COMPONENT],
        cm->mb_rows / 2 * cm->mb_cols / 2, sizeof(struct macroblock));
    cudaCallocErr(
        f->mbs[V_COMPONENT],
        cm->mb_rows / 2 * cm->mb_cols / 2, sizeof(struct macroblock));

    return f;
}

__host__ void free_frame(struct frame *f)
{
    /* First frame doesn't have a reconstructed frame to destroy */
    if (!f)
    {
        return;
    }

    free_yuv(f->recons);
    free_yuv(f->predicted);

    cudaFree(f->residuals->Ydct);
    cudaFree(f->residuals->Udct);
    cudaFree(f->residuals->Vdct);
    cudaFree(f->residuals);

    cudaFree(f->mbs[Y_COMPONENT]);
    cudaFree(f->mbs[U_COMPONENT]);
    cudaFree(f->mbs[V_COMPONENT]);

    cudaFree(f);
}

__host__ void init_tables(struct c63_common *cm) {
    // initialize quantization tables
    cudaMallocErr(cm->quanttbl[Y_COMPONENT], 64);
    cudaMallocErr(cm->quanttbl[U_COMPONENT], 64);
    cudaMallocErr(cm->quanttbl[V_COMPONENT], 64);
    for (int i = 0; i < 64; ++i)
    {
        cm->quanttbl[Y_COMPONENT][i] = yquanttbl_def[i] / (cm->qp / 10.0);
        cm->quanttbl[U_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
        cm->quanttbl[V_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
    }

    // initialize dct lookup
    cudaMallocErr(cm->dctlookup, 64 * sizeof(float));
    for (int i = 0; i < 64; ++i)
    {
        cm->dctlookup[i] = dct_lookup_table[i / 8][i % 8];
    }

    // initialize zigzag tables
    cudaMallocErr(cm->zigzag_U, 64);
    cudaMallocErr(cm->zigzag_V, 64);
    for (int i = 0; i < 64; ++i)
    {
        cm->zigzag_U[i] = zigzag_U_table[i];
        cm->zigzag_V[i] = zigzag_V_table[i];
    }
}

__host__ void free_tables(struct c63_common *cm) {
    cudaFree(cm->quanttbl[Y_COMPONENT]);
    cudaFree(cm->quanttbl[U_COMPONENT]);
    cudaFree(cm->quanttbl[V_COMPONENT]);
    cudaFree(cm->dctlookup);
    cudaFree(cm->zigzag_U);
    cudaFree(cm->zigzag_V);
}
