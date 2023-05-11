#pragma once

#include <stdio.h>

#include "c63.h"

#define TRUE 1
#define FALSE 0

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

#define MEMCPY_YUV(dest_yuv, src_yuv, ysize, usize, vsize) \
    do { \
        memcpy(dest_yuv->Y, src_yuv->Y, ysize * sizeof(uint8_t)); \
        memcpy(dest_yuv->U, src_yuv->U, usize * sizeof(uint8_t)); \
        memcpy(dest_yuv->V, src_yuv->V, vsize * sizeof(uint8_t)); \
    } while (0)

#define MEMCPY_DCT(dest_yuv, src_yuv, ysize, usize, vsize) \
    do { \
        memcpy(dest_yuv->Ydct, src_yuv->Ydct, ysize * sizeof(int16_t)); \
        memcpy(dest_yuv->Udct, src_yuv->Udct, usize * sizeof(int16_t)); \
        memcpy(dest_yuv->Vdct, src_yuv->Vdct, vsize * sizeof(int16_t)); \
    } while (0)

#define MEMCPY_MBS(dest_mbs, src_mbs, mb_count) \
    do { \
        memcpy(dest_mbs[Y_COMPONENT], src_mbs[Y_COMPONENT], mb_count * sizeof(struct macroblock)); \
        memcpy(dest_mbs[U_COMPONENT], src_mbs[U_COMPONENT], mb_count / 4 * sizeof(struct macroblock)); \
        memcpy(dest_mbs[V_COMPONENT], src_mbs[V_COMPONENT], mb_count / 4 * sizeof(struct macroblock)); \
    } while (0)

FILE *errcheck_fopen(const char *filename, const char *mode);

struct c63_common *init_c63_enc(int width, int height);

yuv_t *read_yuv(FILE *file, struct c63_common *cm);
yuv_t *create_yuv(struct c63_common *cm);
void free_yuv(yuv_t *image);

struct frame *create_frame(struct c63_common *cm);
void destroy_frame(struct frame *f);

void dump_image(yuv_t *image, int w, int h, FILE *fp);
