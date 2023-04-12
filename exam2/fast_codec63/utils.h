#pragma once

#include <stdio.h>

#include "c63.h"

// cudaMalloc with error checking
#define cudaMallocErr(ptr, size) \
    do { \
        cudaError_t __cudaCalloc_err = cudaMallocManaged((void **)&ptr, size); \
        if (__cudaCalloc_err != cudaSuccess) fprintf(stderr, "Error when mallocing!"); \
    } while (0)

// there is no cudaCalloc, so I define one myself
#define cudaCallocErr(ptr, nmemb, size) \
    do { \
        cudaError_t __cudaCalloc_err = cudaMallocManaged((void **)&ptr, nmemb*size); \
        if (__cudaCalloc_err == cudaSuccess) { \
            cudaMemset((void *)ptr, 0, nmemb*size); \
         } else { \
            fprintf(stderr, "Error when callocing!"); \
         } \
    } while (0)

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

FILE *errcheck_fopen(const char *filename, const char *mode);

void dump_image(yuv_t *image, int w, int h, FILE *fp);

#ifdef __cplusplus
extern "C"
{
#endif
    yuv_t *read_yuv(FILE *file, struct c63_common *cm);
    yuv_t *create_yuv(struct c63_common *cm);
    void free_yuv(yuv_t *image);

    struct frame *create_frame(struct c63_common *cm);
    void free_frame(struct frame *f);

    void init_tables(struct c63_common *cm);
    void free_tables(struct c63_common *cm);
#ifdef __cplusplus
}
#endif
