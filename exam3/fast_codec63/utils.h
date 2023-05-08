#pragma once

#include <stdio.h>

#include "c63.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

FILE *errcheck_fopen(const char *filename, const char *mode);

yuv_t *read_yuv(FILE *file, struct c63_common *cm);
yuv_t *create_yuv(struct c63_common *cm);
void free_yuv(yuv_t *image);

struct frame *create_frame(struct c63_common *cm);
void destroy_frame(struct frame *f);

void dump_image(yuv_t *image, int w, int h, FILE *fp);
