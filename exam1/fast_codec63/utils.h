#pragma once

#include <stdio.h>

#include "c63.h"

FILE *errcheck_fopen(const char *filename, const char *mode);

yuv_t *read_yuv(FILE *file, struct c63_common *cm);
void free_yuv(yuv_t *image);
