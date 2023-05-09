#pragma once

#include <stdint.h>

#include "c63.h"

#define GROUP 3
#define NO_FLAGS 0
#define NO_CALLBACK NULL

#define GET_SEGMENTID(id) (GROUP << 16 | id)

#define SEGMENT_CLIENT GET_SEGMENTID(1)
#define SEGMENT_SERVER GET_SEGMENTID(2)

enum cmd
{
    CMD_WAIT,
    CMD_QUIT,
    CMD_DONE
};

struct server_segment
{
    uint32_t cmd;
    yuv_t *reference_recons;
    yuv_t *currenct_recons;
    yuv_t *predicted;
    dct_t *residuals;
    struct macroblock *mbs[COLOR_COMPONENTS];
};

struct client_segment
{
    uint32_t cmd;
    yuv_t *image;
};
