#pragma once

#include <stdint.h>

#include "c63.h"

#define GROUP 3
#define NO_FLAGS 0
#define NO_CALLBACK NULL

#define GET_SEGMENTID(id) (GROUP << 16 | id)

#define SEGMENT_CLIENT GET_SEGMENTID(1)
#define SEGMENT_SERVER GET_SEGMENTID(2)

#define LOCAL_ADAPTER_NUMBER 0
#define CLIENT_INTERRUPT_NUMBER GET_SEGMENTID(3)
#define SERVER_INTERRUPT_NUMBER GET_SEGMENTID(4)

enum cmd
{
    CMD_CONTINUE,
    CMD_QUIT
};

struct server_segment
{
    volatile uint8_t *cmd;
    yuv_t *reference_recons;
    yuv_t *currenct_recons;
    yuv_t *predicted;
    dct_t *residuals;
    struct macroblock *mbs[COLOR_COMPONENTS];
};

struct client_segment
{
    volatile uint8_t *cmd;
    yuv_t *image;
};
