#pragma once

#include <stdint.h>

#define GROUP 3
#define NO_FLAGS 0
#define NO_CALLBACK NULL

#define GET_SEGMENTID(id) (GROUP << 16 | id)

#define SEGMENT_CLIENT GET_SEGMENTID(1)
#define SEGMENT_SERVER GET_SEGMENTID(2)

#define PACKET_SIZE 64
#define DATA_SIZE 512

enum cmd
{
    CMD_NULL,
    CMD_QUIT,
    CMD_DONE
};

struct packet
{
    uint32_t cmd;
};

struct server_segment
{
    struct packet packet __attribute__((aligned(64)));
};

struct client_segment
{
    struct packet packet __attribute__((aligned(64)));
};
