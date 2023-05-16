#pragma once

#include <stdio.h>
#include <stdlib.h>

#include <sisci_error.h>
#include <sisci_api.h>

#include "sisci_common.h"

// #define MEMCPY_YUV(dest_yuv, src_yuv, ysize, usize, vsize) \
//     do { \
//         SCIMemCpy(sequence, localMapAddr, remoteMap, remoteOffset, segmentSize, memcpyFlag, &error); \
//         SCIMemCpy(sequence, localMapAddr, remoteMap, remoteOffset, segmentSize, memcpyFlag, &error); \
//         SCIMemCpy(sequence, localMapAddr, remoteMap, remoteOffset, segmentSize, memcpyFlag, &error); \
//     } while (0)

// #define MEMCPY_DCT(dest_yuv, src_yuv, ysize, usize, vsize) \
//     do { \
//         SCIMemCpy(dest_yuv->Ydct, src_yuv->Ydct, ysize * sizeof(int16_t)); \
//         SCIMemCpy(dest_yuv->Udct, src_yuv->Udct, usize * sizeof(int16_t)); \
//         SCIMemCpy(dest_yuv->Vdct, src_yuv->Vdct, vsize * sizeof(int16_t)); \
//     } while (0)

// #define MEMCPY_MBS(dest_mbs, src_mbs, mb_count) \
//     do { \
//         SCIMemCpy(dest_mbs[Y_COMPONENT], src_mbs[Y_COMPONENT], mb_count * sizeof(struct macroblock)); \
//         SCIMemCpy(dest_mbs[U_COMPONENT], src_mbs[U_COMPONENT], mb_count / 4 * sizeof(struct macroblock)); \
//         SCIMemCpy(dest_mbs[V_COMPONENT], src_mbs[V_COMPONENT], mb_count / 4 * sizeof(struct macroblock)); \
//     } while (0)

#define TRIGGER_DATA_INTERRUPT(interrupt, segment, command, error) \
    do { \
        *segment->cmd = command; \
        SCIFlush(NULL, NO_FLAGS); \
        SCITriggerInterrupt(interrupt, NO_FLAGS, &error); \
        ERR_CHECK(error, "SCITriggerInterrupt"); \
    } while (0)

#define ERR_CHECK(error, name) \
    do { \
        if (error != SCI_ERR_OK) { \
            fprintf( \
                stderr, "Error at line %d in %s: %s failed: %s\n", \
                __LINE__, __FILE__, name, SCIGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

void sisci_init(
    int isServer, int remoteNodeId, sci_desc_t *sd, sci_map_t *localMap, sci_map_t *remoteMap,
    sci_local_segment_t *localSegment, sci_remote_segment_t *remoteSegment,
    struct server_segment **server_segment, struct client_segment **client_segment, struct c63_common *cm);

void sisci_create_interrupt(
    int isServer, int remoteNodeId, sci_desc_t *sd, sci_local_interrupt_t *localInterrupt,
    sci_remote_interrupt_t *remoteInterrupt);
