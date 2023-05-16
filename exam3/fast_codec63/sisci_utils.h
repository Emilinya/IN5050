#pragma once

#include <stdio.h>
#include <stdlib.h>

#include <sisci_error.h>
#include <sisci_api.h>

#include "sisci_common.h"

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
