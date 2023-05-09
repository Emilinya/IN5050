#pragma once

#include <stdio.h>
#include <stdlib.h>

#include <sisci_error.h>
#include <sisci_api.h>

#include "sisci_common.h"

void sisci_init(
    int initServer, int localAdapterNo, int remoteNodeId, sci_desc_t *sd, sci_map_t *localMap,
    sci_map_t *remoteMap, sci_local_segment_t *localSegment, sci_remote_segment_t *remoteSegment,
    volatile struct server_segment **server_segment, volatile struct client_segment **client_segment);
char *print_error(sci_error_t error);

#define ERR_CHECK(error, name) \
    do { \
        if (error != SCI_ERR_OK) { \
            fprintf( \
                stderr, "%s at line %d in %s failed - Got error %s\n", \
                name, __LINE__, __FILE__, print_error(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
