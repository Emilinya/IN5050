#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cl_utils.h"
#include "sisci_utils.h"

int main(int argc, char **argv)
{
  int localAdapterNo = 0;
  server_cl_args_t *args = get_server_cl_args(argc, argv);

  sci_desc_t sd;
  sci_map_t localMap;
  sci_map_t remoteMap;
  sci_local_segment_t localSegment;
  sci_remote_segment_t remoteSegment;

  volatile struct server_segment *server_segment;
  volatile struct client_segment *client_segment;

  sisci_init(
      1, localAdapterNo, args->remote_node, &sd, &localMap, &remoteMap,
      &localSegment, &remoteSegment, &server_segment, &client_segment);

  // Tell client that we are done
  client_segment->packet.cmd = CMD_DONE;
  SCIFlush(NULL, NO_FLAGS);

  // wait until client is done
  while (server_segment->packet.cmd == CMD_NULL);

  SCITerminate();
  free(args);

  return EXIT_SUCCESS;
}
