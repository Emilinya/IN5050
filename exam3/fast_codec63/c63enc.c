#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "c63.h"
#include "utils.h"
#include "tables.h"
#include "cl_utils.h"
#include "c63_write.h"
#include "sisci_utils.h"

int main(int argc, char **argv)
{
  int localAdapterNo = 0;
  cl_args_t *args = get_cl_args(argc, argv);

  if (args->frame_limit)
  {
    printf("Limited to %d frames.\n", args->frame_limit);
  }

  sci_desc_t sd;
  // sci_error_t error;
  sci_map_t localMap;
  sci_map_t remoteMap;
  sci_local_segment_t localSegment;
  sci_remote_segment_t remoteSegment;

  volatile struct server_segment *server_segment;
  volatile struct client_segment *client_segment;
  struct c63_common *cm = init_c63_enc(args->width, args->height);

  sisci_init(
      FALSE, localAdapterNo, args->remote_node, &sd, &localMap, &remoteMap,
      &localSegment, &remoteSegment, &server_segment, &client_segment, cm);

  FILE *outfile = errcheck_fopen(args->output_file, "wb");
  cm->e_ctx.fp = outfile;

  // cm contains pointers to server and client segments
  cm->curframe = calloc(1, sizeof(struct frame));
  cm->curframe->recons = server_segment->currenct_recons;
  cm->curframe->predicted = server_segment->predicted;
  cm->curframe->residuals = server_segment->residuals;
  cm->curframe->mbs[Y_COMPONENT] = server_segment->mbs[Y_COMPONENT];
  cm->curframe->mbs[U_COMPONENT] = server_segment->mbs[U_COMPONENT];
  cm->curframe->mbs[V_COMPONENT] = server_segment->mbs[V_COMPONENT];

  cm->ref_recons = server_segment->reference_recons;

  cm->curframe->orig = client_segment->image;

  char *input_file = argv[optind];
  FILE *infile = errcheck_fopen(input_file, "rb");

  /* Encode input frames */
  int numframes = 0;
  while (TRUE)
  {
    client_segment->cmd = CMD_WAIT;

    yuv_t *image = read_yuv(infile, cm);
    if (!image)
    {
      server_segment->cmd = CMD_QUIT;
      SCIFlush(NULL, NO_FLAGS);
      break;
    }

    // send image to server
    // memcpy(client_segment->image->Y, image->Y, cm->ypw * cm->yph * sizeof(uint8_t));
    // memcpy(client_segment->image->U, image->U, cm->upw * cm->uph * sizeof(uint8_t));
    // memcpy(client_segment->image->V, image->V, cm->vpw * cm->vph * sizeof(uint8_t));
    server_segment->cmd = CMD_DONE;
    SCIFlush(NULL, NO_FLAGS);

    free_yuv(image);

    // wait until server has encoded image
    while (client_segment->cmd == CMD_WAIT);

    // copy server data into our cm - we don't really need to copy the whole struct
    // memcpy(cm, (void *)&server_segment->cm, sizeof(struct c63_common));

    // write_frame(cm);

    ++numframes;
    if (args->frame_limit && numframes >= args->frame_limit)
    {
      server_segment->cmd = CMD_QUIT;
      SCIFlush(NULL, NO_FLAGS);
      break;
    }

    server_segment->cmd = CMD_DONE;
    SCIFlush(NULL, NO_FLAGS);
  }

  free(args);

  fclose(outfile);
  fclose(infile);

  SCITerminate();

  return EXIT_SUCCESS;
}
