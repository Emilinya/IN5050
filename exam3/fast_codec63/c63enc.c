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

static void c63_write_image(struct c63_common *cm) {
  /* Check if keyframe */
  if (cm->framenum == 0 || cm->frames_since_keyframe == cm->keyframe_interval)
  {
    cm->curframe->keyframe = 1;
    cm->frames_since_keyframe = 0;
  }
  else
  {
    cm->curframe->keyframe = 0;
  }

  write_frame(cm);

  ++cm->framenum;
  ++cm->frames_since_keyframe;
}

int main(int argc, char **argv)
{
  unsigned int client_interrupt_number = CLIENT_INTERRUPT_NUMBER;

  cl_args_t *args = get_cl_args(argc, argv);

  if (args->frame_limit)
  {
    printf("Limited to %d frames.\n", args->frame_limit);
  }

  sci_desc_t sd;
  sci_error_t error;
  sci_map_t localMap;
  sci_map_t remoteMap;
  sci_local_segment_t localSegment;
  sci_remote_segment_t remoteSegment;
  sci_local_data_interrupt_t localInterrupt;
  sci_remote_data_interrupt_t remoteInterrupt;

  volatile struct server_segment *server_segment;
  volatile struct client_segment *client_segment;
  struct c63_common *cm = init_c63_enc(args->width, args->height);

  sisci_init(
      FALSE, LOCAL_ADAPTER_NUMBER, args->remote_node, &sd, &localMap, &remoteMap,
      &localSegment, &remoteSegment, &server_segment, &client_segment, cm);

  // Create an interupt, and connect to a remote interrupt
  SCICreateDataInterrupt(
      sd, &localInterrupt, LOCAL_ADAPTER_NUMBER, &client_interrupt_number,
      NULL, NULL, SCI_FLAG_FIXED_INTNO, &error);
  ERR_CHECK(error, "SCICreateDataInterrupt");

  while (TRUE) {
    SCIConnectDataInterrupt(
        sd, &remoteInterrupt, args->remote_node, LOCAL_ADAPTER_NUMBER,
        SERVER_INTERRUPT_NUMBER, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    if (error == SCI_ERR_NO_SUCH_INTNO) {
      continue;
    } else if (error == SCI_ERR_OK) {
      break;
    } else {
      ERR_CHECK(error, "SCIConnectDataInterrupt");
    }
  }
  printf("client connected\n");

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

  uint8_t localCommand = CMD_CONTINUE;
  uint8_t remoteCommand = CMD_QUIT;

  /* Encode input frames */
  int numframes = 0;
  while (TRUE)
  {
    yuv_t *image = read_yuv(infile, cm);
    if (!image)
    {
      localCommand = CMD_QUIT;
      SCITriggerDataInterrupt(remoteInterrupt, &localCommand, sizeof(uint8_t), NO_FLAGS, &error);
      ERR_CHECK(error, "SCITriggerDataInterrupt");
      break;
    }

    // send image to server
    memcpy(client_segment->image->Y, image->Y, cm->ypw * cm->yph * sizeof(uint8_t));
    memcpy(client_segment->image->U, image->U, cm->upw * cm->uph * sizeof(uint8_t));
    memcpy(client_segment->image->V, image->V, cm->vpw * cm->vph * sizeof(uint8_t));
    SCIFlush(NULL, NO_FLAGS);

    localCommand = CMD_CONTINUE;
    SCITriggerDataInterrupt(remoteInterrupt, &localCommand, sizeof(uint8_t), NO_FLAGS, &error);
    ERR_CHECK(error, "SCITriggerDataInterrupt");

    free_yuv(image);

    // wait until server has encoded image
    SCIWaitForDataInterrupt(localInterrupt, &remoteCommand, NULL, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    ERR_CHECK(error, "SCIWaitForDataInterrupt");
    if (remoteCommand == CMD_QUIT) {
      fprintf(stderr, "Got quit message from server, this should not happen!\n");
      break;
    }

    printf("Writing frame %d\n", numframes);
    c63_write_image(cm);

    ++numframes;
    if (args->frame_limit && numframes >= args->frame_limit)
    {
      localCommand = CMD_QUIT;
      SCITriggerDataInterrupt(remoteInterrupt, &localCommand, sizeof(uint8_t), NO_FLAGS, &error);
      ERR_CHECK(error, "SCITriggerDataInterrupt");
      break;
    }

    // swap frames to get ready for next frame
    yuv_t *tmp = cm->ref_recons;
    cm->ref_recons = cm->curframe->recons;
    cm->curframe->recons = tmp;
  }

  free(args);

  fclose(outfile);
  fclose(infile);

  SCITerminate();

  return EXIT_SUCCESS;
}
