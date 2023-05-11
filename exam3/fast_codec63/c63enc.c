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
  sci_local_interrupt_t localInterrupt;
  sci_remote_interrupt_t remoteInterrupt;

  struct server_segment *server_segment;
  struct client_segment *client_segment;

  struct c63_common *cm = init_c63_enc(args->width, args->height);
  cm->curframe = create_frame(cm);
  cm->ref_recons = create_yuv(cm);

  int ysize = cm->ypw * cm->yph;
  int usize = cm->upw * cm->uph;
  int vsize = cm->vpw * cm->vph;

  FILE *outfile = errcheck_fopen(args->output_file, "wb");
  cm->e_ctx.fp = outfile;

  sisci_init(
      FALSE, args->remote_node, &sd, &localMap, &remoteMap, &localSegment, &remoteSegment,
      &server_segment, &client_segment, cm);
  sisci_create_interrupt(FALSE, args->remote_node, &sd, &localInterrupt, &remoteInterrupt);

  SCITriggerInterrupt(remoteInterrupt, NO_FLAGS, &error);
  SCIWaitForInterrupt(localInterrupt, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);

  char *input_file = argv[optind];
  FILE *infile = errcheck_fopen(input_file, "rb");

  /* Encode input frames */
  int numframes = 0;
  while (TRUE)
  {
    yuv_t *image = read_yuv(infile, cm);
    if (!image)
    {
      TRIGGER_DATA_INTERRUPT(remoteInterrupt, client_segment, CMD_QUIT, error);
      break;
    }
    cm->curframe->orig = image;

    // send image to server
    MEMCPY_YUV(client_segment->image, cm->curframe->orig, ysize, usize, vsize);
    SCIFlush(NULL, NO_FLAGS);

    TRIGGER_DATA_INTERRUPT(remoteInterrupt, client_segment, CMD_CONTINUE, error);

    free_yuv(image);

    // wait until server has encoded image
    SCIWaitForInterrupt(localInterrupt, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    ERR_CHECK(error, "SCIWaitForInterrupt");
    if (*server_segment->cmd == CMD_QUIT) {
      fprintf(stderr, "Got quit message from server, this should not happen!\n");
      break;
    }

    // copy data from server to cm
    MEMCPY_YUV(cm->ref_recons, server_segment->reference_recons, ysize, usize, vsize);
    MEMCPY_YUV(cm->curframe->recons, server_segment->currenct_recons, ysize, usize, vsize);
    MEMCPY_YUV(cm->curframe->predicted, server_segment->predicted, ysize, usize, vsize);
    MEMCPY_DCT(cm->curframe->residuals, server_segment->residuals, ysize, usize, vsize);
    MEMCPY_MBS(cm->curframe->mbs, server_segment->mbs, cm->mb_rows * cm->mb_cols);

    fprintf(stderr, "Writing frame %d\n", numframes);
    c63_write_image(cm);

    ++numframes;
    if (args->frame_limit && numframes >= args->frame_limit)
    {
      TRIGGER_DATA_INTERRUPT(remoteInterrupt, client_segment, CMD_QUIT, error);
      break;
    }
  }

  free(args);

  fclose(outfile);
  fclose(infile);

  SCITerminate();

  return EXIT_SUCCESS;
}
