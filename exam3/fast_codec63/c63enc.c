#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
    fprintf(stderr, "Writing frame %d - keyframe\n", cm->framenum);
  }
  else
  {
    cm->curframe->keyframe = 0;
    fprintf(stderr, "Writing frame %d\n", cm->framenum);
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

  int ysize = cm->ypw * cm->yph;
  int usize = cm->upw * cm->uph;
  int vsize = cm->vpw * cm->vph;
  int mb_size = cm->mb_rows * cm->mb_cols;

  cm->curframe = malloc(sizeof(struct frame));
  cm->curframe->mbs[Y_COMPONENT] = calloc(mb_size, sizeof(struct macroblock));
  cm->curframe->mbs[U_COMPONENT] = calloc(mb_size / 4, sizeof(struct macroblock));
  cm->curframe->mbs[V_COMPONENT] = calloc(mb_size / 4, sizeof(struct macroblock));
  cm->curframe->residuals = create_dct(cm);

  FILE *outfile = errcheck_fopen(args->output_file, "wb");
  cm->e_ctx.fp = outfile;

  // init SISCI structs
  sisci_init(
      FALSE, args->remote_node, &sd, &localMap, &remoteMap, &localSegment, &remoteSegment,
      &server_segment, &client_segment, cm);
  sisci_create_interrupt(FALSE, args->remote_node, &sd, &localInterrupt, &remoteInterrupt);

  char *input_file = argv[optind];
  FILE *infile = errcheck_fopen(input_file, "rb");

  struct timespec total_start, total_end, wait_start, wait_end, write_start, write_end;
  double wait_time = 0, write_time = 0;

  clock_gettime(CLOCK_MONOTONIC_RAW, &total_start);

  /* Encode input frames */
  int framenum = 0;
  while (TRUE)
  {
    // send image to server
    if (framenum == 0) {
      if (!read_yuv(infile, cm, server_segment->image))
      {
        TRIGGER_DATA_INTERRUPT(remoteInterrupt, client_segment, CMD_QUIT, error);
        break;
      }
      SCIFlush(NULL, NO_FLAGS);

      TRIGGER_DATA_INTERRUPT(remoteInterrupt, client_segment, CMD_CONTINUE, error);
    }

    // wait until server has encoded image
    clock_gettime(CLOCK_MONOTONIC_RAW, &wait_start);
    SCIWaitForInterrupt(localInterrupt, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    clock_gettime(CLOCK_MONOTONIC_RAW, &wait_end);
    wait_time += TIME_IN_SECONDS(wait_start, wait_end);

    ERR_CHECK(error, "SCIWaitForInterrupt");
    if (*server_segment->cmd == CMD_QUIT) {
      fprintf(stderr, "Got quit message from server, this should not happen!\n");
      break;
    }

    // copy data from server to cm
    MEMCPY_DCT(cm->curframe->residuals, client_segment->residuals, ysize, usize, vsize);
    MEMCPY_MBS(cm->curframe->mbs, client_segment->mbs, mb_size);

    // we want to minimize the amount that the server waits, so we advance the frame before writing
    ++framenum;
    int must_exit = FALSE;
    if (args->frame_limit && framenum >= args->frame_limit)
    {
      must_exit = TRUE;
    } else if (!read_yuv(infile, cm, server_segment->image)) { 
      must_exit = TRUE;
    } else {
      SCIFlush(NULL, NO_FLAGS);
      TRIGGER_DATA_INTERRUPT(remoteInterrupt, client_segment, CMD_CONTINUE, error);
    }

    if (must_exit) {
      TRIGGER_DATA_INTERRUPT(remoteInterrupt, client_segment, CMD_QUIT, error);
    }

    // write image
    clock_gettime(CLOCK_MONOTONIC_RAW, &write_start);
    c63_write_image(cm);
    clock_gettime(CLOCK_MONOTONIC_RAW, &write_end);
    write_time += TIME_IN_SECONDS(write_start, write_end);

    if (must_exit) {
      break;
    }
  }

  clock_gettime(CLOCK_MONOTONIC_RAW, &total_end);
  double total_time = TIME_IN_SECONDS(total_start, total_end);

  double wait_percent = 100.0 * wait_time / total_time;
  double write_percent = 100.0 * write_time / total_time;
  double memcpy_percent = 100.0 * (1.0 - (wait_time + write_time) / total_time);
  fprintf(stderr, "Writer timings:\n");
  fprintf(stderr, "  Waiting: %f %%\n", wait_percent);
  fprintf(stderr, "  Writing: %f %%\n", write_percent);
  fprintf(stderr, "   Memcpy: %f %%\n", memcpy_percent);

  free(args);
  free_dct(cm->curframe->residuals);
  free(cm->curframe->mbs[Y_COMPONENT]);
  free(cm->curframe->mbs[U_COMPONENT]);
  free(cm->curframe->mbs[V_COMPONENT]);
  free(cm->curframe);
  free(cm);

  fclose(outfile);
  fclose(infile);

  SCITerminate();

  return EXIT_SUCCESS;
}
