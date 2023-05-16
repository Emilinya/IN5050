#include <assert.h>
#include <errno.h>
#include <getopt.h>
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
#include "sisci_utils.h"
#include "motion_estimate.h"
#include "cosine_transform.h"

static void c63_encode_image(struct c63_common *cm)
{
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

  if (!cm->curframe->keyframe)
  {
    /* Motion Estimation */
    c63_motion_estimate(cm);

    /* Motion Compensation */
    c63_motion_compensate(cm);
  }

  /* DCT and Quantization */
  dct_quantize(cm->curframe->orig->Y, cm->curframe->predicted->Y, cm->ypw,
               cm->yph, cm->curframe->residuals->Ydct,
               cm->quanttbl[Y_COMPONENT]);

  dct_quantize(cm->curframe->orig->U, cm->curframe->predicted->U, cm->padw[U_COMPONENT],
               cm->uph, cm->curframe->residuals->Udct,
               cm->quanttbl[U_COMPONENT]);

  dct_quantize(cm->curframe->orig->V, cm->curframe->predicted->V, cm->vpw,
               cm->vph, cm->curframe->residuals->Vdct,
               cm->quanttbl[V_COMPONENT]);

  /* Reconstruct frame for inter-prediction */
  dequantize_idct(cm->curframe->residuals->Ydct, cm->curframe->predicted->Y,
                  cm->ypw, cm->yph, cm->curframe->recons->Y, cm->quanttbl[Y_COMPONENT]);
  dequantize_idct(cm->curframe->residuals->Udct, cm->curframe->predicted->U,
                  cm->upw, cm->uph, cm->curframe->recons->U, cm->quanttbl[U_COMPONENT]);
  dequantize_idct(cm->curframe->residuals->Vdct, cm->curframe->predicted->V,
                  cm->vpw, cm->vph, cm->curframe->recons->V, cm->quanttbl[V_COMPONENT]);

  ++cm->framenum;
  ++cm->frames_since_keyframe;
}

int main(int argc, char **argv)
{
  cl_args_t *args = get_cl_args(argc, argv);

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
  cm->curframe->orig = create_yuv(cm);

  int ysize = cm->ypw * cm->yph;
  int usize = cm->upw * cm->uph;
  int vsize = cm->vpw * cm->vph;

  sisci_init(
      TRUE, args->remote_node, &sd, &localMap, &remoteMap, &localSegment, &remoteSegment,
      &server_segment, &client_segment, cm);
  sisci_create_interrupt(TRUE, args->remote_node, &sd, &localInterrupt, &remoteInterrupt);

  cm->curframe->orig = server_segment->image;

  struct timespec total_start, total_end, wait_start, wait_end, encode_start, encode_end;
  double wait_time = 0, encode_time = 0;

  clock_gettime(CLOCK_MONOTONIC_RAW, &total_start);

  int framenum = 0;
  while (TRUE)
  {
    // wait for data from client
    clock_gettime(CLOCK_MONOTONIC_RAW, &wait_start);
    SCIWaitForInterrupt(localInterrupt, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    clock_gettime(CLOCK_MONOTONIC_RAW, &wait_end);
    wait_time += TIME_IN_SECONDS(wait_start, wait_end);

    ERR_CHECK(error, "SCIWaitForInterrupt");
    if (*client_segment->cmd == CMD_QUIT) {
      break;
    }

    fprintf(stderr, "Encoding frame %d\n", framenum);
    clock_gettime(CLOCK_MONOTONIC_RAW, &encode_start);
    c63_encode_image(cm);
    clock_gettime(CLOCK_MONOTONIC_RAW, &encode_end);
    encode_time += TIME_IN_SECONDS(encode_start, encode_end);

    // copy data from cm to server
    MEMCPY_YUV(client_segment->reference_recons, cm->ref_recons, ysize, usize, vsize);
    MEMCPY_YUV(client_segment->currenct_recons, cm->curframe->recons, ysize, usize, vsize);
    MEMCPY_YUV(client_segment->predicted, cm->curframe->predicted, ysize, usize, vsize);
    MEMCPY_DCT(client_segment->residuals, cm->curframe->residuals, ysize, usize, vsize);
    MEMCPY_MBS(client_segment->mbs, cm->curframe->mbs, cm->mb_rows * cm->mb_cols);
    SCIFlush(NULL, NO_FLAGS);

    // tell client that we are done
    TRIGGER_DATA_INTERRUPT(remoteInterrupt, server_segment, CMD_CONTINUE, error);

    ++framenum;

    // swap frames to get ready for next frame
    yuv_t *tmp = cm->ref_recons;
    cm->ref_recons = cm->curframe->recons;
    cm->curframe->recons = tmp;
  }

  clock_gettime(CLOCK_MONOTONIC_RAW, &total_end);
  double total_time = TIME_IN_SECONDS(total_start, total_end);

  double wait_percent = 100.0 * wait_time / total_time;
  double encode_percent = 100.0 * encode_time / total_time;
  double memcpy_percent = 100.0 * (1.0 - (wait_time + encode_time) / total_time);
  fprintf(stderr, "Encoder timings:\n");
  fprintf(stderr, "   Waiting: %f %%\n", wait_percent);
  fprintf(stderr, "  Encoding: %f %%\n", encode_percent);
  fprintf(stderr, "    Memcpy: %f %%\n", memcpy_percent);

  free(args);

  SCITerminate();

  return EXIT_SUCCESS;
}
