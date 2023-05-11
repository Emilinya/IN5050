#include <assert.h>
#include <errno.h>
#include <getopt.h>
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

  int framenum = 0;
  while (TRUE)
  {
    // wait for data from client
    SCIWaitForInterrupt(localInterrupt, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    ERR_CHECK(error, "SCIWaitForInterrupt");
    if (*client_segment->cmd == CMD_QUIT) {
      break;
    }

    // copy data from client to cm
    MEMCPY_YUV(cm->curframe->orig, client_segment->image, ysize, usize, vsize);

    fprintf(stderr, "Encoding frame %d\n", framenum);
    c63_encode_image(cm);

    // copy data from cm to server
    MEMCPY_YUV(server_segment->reference_recons, cm->ref_recons, ysize, usize, vsize);
    MEMCPY_YUV(server_segment->currenct_recons, cm->curframe->recons, ysize, usize, vsize);
    MEMCPY_YUV(server_segment->predicted, cm->curframe->predicted, ysize, usize, vsize);
    MEMCPY_DCT(server_segment->residuals, cm->curframe->residuals, ysize, usize, vsize);
    MEMCPY_MBS(server_segment->mbs, cm->curframe->mbs, cm->mb_rows * cm->mb_cols);
    SCIFlush(NULL, NO_FLAGS);

    // tell client that we are done
    TRIGGER_DATA_INTERRUPT(remoteInterrupt, server_segment, CMD_CONTINUE, error);

    ++framenum;

    // swap frames to get ready for next frame
    yuv_t *tmp = cm->ref_recons;
    cm->ref_recons = cm->curframe->recons;
    cm->curframe->recons = tmp;
  }

  SCITerminate();
  free(args);

  return EXIT_SUCCESS;
}
