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
  int localAdapterNo = 0;
  cl_args_t *args = get_cl_args(argc, argv);

  sci_desc_t sd;
  sci_map_t localMap;
  sci_map_t remoteMap;
  sci_local_segment_t localSegment;
  sci_remote_segment_t remoteSegment;

  volatile struct server_segment *server_segment;
  volatile struct client_segment *client_segment;
  struct c63_common *cm = init_c63_enc(args->width, args->height);

  sisci_init(
      TRUE, localAdapterNo, args->remote_node, &sd, &localMap, &remoteMap,
      &localSegment, &remoteSegment, &server_segment, &client_segment, cm);

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

  int framenum = 0;
  while (TRUE)
  {
    server_segment->cmd = CMD_WAIT;

    // wait until we have recived data from the client 
    while (server_segment->cmd == CMD_WAIT);
    if (server_segment->cmd == CMD_QUIT) {
      break;
    }

    printf("Encoding frame %d\n", framenum);
    // c63_encode_image(cm);

    client_segment->cmd = CMD_DONE;
    SCIFlush(NULL, NO_FLAGS);

    framenum++;

    // swap frames to get ready for next frame
    // yuv_t *tmp = cm->ref_recons;
    // cm->ref_recons = cm->curframe->recons;
    // cm->curframe->recons = tmp;
  }

  SCITerminate();
  free(args);

  return EXIT_SUCCESS;
}
