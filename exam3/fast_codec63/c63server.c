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
  unsigned int server_interrupt_number = SERVER_INTERRUPT_NUMBER;
  cl_args_t *args = get_cl_args(argc, argv);

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
      TRUE, LOCAL_ADAPTER_NUMBER, args->remote_node, &sd, &localMap, &remoteMap,
      &localSegment, &remoteSegment, &server_segment, &client_segment, cm);

  // Create an interupt, and connect to a remote interrupt
  SCICreateDataInterrupt(
      sd, &localInterrupt, LOCAL_ADAPTER_NUMBER, &server_interrupt_number,
      NULL, NULL, SCI_FLAG_FIXED_INTNO, &error);
  ERR_CHECK(error, "SCICreateDataInterrupt");

  while (TRUE) {
    SCIConnectDataInterrupt(
        sd, &remoteInterrupt, args->remote_node, LOCAL_ADAPTER_NUMBER,
        CLIENT_INTERRUPT_NUMBER, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    if (error == SCI_ERR_NO_SUCH_INTNO) {
      continue;
    } else if (error == SCI_ERR_OK) {
      break;
    } else {
      ERR_CHECK(error, "SCIConnectDataInterrupt");
    }
  }
  printf("server connected\n");

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

  uint8_t localCommand = CMD_CONTINUE;
  uint8_t remoteCommand = CMD_QUIT;

  int framenum = 0;
  while (TRUE)
  {
    // wait for data from client
    SCIWaitForDataInterrupt(
        localInterrupt, &remoteCommand, NULL, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
    ERR_CHECK(error, "SCIWaitForDataInterrupt");
    if (remoteCommand == CMD_QUIT) {
      break;
    }

    printf("Encoding frame %d\n", framenum);
    c63_encode_image(cm);
    SCIFlush(NULL, NO_FLAGS);

    // tell client that we are done
    SCITriggerDataInterrupt(remoteInterrupt, &localCommand, sizeof(uint8_t), NO_FLAGS, &error);
    ERR_CHECK(error, "SCITriggerDataInterrupt");

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
