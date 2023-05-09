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
#include "motion_estimate.h"
#include "cosine_transform.h"

struct c63_common *init_c63_enc(int width, int height)
{
  struct c63_common *cm = calloc(1, sizeof(struct c63_common));

  cm->width = width;
  cm->height = height;

  cm->padw[Y_COMPONENT] = cm->ypw = (uint32_t)(ceil(width / 16.0f) * 16);
  cm->padh[Y_COMPONENT] = cm->yph = (uint32_t)(ceil(height / 16.0f) * 16);
  cm->padw[U_COMPONENT] = cm->upw = (uint32_t)(ceil(width * UX / (YX * 8.0f)) * 8);
  cm->padh[U_COMPONENT] = cm->uph = (uint32_t)(ceil(height * UY / (YY * 8.0f)) * 8);
  cm->padw[V_COMPONENT] = cm->vpw = (uint32_t)(ceil(width * VX / (YX * 8.0f)) * 8);
  cm->padh[V_COMPONENT] = cm->vph = (uint32_t)(ceil(height * VY / (YY * 8.0f)) * 8);

  cm->mb_cols = cm->ypw / 8;
  cm->mb_rows = cm->yph / 8;

  /* Quality parameters -- Home exam deliveries should have original values,
   i.e., quantization factor should be 25, search range should be 16, and the
   keyframe interval should be 100. */
  cm->qp = 25;                 // Constant quantization factor. Range: [1..50]
  cm->me_search_range = 16;    // Pixels in every direction
  cm->keyframe_interval = 100; // Distance between keyframes

  /* Initialize quantization tables */
  for (int i = 0; i < 64; ++i)
  {
    cm->quanttbl[Y_COMPONENT][i] = yquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[U_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[V_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
  }

  return cm;
}

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

  write_frame(cm);

  ++cm->framenum;
  ++cm->frames_since_keyframe;
}

int main(int argc, char **argv)
{
  int localAdapterNo = 0;
  encoder_cl_args_t *args = get_encoder_cl_args(argc, argv);

  sci_desc_t sd;
  sci_map_t localMap;
  sci_map_t remoteMap;
  sci_local_segment_t localSegment;
  sci_remote_segment_t remoteSegment;

  volatile struct server_segment *server_segment;
  volatile struct client_segment *client_segment;

  if (args->frame_limit)
  {
    printf("Limited to %d frames.\n", args->frame_limit);
  }

  FILE *outfile = errcheck_fopen(args->output_file, "wb");

  struct c63_common *cm = init_c63_enc(args->width, args->height);
  cm->e_ctx.fp = outfile;
  cm->curframe = create_frame(cm);
  cm->ref_recons = create_yuv(cm);

  char *input_file = argv[optind];
  FILE *infile = errcheck_fopen(input_file, "rb");

  sisci_init(
      0, localAdapterNo, args->remote_node, &sd, &localMap, &remoteMap,
      &localSegment, &remoteSegment, &server_segment, &client_segment);

  /* Encode input frames */
  int numframes = 0;
  while (1)
  {
    yuv_t *image = read_yuv(infile, cm);
    if (!image)
    {
      break;
    }
    cm->curframe->orig = image;

    printf("Encoding frame %d\n", numframes);
    c63_encode_image(cm);

    free_yuv(image);

    ++numframes;

    if (args->frame_limit && numframes >= args->frame_limit)
    {
      break;
    }

    // swap frames to get ready for next frame
    yuv_t *tmp = cm->ref_recons;
    cm->ref_recons = cm->curframe->recons;
    cm->curframe->recons = tmp;
  }

  // tell server that we are done
  server_segment->packet.cmd = CMD_DONE;
  SCIFlush(NULL, NO_FLAGS);

  // wait until server is done
  while (client_segment->packet.cmd == CMD_NULL);

  destroy_frame(cm->curframe);
  free_yuv(cm->ref_recons);
  free(args);
  free(cm);

  fclose(outfile);
  fclose(infile);

  SCITerminate();

  return EXIT_SUCCESS;
}
