#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sisci_error.h>
#include <sisci_api.h>

#include "c63.h"
#include "utils.h"
#include "tables.h"
#include "cl_utils.h"
#include "c63_write.h"
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

  /* Function dump_image(), found in utils.c, can be used here to check if the
     prediction is correct */

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

  FILE *outfile = errcheck_fopen(args->output_file, "wb");

  struct c63_common *cm = init_c63_enc(args->width, args->height);
  cm->e_ctx.fp = outfile;
  cm->curframe = create_frame(cm);
  cm->ref_recons = create_yuv(cm);

  char *input_file = argv[optind];
  FILE *infile = errcheck_fopen(input_file, "rb");

  sci_desc_t sd;
  sci_error_t error;

  SCIInitialize(0, &error);
  if (error != SCI_ERR_OK)
  {
    fprintf(stderr, "SCIInitialize failed: %s\n", SCIGetErrorString(error));
    exit(EXIT_FAILURE);
  }

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

  destroy_frame(cm->curframe);
  free_yuv(cm->ref_recons);
  free(args);
  free(cm);

  fclose(outfile);
  fclose(infile);

  SCITerminate();

  return EXIT_SUCCESS;
}
