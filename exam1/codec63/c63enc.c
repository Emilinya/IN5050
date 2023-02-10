#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "me.h"
#include "c63.h"
#include "utils.h"
#include "common.h"
#include "tables.h"
#include "c63_write.h"

/* getopt */
extern int optind;
extern char *optarg;

static void print_help()
{
  printf("Usage: ./c63enc [options] input_file\n");
  printf("Commandline options:\n");
  printf("  -h                             Height of images to compress\n");
  printf("  -w                             Width of images to compress\n");
  printf("  -o                             Output file (.c63)\n");
  printf("  [-f]                           Limit number of frames to encode\n");
  printf("\n");

  exit(EXIT_FAILURE);
}

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

void free_c63_enc(struct c63_common *cm)
{
  destroy_frame(cm->curframe);
  free(cm);
}

static void c63_encode_image(struct c63_common *cm, yuv_t *image)
{
  /* Advance to next frame */
  destroy_frame(cm->refframe);
  cm->refframe = cm->curframe;
  cm->curframe = create_frame(cm, image);

  /* Check if keyframe */
  if (cm->framenum == 0 || cm->frames_since_keyframe == cm->keyframe_interval)
  {
    cm->curframe->keyframe = 1;
    cm->frames_since_keyframe = 0;

    fprintf(stderr, " (keyframe) ");
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
  dct_quantize(image->Y, cm->curframe->predicted->Y, cm->padw[Y_COMPONENT],
               cm->padh[Y_COMPONENT], cm->curframe->residuals->Ydct,
               cm->quanttbl[Y_COMPONENT]);

  dct_quantize(image->U, cm->curframe->predicted->U, cm->padw[U_COMPONENT],
               cm->padh[U_COMPONENT], cm->curframe->residuals->Udct,
               cm->quanttbl[U_COMPONENT]);

  dct_quantize(image->V, cm->curframe->predicted->V, cm->padw[V_COMPONENT],
               cm->padh[V_COMPONENT], cm->curframe->residuals->Vdct,
               cm->quanttbl[V_COMPONENT]);

  /* Reconstruct frame for inter-prediction */
  dequantize_idct(cm->curframe->residuals->Ydct, cm->curframe->predicted->Y,
                  cm->ypw, cm->yph, cm->curframe->recons->Y, cm->quanttbl[Y_COMPONENT]);
  dequantize_idct(cm->curframe->residuals->Udct, cm->curframe->predicted->U,
                  cm->upw, cm->uph, cm->curframe->recons->U, cm->quanttbl[U_COMPONENT]);
  dequantize_idct(cm->curframe->residuals->Vdct, cm->curframe->predicted->V,
                  cm->vpw, cm->vph, cm->curframe->recons->V, cm->quanttbl[V_COMPONENT]);

  /* Function dump_image(), found in common.c, can be used here to check if the
     prediction is correct */

  write_frame(cm);

  ++cm->framenum;
  ++cm->frames_since_keyframe;
}

int main(int argc, char **argv)
{
  if (argc == 1)
  {
    print_help();
  }

  int c;
  static uint32_t width = 0;
  static uint32_t height = 0;
  static int frame_limit = 0;
  static char *output_file = NULL;
  while ((c = getopt(argc, argv, "h:w:o:f:i:")) != -1)
  {
    switch (c)
    {
    case 'h':
      height = atoi(optarg);
      break;
    case 'w':
      width = atoi(optarg);
      break;
    case 'o':
      output_file = optarg;
      break;
    case 'f':
      frame_limit = atoi(optarg);
      break;
    default:
      print_help();
      break;
    }
  }

  if (!width || !height || !output_file)
  {
    fprintf(stderr, "Too few program options, see --help.\n");
    exit(EXIT_FAILURE);
  }

  if (optind >= argc)
  {
    fprintf(stderr, "Error getting program options, try --help.\n");
    exit(EXIT_FAILURE);
  }

  FILE *outfile = errcheck_fopen(output_file, "wb");

  struct c63_common *cm = init_c63_enc(width, height);
  cm->e_ctx.fp = outfile;

  if (frame_limit)
  {
    printf("Limited to %d frames.\n", frame_limit);
  }

  char *input_file = argv[optind];
  FILE *infile = errcheck_fopen(input_file, "rb");

  /* Encode input frames */
  int numframes = 0;

  while (1)
  {
    yuv_t *image = read_yuv(infile, cm);
    if (!image)
    {
      break;
    }

    printf("Encoding frame %d, ", numframes);
    c63_encode_image(cm, image);

    free_yuv(image);
    printf("Done!\n");

    ++numframes;

    if (frame_limit && numframes >= frame_limit)
    {
      break;
    }
  }

  free_c63_enc(cm);
  fclose(outfile);
  fclose(infile);

  return EXIT_SUCCESS;
}
