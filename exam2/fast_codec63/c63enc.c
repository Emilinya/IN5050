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

  if (args->run_count)
  {
    printf("Running encoder to %d times.\n", args->run_count);
  }
  if (args->frame_limit)
  {
    printf("Limited to %d frames.\n", args->frame_limit);
  }

  double *runtimes = calloc(args->run_count, sizeof(double));
  int num_runs = 1;
  if (args->run_count)
  {
    num_runs = args->run_count;
  }

  for (int i = 0; i < num_runs; i++)
  {
    FILE *outfile = errcheck_fopen(args->output_file, "wb");

    struct c63_common *cm = init_c63_enc(args->width, args->height);
    cm->e_ctx.fp = outfile;
    cm->curframe = create_frame(cm);
    cm->ref_recons = create_yuv(cm);

    char *input_file = argv[optind];
    FILE *infile = errcheck_fopen(input_file, "rb");

    /* Encode input frames */
    int numframes = 0;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    while (1)
    {
      yuv_t *image = read_yuv(infile, cm);
      if (!image)
      {
        break;
      }
      cm->curframe->orig = image;

      printf("\rEncoding frame %d", numframes);
      fflush(stdout);
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
    printf("\n");
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    if (args->run_count)
    {
      double time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
      runtimes[i] = time;
    }

    destroy_frame(cm->curframe);
    free_yuv(cm->ref_recons);
    free(cm);

    fclose(outfile);
    fclose(infile);
  }

  if (args->run_count)
  {
    double avg = 0;
    double min = 1e6;
    double max = 0;
    for (size_t i = 0; i < args->run_count; i++)
    {
      avg += runtimes[i];
      min = MIN(min, runtimes[i]);
      max = MAX(max, runtimes[i]);
    }
    avg /= (double)args->run_count;

    double std = 0;
    for (size_t i = 0; i < args->run_count; i++)
    {
      double diff = runtimes[i] - avg;
      std += diff * diff;
    }
    std = sqrt(std) / (double)args->run_count;

    FILE *perf_file = errcheck_fopen("profiling/runtimes.txt", "w");
    fprintf(
        perf_file, "Runtime data from %d runs. Each run encoded %d frames of a %dx%d video.\n",
        args->run_count, args->frame_limit, args->width, args->height);
    fprintf(perf_file, "avg ± std [s] | min [s] | max [s]\n");
    fprintf(perf_file, "%f ± %f | %f | %f\n", avg, std, min, max);
    fprintf(perf_file, "\nData:\n");
    for (size_t i = 0; i < args->run_count - 1; i++)
    {
      fprintf(perf_file, "%f ", runtimes[i]);
    }
    fprintf(perf_file, "%f\n", runtimes[args->run_count - 1]);
    fclose(perf_file);
  }

  free(runtimes);
  free(args);

  return EXIT_SUCCESS;
}
