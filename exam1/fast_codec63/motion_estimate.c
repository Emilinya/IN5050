#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arm_neon.h>

#include "cosine_transform.h"
#include "motion_estimate.h"
#include "utils.h"

__attribute__((always_inline)) void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
{
  uint16x8_t sum_v = vdupq_n_u16(0);
  for (int v = 0; v < 8; ++v)
  {
    // load 8 values from block1 and block2 in vectors
    uint8x8_t block1_v = vld1_u8(&block1[v * stride]);
    uint8x8_t block2_v = vld1_u8(&block2[v * stride]);

    // calculate absolute difference of all the values in one instruction
    uint8x8_t abdiff = vabd_u8(block1_v, block2_v);

    // add the absolute differences to the sum vector in a widening add
    sum_v = vaddw_u8(sum_v, abdiff);
  }
  // add the 8 values in the sum vector together to get the total sum
  *result = vaddvq_u16(sum_v);
}

/* Motion estimation for 8x8 block */
__attribute__((always_inline)) static void me_block_8x8(
    struct c63_common *cm, int mb_x, int mb_y,
    uint8_t *orig, uint8_t *ref, int color_component)
{
  struct macroblock *mb =
      &cm->curframe->mbs[color_component][mb_y * cm->padw[color_component] / 8 + mb_x];

  int range = cm->me_search_range;

  /* Quarter resolution for chroma channels. */
  if (color_component > 0)
  {
    range /= 2;
  }

  int w = cm->padw[color_component];
  int h = cm->padh[color_component];

  /* Make sure we are within bounds of reference frame. TODO: Support partial
     frame bounds. */
  int left = MAX(mb_x * 8 - range, 0);
  int top = MAX(mb_y * 8 - range, 0);
  int right = MIN(mb_x * 8 + range, w - 8);
  int bottom = MIN(mb_y * 8 + range, h - 8);

  int x, y;

  int mx = mb_x * 8;
  int my = mb_y * 8;

  int best_sad = INT_MAX;

  uint8_t *orig_block = orig + my * w + mx;

  for (y = top; y < bottom; ++y)
  {
    for (x = left; x < right; ++x)
    {
      int sad;
      sad_block_8x8(orig_block, ref + y * w + x, w, &sad);

      if (sad < best_sad)
      {
        mb->mv_x = x - mx;
        mb->mv_y = y - my;
        best_sad = sad;
      }
    }
  }

  /* Here, there should be a threshold on SAD that checks if the motion vector
     is cheaper than intraprediction. We always assume MV to be beneficial */

  /* printf("Using motion vector (%d, %d) with SAD %d\n", mb->mv_x, mb->mv_y,
     best_sad); */

  mb->use_mv = 1;
}

void c63_motion_estimate(struct c63_common *cm)
{
  /* Compare this frame with previous reconstructed frame */
  int mb_x, mb_y;

  /* Luma */
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
    {
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->Y,
                   cm->refframe->recons->Y, Y_COMPONENT);
    }
  }

  /* Chroma */
  for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
    {
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->U,
                   cm->refframe->recons->U, U_COMPONENT);
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->V,
                   cm->refframe->recons->V, V_COMPONENT);
    }
  }
}

/* Motion compensation for 8x8 block */
static void mc_block_8x8(struct c63_common *cm, int mb_x, int mb_y,
                         uint8_t *predicted, uint8_t *ref, int color_component)
{
  struct macroblock *mb =
      &cm->curframe->mbs[color_component][mb_y * cm->padw[color_component] / 8 + mb_x];

  if (!mb->use_mv)
  {
    return;
  }

  int left = mb_x * 8;
  int top = mb_y * 8;
  int right = left + 8;
  int bottom = top + 8;

  int w = cm->padw[color_component];

  /* Copy block from ref mandated by MV */
  int x, y;

  for (y = top; y < bottom; ++y)
  {
    for (x = left; x < right; ++x)
    {
      predicted[y * w + x] = ref[(y + mb->mv_y) * w + (x + mb->mv_x)];
    }
  }
}

void c63_motion_compensate(struct c63_common *cm)
{
  int mb_x, mb_y;

  /* Luma */
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
    {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->Y,
                   cm->refframe->recons->Y, Y_COMPONENT);
    }
  }

  /* Chroma */
  for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
    {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->U,
                   cm->refframe->recons->U, U_COMPONENT);
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->V,
                   cm->refframe->recons->V, V_COMPONENT);
    }
  }
}
