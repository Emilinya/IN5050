#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cosine_transform.h"
#include "motion_estimate.h"
#include "utils.h"

__global__ void sad_block_8x8(
    uint8_t *block1, uint8_t *block2, int top,
    int left, int stride, uint16_t *sad_grid)
{
    uint8_t *my_block2 = block2 + (left + threadIdx.x) + (top + threadIdx.y) * stride;

    uint16_t abssum = 0;
    for (int v = 0; v < 8; ++v)
    {
        for (int u = 0; u < 8; ++u)
        {
            abssum += abs(block1[u + v*stride] - my_block2[u + v*stride]);
        }
    }
    sad_grid[threadIdx.x + threadIdx.y * blockDim.x] = abssum;
}

/* Motion estimation for 8x8 block */
__host__ static void me_block_8x8(
    struct c63_common *cm, uint16_t* sad_grid, int mb_x, int mb_y,
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

    int mx = mb_x * 8;
    int my = mb_y * 8;
    uint8_t *origin = orig + my * w + mx;
    
    // define block and thread size
    dim3 block_grid;
    block_grid.x = 1;
    block_grid.y = 1;

    dim3 thread_grid;
    thread_grid.x = right - left;
    thread_grid.y = bottom - top;
    
    // sad_block_8x8 <<<block_grid, thread_grid>>> (origin, ref, top, left, w, sad_grid);
    // cudaDeviceSynchronize();
 
    int best_sad = INT_MAX;
    for (int y = top; y < bottom; ++y)
    {
        for (int x = left; x < right; ++x)
        {
            int i = x-left + (y-top) * thread_grid.x;
            if (sad_grid[i] < best_sad)
            {
                mb->mv_x = x - mx;
                mb->mv_y = y - my;
                best_sad = sad_grid[i];
            }
        }
    }

    mb->use_mv = 1;
}

__host__ void c63_motion_estimate(struct c63_common *cm, struct gpu_frame *gpu_frame)
{
    /* Compare this frame with previous reconstructed frame */
    int mb_x, mb_y;

    // copy data to gpu
    memcpy(gpu_frame->input->Y, cm->curframe->orig->Y, cm->ypw * cm->yph);
    memcpy(gpu_frame->input->U, cm->curframe->orig->U, cm->upw * cm->uph);
    memcpy(gpu_frame->input->V, cm->curframe->orig->V, cm->vpw * cm->vph);

    memcpy(gpu_frame->reference->Y, cm->ref_recons->Y, cm->ypw * cm->yph);
    memcpy(gpu_frame->reference->U, cm->ref_recons->U, cm->upw * cm->uph);
    memcpy(gpu_frame->reference->V, cm->ref_recons->V, cm->vpw * cm->vph);

    // create array for sad values
    uint16_t *sad_grid;
    cudaMallocManaged(
        (void **)&sad_grid,
        4 * cm->me_search_range * cm->me_search_range * sizeof(uint16_t));

    for (int i = 0; i < 4 * cm->me_search_range * cm->me_search_range; i++) {
        sad_grid[i] = 0;
    }

    /* Luma */
    for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
    {
        for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
        {
            me_block_8x8(
                cm, sad_grid, mb_x, mb_y, gpu_frame->input->Y,
                gpu_frame->reference->Y, Y_COMPONENT);
        }
    }

    /* Chroma */
    for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
    {
        for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
        {
            me_block_8x8(
                cm, sad_grid, mb_x, mb_y, gpu_frame->input->U,
                gpu_frame->reference->U, U_COMPONENT);
            me_block_8x8(
                cm, sad_grid, mb_x, mb_y, gpu_frame->input->V,
                gpu_frame->reference->V, V_COMPONENT);
        }
    }
    cudaFree(sad_grid);
}
