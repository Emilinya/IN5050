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

/* Motion estimation for 8x8 block */
__global__ void me_block_8x8(
    int w, int h, uint8_t* mv_vecs, uint8_t *orig, uint8_t *ref, int color_component)
{
    int mb_x = blockIdx.x;
    int mb_y = blockIdx.y;

    int range = blockDim.x / 2;

    // Make sure we are within bounds of reference frame. TODO: Support partial frame bounds.
    int left = MAX(mb_x * 8 - range, 0);
    int top = MAX(mb_y * 8 - range, 0);
    int right = MIN(mb_x * 8 + range, w - 8);
    int bottom = MIN(mb_y * 8 + range, h - 8);

    int bounds_w = right - left;
    int bounds_h = bottom - top;

    int mx = mb_x * 8;
    int my = mb_y * 8;

    // I would like not to hardcode this, but oh well 
    __shared__ uint16_t sad_grid[32 * 32];

    if (threadIdx.x < bounds_w && threadIdx.y < bounds_h) {
        uint8_t *origin = orig + mx + my * w;
        uint8_t *reference = ref + (left + threadIdx.x) + (top + threadIdx.y) * w;
    
        uint16_t abssum = 0;
        for (int v = 0; v < 8; ++v)
        {
            for (int u = 0; u < 8; ++u)
            {
                abssum += abs(origin[u + v*w] - reference[u + v*w]);
            }
        }
        sad_grid[threadIdx.x + threadIdx.y * bounds_w] = abssum;
    }
    
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int idx = 2 * (blockIdx.x + blockIdx.y * gridDim.x);
        int best_sad = INT_MAX;
        for (int y = 0; y < bounds_h; ++y)
        {
            for (int x = 0; x < bounds_w; ++x)
            {
                int i = x + y * bounds_w;
                if (sad_grid[i] < best_sad)
                {
                    mv_vecs[idx] = left + x - mx;
                    mv_vecs[idx + 1] = top + y - my;
                    best_sad = sad_grid[i];
                }
            }
        }
    }
}

__host__ void c63_motion_estimate(struct c63_common *cm, struct gpu_frame *gpu_frame)
{
    // copy data to gpu
    memcpy(gpu_frame->input->Y, cm->curframe->orig->Y, cm->ypw * cm->yph);
    memcpy(gpu_frame->input->U, cm->curframe->orig->U, cm->upw * cm->uph);
    memcpy(gpu_frame->input->V, cm->curframe->orig->V, cm->vpw * cm->vph);

    memcpy(gpu_frame->reference->Y, cm->ref_recons->Y, cm->ypw * cm->yph);
    memcpy(gpu_frame->reference->U, cm->ref_recons->U, cm->upw * cm->uph);
    memcpy(gpu_frame->reference->V, cm->ref_recons->V, cm->vpw * cm->vph);

    // define block grid
    dim3 block_grid_Y;
    dim3 block_grid_UV;
    block_grid_Y.x = cm->mb_cols;
    block_grid_Y.y = cm->mb_rows;
    block_grid_UV.x = cm->mb_cols / 2;
    block_grid_UV.y = cm->mb_rows / 2;
    
    // define thread grid
    dim3 thread_grid_Y;
    dim3 thread_grid_UV;
    thread_grid_Y.x = cm->me_search_range * 2;
    thread_grid_Y.y = cm->me_search_range * 2;
    thread_grid_UV.x = cm->me_search_range;
    thread_grid_UV.y = cm->me_search_range;

    // create array for motion vectors
    uint8_t *mv_vecs;
    cudaMallocManaged((void **)&mv_vecs, 2 * block_grid_Y.x * block_grid_Y.y);

    // Luma
    me_block_8x8 <<<block_grid_Y, thread_grid_Y>>> (
        cm->padw[Y_COMPONENT], cm->padh[Y_COMPONENT], mv_vecs,
        gpu_frame->input->Y, gpu_frame->reference->Y, Y_COMPONENT);
    cudaDeviceSynchronize();

    for (int mb_y = 0; mb_y < block_grid_Y.y; ++mb_y)
    {
        for (int mb_x = 0; mb_x < block_grid_Y.x; ++mb_x)
        {
            int idx = 2 * (mb_x + mb_y * block_grid_Y.x);
            struct macroblock *mb =
                &cm->curframe->mbs[Y_COMPONENT][mb_y * cm->padw[Y_COMPONENT] / 8 + mb_x];

            mb->mv_x = mv_vecs[idx];
            mb->mv_y = mv_vecs[idx + 1];
            mb->use_mv = 1;
        }
    }

    // Chroma U
    me_block_8x8 <<<block_grid_UV, thread_grid_UV>>> (
        cm->padw[U_COMPONENT], cm->padh[U_COMPONENT], mv_vecs,
        gpu_frame->input->U, gpu_frame->reference->U, U_COMPONENT);
    cudaDeviceSynchronize();

    for (int mb_y = 0; mb_y < block_grid_UV.y; ++mb_y)
    {
        for (int mb_x = 0; mb_x < block_grid_UV.x; ++mb_x)
        {
            int idx = 2 * (mb_x + mb_y * block_grid_UV.x);
            struct macroblock *mb =
                &cm->curframe->mbs[U_COMPONENT][mb_y * cm->padw[U_COMPONENT] / 8 + mb_x];

            mb->mv_x = mv_vecs[idx];
            mb->mv_y = mv_vecs[idx + 1];
            mb->use_mv = 1;
        }
    }

    // Chroma V
    me_block_8x8 <<<block_grid_UV, thread_grid_UV>>> (
        cm->padw[V_COMPONENT], cm->padh[V_COMPONENT], mv_vecs,
        gpu_frame->input->V, gpu_frame->reference->V, V_COMPONENT);
    cudaDeviceSynchronize();

    for (int mb_y = 0; mb_y < block_grid_UV.y; ++mb_y)
    {
        for (int mb_x = 0; mb_x < block_grid_UV.x; ++mb_x)
        {
            int idx = 2 * (mb_x + mb_y * block_grid_UV.x);
            struct macroblock *mb =
                &cm->curframe->mbs[V_COMPONENT][mb_y * cm->padw[V_COMPONENT] / 8 + mb_x];

            mb->mv_x = mv_vecs[idx];
            mb->mv_y = mv_vecs[idx + 1];
            mb->use_mv = 1;
        }
    }

    cudaFree(mv_vecs);
}
