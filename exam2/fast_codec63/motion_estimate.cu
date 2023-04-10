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
    int size = bounds_w * bounds_h;

    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    int mx = mb_x * 8;
    int my = mb_y * 8;

    uint8_t *origin = orig + mx + my * w;
    uint8_t *reference = ref + left + top * w;
    int ref_idx = threadIdx.x + threadIdx.y * w;

    // copy origin into fast shared memory
    __shared__ uint8_t temp_orig[8 * 8];
    if (threadIdx.x < 8 && threadIdx.y < 8) {
        temp_orig[threadIdx.x + threadIdx.y * 8] = origin[ref_idx];
    }
    
    // we access (bounds_w + 7) * (bounds_h + 7) items from ref.
    // When copying over to temp_ref, we might not have enough threads,
    // so we copy in multiple moves
    __shared__ uint8_t temp_ref[(32+7) * (32+7)];
    int temp_ref_w = bounds_w + 7;
    int temp_ref_h = bounds_h + 7;
    
    int shifted_tid = tid;
    int remaining = temp_ref_w * temp_ref_h;
    while (remaining > 0) {
        if (tid < remaining) {
            int y = shifted_tid / temp_ref_w;
            int x = shifted_tid - y * temp_ref_w;
            temp_ref[shifted_tid] = reference[x + y * w];
        }
        remaining -= blockDim.x * blockDim.y;
        shifted_tid += blockDim.x * blockDim.y;
    }
    __syncthreads();
    
    __shared__ uint16_t sad_grid[32 * 32];
    if (threadIdx.x < bounds_w && threadIdx.y < bounds_h) {
        int ref_id = threadIdx.x + threadIdx.y * temp_ref_w;

        uint16_t abssum = 0;
        for (int v = 0; v < 8; ++v)
        {
            for (int u = 0; u < 8; ++u)
            {
                abssum += abs(temp_orig[u + v * 8] - temp_ref[ref_id + u + v * temp_ref_w]);
            }
        }
        sad_grid[threadIdx.x + threadIdx.y * bounds_w] = abssum;
    }
    __syncthreads();

    // optimal reduction number is sqrt(size) - set reduction number
    // to first order taylor expansion of sqrt(x) around 640
    const int reduction_num = 0.01976423537605 * (float)size + 12.64911064067;
    __shared__ uint16_t best_i_list[32];

    if (tid < reduction_num) {
        int start = tid * size / reduction_num;
        int end = (tid + 1) * size / reduction_num;

        int best_i = start;
        for (int i = start + 1; i < end; ++i)
        {
            if (sad_grid[i] < sad_grid[best_i])
            {
                best_i = i;
            }
        }
        best_i_list[tid] = best_i;
    }
    __syncthreads();

    if (tid == 0) {
        int idx = 2 * (blockIdx.x + blockIdx.y * gridDim.x);

        int best_i = best_i_list[0];
        for (int i = 1; i < reduction_num; ++i)
        {
            if (sad_grid[best_i_list[i]] < sad_grid[best_i])
            {
                best_i = best_i_list[i];
            }
        }

        mv_vecs[idx] = left + (best_i % bounds_w) - mx;
        mv_vecs[idx + 1] = top + (best_i / bounds_w) - my;
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
