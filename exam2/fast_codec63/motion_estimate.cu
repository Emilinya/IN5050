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
    int w, int h, uint8_t *orig, uint8_t *ref, struct macroblock *mbs)
{
    int mb_x = blockIdx.x;
    int mb_y = blockIdx.y;

    int range = blockDim.x / 2;

    // make sure we are within bounds of reference frame. TODO: Support partial frame bounds.
    int left = MAX(mb_x * 8 - range, 0);
    int top = MAX(mb_y * 8 - range, 0);
    int right = MIN(mb_x * 8 + range, w - 8);
    int bottom = MIN(mb_y * 8 + range, h - 8);

    int bounds_w = right - left;
    int bounds_h = bottom - top;
    int size = bounds_w * bounds_h;

    int tid = threadIdx.x + threadIdx.y * bounds_w;

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
            abssum += abs(origin[0 + v * w] - reference[0 + v * w]);
            abssum += abs(origin[1 + v * w] - reference[1 + v * w]);
            abssum += abs(origin[2 + v * w] - reference[2 + v * w]);
            abssum += abs(origin[3 + v * w] - reference[3 + v * w]);
            abssum += abs(origin[4 + v * w] - reference[4 + v * w]);
            abssum += abs(origin[5 + v * w] - reference[5 + v * w]);
            abssum += abs(origin[6 + v * w] - reference[6 + v * w]);
            abssum += abs(origin[7 + v * w] - reference[7 + v * w]);
        }
        sad_grid[tid] = abssum;
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
        int best_i = best_i_list[0];
        for (int i = 1; i < reduction_num; ++i)
        {
            if (sad_grid[best_i_list[i]] < sad_grid[best_i])
            {
                best_i = best_i_list[i];
            }
        }

        struct macroblock *mb = &mbs[mb_x + mb_y * w / 8];
        mb->mv_x = left + (best_i % bounds_w) - mx;
        mb->mv_y = top + (best_i / bounds_w) - my;
        mb->use_mv = 1;
    }
}

__host__ void c63_motion_estimate(struct c63_common *cm)
{
    // define block grid
    dim3 block_grid_Y(cm->mb_cols, cm->mb_rows, 1);
    dim3 block_grid_UV(cm->mb_cols / 2, cm->mb_rows / 2, 1);
    
    // define thread grid
    dim3 thread_grid_Y(cm->me_search_range * 2, cm->me_search_range * 2, 1);
    dim3 thread_grid_UV(cm->me_search_range, cm->me_search_range, 1);

    // define streams
    cudaStream_t Ystream, Ustream, Vstream;
    cudaStreamCreate(&Ystream);
    cudaStreamCreate(&Ustream);
    cudaStreamCreate(&Vstream);

    // TODO: do something to properly use streams

    // Luma
    me_block_8x8 <<<block_grid_Y, thread_grid_Y, 0, Ystream>>> (
        cm->ypw, cm->yph,
        cm->curframe->orig->Y, cm->ref_recons->Y,
        cm->curframe->mbs[Y_COMPONENT]);
    cudaStreamSynchronize(Ystream);
        
    // Chroma U
    me_block_8x8 <<<block_grid_UV, thread_grid_UV, 0, Ustream>>> (
        cm->upw, cm->uph,
        cm->curframe->orig->U, cm->ref_recons->U,
        cm->curframe->mbs[U_COMPONENT]);
    cudaStreamSynchronize(Ustream);

    // Chroma V
    me_block_8x8 <<<block_grid_UV, thread_grid_UV, 0, Vstream>>> (
        cm->vpw, cm->vph,
        cm->curframe->orig->V, cm->ref_recons->V,
        cm->curframe->mbs[V_COMPONENT]);
    cudaStreamSynchronize(Vstream);
}
