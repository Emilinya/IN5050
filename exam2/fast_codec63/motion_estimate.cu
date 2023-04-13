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
    int *ws, int *hs, yuv_t *origs,
    yuv_t *refs, struct macroblock *mbss[COLOR_COMPONENTS])
{
    // find out which color component we are
    int w = ws[blockIdx.z];
    int h = hs[blockIdx.z];
    struct macroblock *mbs = mbss[blockIdx.z];

    uint8_t *orig, *ref;
    if (blockIdx.z == 0) {
        orig = origs->Y;
        ref = refs->Y;
    } else if (blockIdx.z == 1) {
        orig = origs->U;
        ref = refs->U;
    } else {
        orig = origs->V;
        ref = refs->V;
    }

    int mb_x = blockIdx.x;
    int mb_y = blockIdx.y;

    int range = blockDim.x / 2;
    if (blockIdx.z != 0) {
        range /= 2;

        // gridDim is too large for U and V components
        if (!(mb_x < gridDim.x / 2 && mb_y < gridDim.y / 2)) {
            return;
        }
    }

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
    } else {
        return;
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

        // if (blockIdx.x == 5 && blockIdx.y == 5 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        //     printf("%d, %d, %d, %d, %d\n", size, best_i, sad_grid[best_i], mb->mv_x, mb->mv_y);
        // }
    }
}

__host__ void c63_motion_estimate(struct c63_common *cm)
{
    // define block grid
    dim3 block_grid(cm->mb_cols, cm->mb_rows, 3);
    
    // define thread grid
    int range = cm->me_search_range * 2;
    dim3 thread_grid(range, range, 1);

    // is this really neccesary?
    int *ws, *hs;
    cudaMallocErr(ws, 3*sizeof(int));
    cudaMallocErr(hs, 3*sizeof(int));

    memcpy(ws, cm->padw, 3*sizeof(int));
    memcpy(hs, cm->padh, 3*sizeof(int));

    // printf("\n");
    me_block_8x8 <<<block_grid, thread_grid>>> (
        ws, hs, cm->curframe->orig, cm->ref_recons, cm->curframe->mbs);
    cudaDeviceSynchronize();

    cudaFree(ws);
    cudaFree(hs);
}
