#include <inttypes.h>
#include <math.h>
#include <stdlib.h>

#include "cosine_transform.h"
#include "tables.h"
#include "utils.h"

__global__ void dequant_idct_block_8x8(
  int w, int16_t *in_data, uint8_t *prediction, uint8_t *out_data,
  uint8_t *quant_tbl, float *dctlookup, uint8_t *zigzag_U, uint8_t *zigzag_V)
{
  int v;

  int x = threadIdx.x;
  int y = threadIdx.y;
  int tid = x + y * 8;

  int idx = (blockIdx.x + blockIdx.y * w) * 8;
  int16_t *in = in_data + idx + blockIdx.x * 56;
  uint8_t *pred = prediction + idx;
  uint8_t *out = out_data + idx;

  __shared__ float mb[8 * 8];
  __shared__ float mb2[8 * 8];

  // input
  mb[tid] = in[tid];

  // zig-zag and dequantize
  uint8_t zigzag_u = zigzag_U[tid];
  uint8_t zigzag_v = zigzag_V[tid];
  mb2[zigzag_v * 8 + zigzag_u] = (float)round((mb[tid] * quant_tbl[tid]) / 4.0);
  
  __syncthreads();

  // scale according to normalizing function
  float a1 = !x ? ISQRT2 : 1.0f;
  float a2 = !y ? ISQRT2 : 1.0f;
  mb[tid] = mb2[tid] * a1 * a2;

  __syncthreads();

  // idct rows
  float idct = 0;
  for (v = 0; v < 8; ++v)
  {
   idct += mb[v + y * 8] * dctlookup[v + x * 8];
  }
  // set data column vise -> transpose block
  mb2[y + x * 8] = idct;

  __syncthreads();

  // idct columns
  idct = 0;
  for (v = 0; v < 8; ++v)
  {
    idct += mb2[v + y * 8] * dctlookup[v + x * 8];
  }
  // set data column vise - transpose block again
  mb[y + x * 8] = idct;

  __syncthreads();

  // output
  int16_t tmp = (int16_t)mb[tid] + pred[x + y * w];

  // DCT is not precise - clamp to legal values
  out[x + y * w] = MAX(MIN(tmp, 255), 0);
}

__global__ void dct_quant_block_8x8(
    int w, uint8_t *in_data, uint8_t *prediction, int16_t *out_data,
    uint8_t *quant_tbl, float *dctlookup, uint8_t *zigzag_U, uint8_t *zigzag_V)
{
  int v;

  int x = threadIdx.x;
  int y = threadIdx.y;
  int tid = x + y * 8;

  int idx = (blockIdx.x + blockIdx.y * w) * 8;
  uint8_t *in = in_data + idx;
  uint8_t *pred = prediction + idx;
  int16_t *out = out_data + idx + blockIdx.x * 56;

  /* Store MBs linear in memory, i.e. the 64 coefficients are stored
     continous. This allows us to ignore stride in DCT/iDCT and other
     functions. */
  __shared__ float mb[8 * 8];
  __shared__ float mb2[8 * 8];

  // input
  mb[tid] = in[x + y * w] - pred[x + y * w];

  // transpose dctlookup to avoid strided lookup
  __shared__ float dctlookup_T[8 * 8];
  dctlookup_T[y + x * 8] = dctlookup[tid];

  __syncthreads();

  // dct rows
  float dct = 0;
  for (v = 0; v < 8; ++v)
  {
    dct += mb[v + y * 8] * dctlookup_T[v + x * 8];
  }
  // set data column vise -> transpose block
  mb2[y + x * 8] = dct;

  __syncthreads();

  // dct columns
  dct = 0;
  for (v = 0; v < 8; ++v)
  {
    dct += mb2[y * 8 + v] * dctlookup[x + v * 8];
  }
  // set data column vise -> transpose block again
  mb[y + x * 8] = dct;

  __syncthreads();

  // scale according to normalizing function
  float a1 = !x ? ISQRT2 : 1.0f;
  float a2 = !y ? ISQRT2 : 1.0f;
  mb2[tid] = mb[tid] * a1 * a2;

  __syncthreads();

  // zig-zag and quantize
  uint8_t zigzag_u = zigzag_U[tid];
  uint8_t zigzag_v = zigzag_V[tid];
  mb[tid] = (float)round((mb2[zigzag_u + zigzag_v * 8] / 4.0) / quant_tbl[tid]);

  // output
  out[tid] = mb[tid];
}

__host__ void dequantize_idct(
  int16_t *in_data, uint8_t *prediction, uint32_t width, uint32_t height,
  uint8_t *out_data, uint8_t *quantization, struct c63_common *cm)
{
  // define block grid
  dim3 block_grid;
  block_grid.x = width / 8;
  block_grid.y = height / 8;
    
  // define thread grid
  dim3 thread_grid;
  thread_grid.x = 8;
  thread_grid.y = 8;

  dequant_idct_block_8x8 <<<block_grid, thread_grid>>> (
    width, in_data, prediction, out_data, quantization,
    cm->dctlookup, cm->zigzag_U, cm->zigzag_V);
  cudaDeviceSynchronize();
}

__host__ void dct_quantize(
  uint8_t *in_data, uint8_t *prediction, uint32_t width, uint32_t height,
  int16_t *out_data, uint8_t *quantization, struct c63_common *cm)
{
  // define block grid
  dim3 block_grid;
  block_grid.x = width / 8;
  block_grid.y = height / 8;
    
  // define thread grid
  dim3 thread_grid;
  thread_grid.x = 8;
  thread_grid.y = 8;

  dct_quant_block_8x8 <<<block_grid, thread_grid>>> (
    width, in_data, prediction, out_data, quantization,
    cm->dctlookup, cm->zigzag_U, cm->zigzag_V);
  cudaDeviceSynchronize();
}
