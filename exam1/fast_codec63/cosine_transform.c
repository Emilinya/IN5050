#include <math.h>
#include <stdlib.h>
#include <inttypes.h>
#include <arm_neon.h>

#include "cosine_transform.h"
#include "tables.h"

#include <stdio.h>

static void transpose_block(float *in_data, float *out_data)
{
  int i, j;

  for (i = 0; i < 8; ++i)
  {
    for (j = 0; j < 8; ++j)
    {
      out_data[i * 8 + j] = in_data[j * 8 + i];
    }
  }
}

__attribute__((always_inline)) static void dct_1d(float *in_data, float *out_data)
{
  float32x4_t out_data_1_v = vdupq_n_f32(0.);
  float32x4_t out_data_2_v = vdupq_n_f32(0.);

  for (int i = 0; i < 8; ++i)
  {
    float32x4_t dctlookup_1_v = vld1q_f32(dctlookup[i]);
    float32x4_t row_prod_1 = vmulq_n_f32(dctlookup_1_v, in_data[i]);
    out_data_1_v = vaddq_f32(out_data_1_v, row_prod_1);

    float32x4_t dctlookup_2_v = vld1q_f32(&dctlookup[i][4]);
    float32x4_t row_prod_2 = vmulq_n_f32(dctlookup_2_v, in_data[i]);
    out_data_2_v = vaddq_f32(out_data_2_v, row_prod_2);
  }

  vst1q_f32(out_data, out_data_1_v);
  vst1q_f32(&out_data[4], out_data_2_v);
}

__attribute__((always_inline)) static void idct_1d(float *in_data, float *out_data)
{
  float32x4_t in_data_1_v = vld1q_f32(in_data);
  float32x4_t in_data_2_v = vld1q_f32(&in_data[4]);

  for (int i = 0; i < 8; ++i)
  {
    float32x4_t dctlookup_1_v = vld1q_f32(dctlookup[i]);
    float32x4_t dctlookup_2_v = vld1q_f32(&dctlookup[i][4]);

    float32x4_t dct_1_v = vmulq_f32(in_data_1_v, dctlookup_1_v);
    float32x4_t dct_2_v = vmulq_f32(in_data_2_v, dctlookup_2_v);
    float32x4_t dct_v = vaddq_f32(dct_1_v, dct_2_v);

    out_data[i] = vaddvq_f32(dct_v);
  }
}

static void scale_block(float *in_data, float *out_data)
{
  int u, v;

  for (v = 0; v < 8; ++v)
  {
    for (u = 0; u < 8; ++u)
    {
      float a1 = !u ? ISQRT2 : 1.0f;
      float a2 = !v ? ISQRT2 : 1.0f;

      /* Scale according to normalizing function */
      out_data[v * 8 + u] = in_data[v * 8 + u] * a1 * a2;
    }
  }
}

static void quantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
{
  int zigzag;

  for (zigzag = 0; zigzag < 64; ++zigzag)
  {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    float dct = in_data[v * 8 + u];

    /* Zig-zag and quantize */
    out_data[zigzag] = (float)round((dct / 4.0) / quant_tbl[zigzag]);
  }
}

static void dequantize_block(float *in_data, float *out_data,
                             uint8_t *quant_tbl)
{
  int zigzag;

  for (zigzag = 0; zigzag < 64; ++zigzag)
  {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    float dct = in_data[zigzag];

    /* Zig-zag and de-quantize */
    out_data[v * 8 + u] = (float)round((dct * quant_tbl[zigzag]) / 4.0);
  }
}

void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data,
                         uint8_t *quant_tbl)
{
  float mb[8 * 8] __attribute((aligned(16)));
  float mb2[8 * 8] __attribute((aligned(16)));

  int i, v;

  for (i = 0; i < 64; ++i)
  {
    mb2[i] = in_data[i];
  }

  /* Two 1D DCT operations with transpose */
  for (v = 0; v < 8; ++v)
  {
    dct_1d(mb2 + v * 8, mb + v * 8);
  }
  transpose_block(mb, mb2);
  for (v = 0; v < 8; ++v)
  {
    dct_1d(mb2 + v * 8, mb + v * 8);
  }
  transpose_block(mb, mb2);

  scale_block(mb2, mb);
  quantize_block(mb, mb2, quant_tbl);

  for (i = 0; i < 64; ++i)
  {
    out_data[i] = mb2[i];
  }
}

void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data,
                            uint8_t *quant_tbl)
{
  float mb[8 * 8] __attribute((aligned(16)));
  float mb2[8 * 8] __attribute((aligned(16)));

  int i, v;

  for (i = 0; i < 64; ++i)
  {
    mb[i] = in_data[i];
  }

  dequantize_block(mb, mb2, quant_tbl);
  scale_block(mb2, mb);

  /* Two 1D inverse DCT operations with transpose */
  for (v = 0; v < 8; ++v)
  {
    idct_1d(mb + v * 8, mb2 + v * 8);
  }
  transpose_block(mb2, mb);
  for (v = 0; v < 8; ++v)
  {
    idct_1d(mb + v * 8, mb2 + v * 8);
  }
  transpose_block(mb2, mb);

  for (i = 0; i < 64; ++i)
  {
    out_data[i] = mb[i];
  }
}

void dequantize_idct_row(int16_t *in_data, uint8_t *prediction, int w, int h,
                         int y, uint8_t *out_data, uint8_t *quantization)
{
  int x;

  int16_t block[8 * 8];

  /* Perform the dequantization and iDCT */
  for (x = 0; x < w; x += 8)
  {
    int i, j;

    dequant_idct_block_8x8(in_data + (x * 8), block, quantization);

    for (i = 0; i < 8; ++i)
    {
      for (j = 0; j < 8; ++j)
      {
        /* Add prediction block. Note: DCT is not precise -
           Clamp to legal values */
        int16_t tmp = block[i * 8 + j] + (int16_t)prediction[i * w + j + x];

        if (tmp < 0)
        {
          tmp = 0;
        }
        else if (tmp > 255)
        {
          tmp = 255;
        }

        out_data[i * w + j + x] = tmp;
      }
    }
  }
}

void dequantize_idct(int16_t *in_data, uint8_t *prediction, uint32_t width,
                     uint32_t height, uint8_t *out_data, uint8_t *quantization)
{
  int y;

  for (y = 0; y < height; y += 8)
  {
    dequantize_idct_row(in_data + y * width, prediction + y * width, width, height, y,
                        out_data + y * width, quantization);
  }
}

void dct_quantize_row(uint8_t *in_data, uint8_t *prediction, int w, int h,
                      int16_t *out_data, uint8_t *quantization)
{
  int x;

  int16_t block[8 * 8];

  /* Perform the DCT and quantization */
  for (x = 0; x < w; x += 8)
  {
    int i, j;

    for (i = 0; i < 8; ++i)
    {
      for (j = 0; j < 8; ++j)
      {
        block[i * 8 + j] = ((int16_t)in_data[i * w + j + x] - prediction[i * w + j + x]);
      }
    }

    /* Store MBs linear in memory, i.e. the 64 coefficients are stored
       continous. This allows us to ignore stride in DCT/iDCT and other
       functions. */
    dct_quant_block_8x8(block, out_data + (x * 8), quantization);
  }
}

void dct_quantize(uint8_t *in_data, uint8_t *prediction, uint32_t width,
                  uint32_t height, int16_t *out_data, uint8_t *quantization)
{
  int y;

  for (y = 0; y < height; y += 8)
  {
    dct_quantize_row(in_data + y * width, prediction + y * width, width, height,
                     out_data + y * width, quantization);
  }
}
