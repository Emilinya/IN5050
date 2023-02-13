#include <math.h>
#include <stdlib.h>
#include <inttypes.h>
#include <arm_neon.h>

#include "cosine_transform.h"
#include "tables.h"

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

static void dct_1d(float *in_data, float *out_data)
{
  int i, j;

  for (i = 0; i < 8; ++i)
  {
    float dct = 0;

    for (j = 0; j < 8; ++j)
    {
      dct += in_data[j] * dctlookup[j][i];
    }

    out_data[i] = dct;
  }
}

static void idct_1d(float *in_data, float *out_data)
{
  int i, j;

  for (i = 0; i < 8; ++i)
  {
    float idct = 0;

    for (j = 0; j < 8; ++j)
    {
      idct += in_data[j] * dctlookup[i][j];
    }

    out_data[i] = idct;
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

void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
{
  // load block 1

  uint8x8_t block1_1_1 = vld1_u8(&block1[0]);
  uint8x8_t block1_1_2 = vld1_u8(&block1[stride]);
  uint8x16_t block1_1 = vcombine_u8(block1_1_1, block1_1_2);

  uint8x8_t block1_2_1 = vld1_u8(&block1[2 * stride]);
  uint8x8_t block1_2_2 = vld1_u8(&block1[3 * stride]);
  uint8x16_t block1_2 = vcombine_u8(block1_2_1, block1_2_2);

  uint8x8_t block1_3_1 = vld1_u8(&block1[4 * stride]);
  uint8x8_t block1_3_2 = vld1_u8(&block1[5 * stride]);
  uint8x16_t block1_3 = vcombine_u8(block1_3_1, block1_3_2);

  uint8x8_t block1_4_1 = vld1_u8(&block1[6 * stride]);
  uint8x8_t block1_4_2 = vld1_u8(&block1[7 * stride]);
  uint8x16_t block1_4 = vcombine_u8(block1_4_1, block1_4_2);

  // load block 2

  uint8x8_t block2_1_1 = vld1_u8(&block2[0]);
  uint8x8_t block2_1_2 = vld1_u8(&block2[stride]);
  uint8x16_t block2_1 = vcombine_u8(block2_1_1, block2_1_2);

  uint8x8_t block2_2_1 = vld1_u8(&block2[2 * stride]);
  uint8x8_t block2_2_2 = vld1_u8(&block2[3 * stride]);
  uint8x16_t block2_2 = vcombine_u8(block2_2_1, block2_2_2);

  uint8x8_t block2_3_1 = vld1_u8(&block2[4 * stride]);
  uint8x8_t block2_3_2 = vld1_u8(&block2[5 * stride]);
  uint8x16_t block2_3 = vcombine_u8(block2_3_1, block2_3_2);

  uint8x8_t block2_4_1 = vld1_u8(&block2[6 * stride]);
  uint8x8_t block2_4_2 = vld1_u8(&block2[7 * stride]);
  uint8x16_t block2_4 = vcombine_u8(block2_4_1, block2_4_2);

  // calculate absolute difference

  uint8x16_t abdiff_1 = vabdq_u8(block2_1, block1_1);
  uint8x16_t abdiff_2 = vabdq_u8(block2_2, block1_2);
  uint8x16_t abdiff_3 = vabdq_u8(block2_3, block1_3);
  uint8x16_t abdiff_4 = vabdq_u8(block2_4, block1_4);

  // first round -> 64 values to 32 values, use addl to get uint16 to avoid overflow

  uint8x8_t abdiff_1_1 = vget_low_u8(abdiff_1);
  uint8x8_t abdiff_1_2 = vget_high_u8(abdiff_1);
  uint16x8_t absdiffsum_1_1 = vaddl_u8(abdiff_1_1, abdiff_1_2);

  uint8x8_t abdiff_2_1 = vget_low_u8(abdiff_2);
  uint8x8_t abdiff_2_2 = vget_high_u8(abdiff_2);
  uint16x8_t absdiffsum_1_2 = vaddl_u8(abdiff_2_1, abdiff_2_2);

  uint8x8_t abdiff_3_1 = vget_low_u8(abdiff_3);
  uint8x8_t abdiff_3_2 = vget_high_u8(abdiff_3);
  uint16x8_t absdiffsum_1_3 = vaddl_u8(abdiff_3_1, abdiff_3_2);

  uint8x8_t abdiff_4_1 = vget_low_u8(abdiff_4);
  uint8x8_t abdiff_4_2 = vget_high_u8(abdiff_4);
  uint16x8_t absdiffsum_1_4 = vaddl_u8(abdiff_4_1, abdiff_4_2);

  // second round -> 32 values to 16 values

  uint16x8_t absdiffsum_2_1 = vaddq_u16(absdiffsum_1_1, absdiffsum_1_2);
  uint16x8_t absdiffsum_2_2 = vaddq_u16(absdiffsum_1_3, absdiffsum_1_4);

  // third round -> 16 values to 8 values

  uint16x8_t absdiffsum_3 = vaddq_u16(absdiffsum_2_1, absdiffsum_2_2);

  // fourth round -> 8 values to 4 values

  uint16x4_t absdiffsum_3_1 = vget_low_u16(absdiffsum_3);
  uint16x4_t absdiffsum_3_2 = vget_high_u16(absdiffsum_3);
  uint16x4_t absdiffsum_4 = vadd_u16(absdiffsum_3_1, absdiffsum_3_2);

  // finally add the remaining 4 values together

  uint16_t vals[4];
  vst1_u16(vals, absdiffsum_4);

  *result = vals[0] + vals[1] + vals[2] + vals[3];
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
