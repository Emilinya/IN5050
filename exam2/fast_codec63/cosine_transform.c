#include <inttypes.h>
#include <math.h>
#include <stdlib.h>

#include "cosine_transform.h"
#include "tables.h"
#include "utils.h"

void dequant_idct_block_8x8(
  int x, int y, int w, int16_t *in_data,
  uint8_t *prediction, uint8_t *out_data, uint8_t *quant_tbl)
{
  int i, j, v;

  float mb[8 * 8] __attribute((aligned(16)));
  float mb2[8 * 8] __attribute((aligned(16)));

  int idx = x * 8 + y * 8 * w;
  int16_t *in = in_data + idx + 7 * x * 8;
  uint8_t *pred = prediction + idx;
  uint8_t *out = out_data + idx;

  // input
  for (i = 0; i < 64; ++i)
  {
    mb[i] = in[i];
  }

  // dequantize
  for (i = 0; i < 64; ++i)
  {
    uint8_t u = zigzag_U[i];
    uint8_t v = zigzag_V[i];

    float dct = mb[i];

    /* Zig-zag and de-quantize */
    mb2[v * 8 + u] = (float)round((dct * quant_tbl[i]) / 4.0);
  }

  // scale
  for (i = 0; i < 8; ++i)
  {
    for (j = 0; j < 8; ++j)
    {
      float a1 = !j ? ISQRT2 : 1.0f;
      float a2 = !i ? ISQRT2 : 1.0f;

      /* Scale according to normalizing function */
      mb[i * 8 + j] = mb2[i * 8 + j] * a1 * a2;
    }
  }

  // idct rows
  for (i = 0; i < 8; ++i)
  {
    for (j = 0; j < 8; ++j)
    {
      float idct = 0;

      for (v = 0; v < 8; ++v)
      {
        idct += mb[i * 8 + v] * dctlookup[j][v];
      }

      // set data column vise - transpose block
      mb2[i + j * 8] = idct;
    }
  }

  // idct columns
  for (i = 0; i < 8; ++i)
  {
    for (j = 0; j < 8; ++j)
    {
      float idct = 0;

      for (v = 0; v < 8; ++v)
      {
        idct += mb2[i * 8 + v] * dctlookup[j][v];
      }

      // set data column vise - transpose block again
      mb[i + j * 8] = idct;
    }
  }

  // output
  for (i = 0; i < 8; ++i)
  {
    for (j = 0; j < 8; ++j)
    {
      int16_t tmp = (int16_t)mb[i * 8 + j] + pred[i * w + j];

      // DCT is not precise - Clamp to legal values
      out[i * w + j] = MAX(MIN(tmp, 255), 0);
    }
  }
}

void dequantize_idct(int16_t *in_data, uint8_t *prediction, uint32_t width,
                     uint32_t height, uint8_t *out_data, uint8_t *quantization)
{
  int x, y;

  int ho8 = height / 8; // "height over 8"
  int wo8 = width / 8;  // "width over 8"

  for (y = 0; y < ho8; ++y)
  {
    for (x = 0; x < wo8; ++x)
    {
      dequant_idct_block_8x8(x, y, width, in_data, prediction, out_data, quantization);
    }
  }
}


void dct_quant_block_8x8(
    int x, int y, int w, uint8_t *in_data,
    uint8_t *prediction, int16_t *out_data, uint8_t *quant_tbl)
{
  int i, j, v;

  int idx = (x + y * w) * 8;
  uint8_t *in = in_data + idx;
  uint8_t *pred = prediction + idx;
  int16_t *out = out_data + idx + x * 56;

  /* Store MBs linear in memory, i.e. the 64 coefficients are stored
     continous. This allows us to ignore stride in DCT/iDCT and other
     functions. */
  float mb[8 * 8] __attribute((aligned(16)));
  float mb2[8 * 8] __attribute((aligned(16)));

  // input
  for (i = 0; i < 8; ++i)
  {
    for (j = 0; j < 8; ++j)
    {
      mb[i * 8 + j] = in[i * w + j] - pred[i * w + j];
    }
  }

  // dct rows
  for (i = 0; i < 8; ++i)
  {
    for (j = 0; j < 8; ++j)
    {
      float dct = 0;

      for (v = 0; v < 8; ++v)
      {
        dct += mb[i * 8 + v] * dctlookup[v][j];
      }

      // set data column vise - transpose block
      mb2[i + j * 8] = dct;
    }
  }

  // dct columns
  for (i = 0; i < 8; ++i)
  {
    for (j = 0; j < 8; ++j)
    {
      float dct = 0;

      for (v = 0; v < 8; ++v)
      {
        dct += mb2[i * 8 + v] * dctlookup[v][j];
      }

      // set data column vise - transpose block again
      mb[i + j * 8] = dct;
    }
  }

  // scale
  for (i = 0; i < 8; ++i)
  {
    for (j = 0; j < 8; ++j)
    {
      float a1 = !j ? ISQRT2 : 1.0f;
      float a2 = !i ? ISQRT2 : 1.0f;

      /* Scale according to normalizing function */
      mb2[i * 8 + j] = mb[i * 8 + j] * a1 * a2;
    }
  }

  // quantize
  for (int i = 0; i < 64; ++i)
  {
    uint8_t u = zigzag_U[i];
    uint8_t v = zigzag_V[i];

    float dct = mb2[v * 8 + u];

    /* Zig-zag and quantize */
    mb[i] = (float)round((dct / 4.0) / quant_tbl[i]);
  }

  // output
  for (i = 0; i < 64; ++i)
  {
    out[i] = mb[i];
  }
}

void dct_quantize(uint8_t *in_data, uint8_t *prediction, uint32_t width,
                  uint32_t height, int16_t *out_data, uint8_t *quantization)
{
  int ho8 = height / 8; // "height over 8"
  int wo8 = width / 8;  // "width over 8"

  for (int y = 0; y < ho8; ++y)
  {
    for (int x = 0; x < wo8; ++x)
    {
      dct_quant_block_8x8(x, y, width, in_data, prediction, out_data, quantization);
    }
  }
}
