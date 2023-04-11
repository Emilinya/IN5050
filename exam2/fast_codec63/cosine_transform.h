#pragma once

#define ISQRT2 0.70710678118654f

#include <inttypes.h>

void dct_quantize(
    uint8_t *in_data, uint8_t *prediction, uint32_t width,
    uint32_t height, int16_t *out_data, uint8_t *quantization);

void dequantize_idct(
    int16_t *in_data, uint8_t *prediction, uint32_t width,
    uint32_t height, uint8_t *out_data, uint8_t *quantization);
