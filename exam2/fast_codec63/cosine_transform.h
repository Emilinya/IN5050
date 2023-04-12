#pragma once

#include "c63.h"

#include <inttypes.h>

#define ISQRT2 0.70710678118654f

#ifdef __cplusplus
extern "C"
{
#endif
    void dct_quantize(
        uint8_t *in_data, uint8_t *prediction, uint32_t width, uint32_t height,
        int16_t *out_data, uint8_t *quantization, struct c63_common *cm);

    void dequantize_idct(
        int16_t *in_data, uint8_t *prediction, uint32_t width, uint32_t height,
        uint8_t *out_data, uint8_t *quantization, struct c63_common *cm);
#ifdef __cplusplus
}
#endif

