#pragma once

#include "c63.h"

// Declaration
#ifdef __cplusplus
extern "C"
{
#endif
    void c63_motion_estimate(struct c63_common *cm, struct gpu_frame *gpu_frame);
#ifdef __cplusplus
}
#endif

void c63_motion_compensate(struct c63_common *cm);
