#pragma once

// Declaration
#ifdef __cplusplus
extern "C"
{
#endif
    struct gpu_frame *gpu_init(struct c63_common *cm);
    void gpu_cleanup(struct gpu_frame *gpu_frame);
#ifdef __cplusplus
}
#endif

